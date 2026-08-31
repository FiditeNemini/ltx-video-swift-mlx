"""Reference intermediates for the real audio decode chain: AudioVAE decoder
+ BigVGAN vocoder with bandwidth extension.

This is `ltx_core.model.audio_vae.audio_vae.decode_audio`'s actual production
path: audio latent -> AudioDecoder -> mel spectrogram -> VocoderWithBWE ->
48 kHz waveform. Runs Lightricks' own modules on CPU float32 (the vocoder's
own docstring: bf16 accumulation across ~108 sequential convolutions degrades
spectral metrics 40-90%) over a fixed latent, dumping every tap our Swift
port must reproduce.

Both AudioDecoder and VocoderConfigurator.from_metadata auto-detect their
shape entirely from the checkpoint's own config block, so this script needs
no hardcoded architecture constants — unlike the video-side reference
scripts, which do (LTX-2.5's video config has no equivalent auto-detecting
configurator).

    PYTHONPATH=<ltx-core>/src python3 scripts/audio_vae_reference.py <ltx-2.5-audio-vae-bf16.safetensors> out.safetensors
"""
import json
import struct
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ltx_core.model.audio_vae.model_configurator import AudioDecoderConfigurator, VocoderConfigurator

CKPT = sys.argv[1]
OUT = sys.argv[2]

with open(CKPT, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
metadata = {"config": json.loads(header["__metadata__"]["config"])}

decoder = AudioDecoderConfigurator.from_metadata(metadata)
decoder.eval().to(torch.float32)
vocoder = VocoderConfigurator.from_metadata(metadata)
vocoder.eval().to(torch.float32)

raw = load_file(CKPT)

decoder_sd = {}
for k, v in raw.items():
    if k.startswith("audio_vae.decoder."):
        decoder_sd[k[len("audio_vae.decoder."):]] = v.to(torch.float32)
    elif k.startswith("audio_vae.per_channel_statistics."):
        decoder_sd["per_channel_statistics." + k[len("audio_vae.per_channel_statistics."):]] = v.to(torch.float32)
dec_missing, dec_unexpected = decoder.load_state_dict(decoder_sd, strict=False)
print("DECODER MISSING:", len(dec_missing), dec_missing[:6])
print("DECODER UNEXPECTED:", len(dec_unexpected), dec_unexpected[:6])
assert not dec_missing, "reference decoder load incomplete — comparison would be meaningless"

vocoder_sd = {}
for k, v in raw.items():
    if k.startswith("vocoder."):
        vocoder_sd[k[len("vocoder."):]] = v.to(torch.float32)
voc_missing, voc_unexpected = vocoder.load_state_dict(vocoder_sd, strict=False)
print("VOCODER MISSING:", len(voc_missing), voc_missing[:6])
print("VOCODER UNEXPECTED:", len(voc_unexpected), voc_unexpected[:6])
assert not voc_missing, "reference vocoder load incomplete — comparison would be meaningless"

torch.manual_seed(0)
# (B, z_channels=8, T_latent, mel_bins/4=16). T_latent=3 -> 3*4-3=9 mel frames,
# small but exercises every up-level and both vocoder stages.
latent = torch.randn(1, 8, 3, 16, dtype=torch.float32) * 0.5

taps = {}
def tap(name):
    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        taps[name] = t.detach().float().clone().contiguous()
    return hook

decoder.conv_in.register_forward_hook(tap("decoder_conv_in"))
decoder.mid.block_2.register_forward_hook(tap("decoder_mid"))
decoder.up[2].upsample.register_forward_hook(tap("decoder_up2"))
decoder.up[1].upsample.register_forward_hook(tap("decoder_up1"))
decoder.up[0].block[-1].register_forward_hook(tap("decoder_up0"))
decoder.conv_out.register_forward_hook(tap("decoder_out"))

vocoder.vocoder.register_forward_hook(tap("vocoder_base"))
vocoder.bwe_generator.register_forward_hook(tap("vocoder_bwe_residual"))
vocoder.resampler.register_forward_hook(tap("vocoder_skip"))
# MelSTFT has no forward() (_compute_mel calls .mel_spectrogram() directly,
# bypassing __call__), so a forward_hook on it never fires — call it directly
# on the padded base output instead, duplicating what _compute_mel does.

with torch.no_grad():
    mel = decoder(latent)
    waveform = vocoder(mel)

    low = taps["vocoder_base"]
    remainder = low.shape[-1] % vocoder.hop_length
    low_padded = F.pad(low, (0, vocoder.hop_length - remainder)) if remainder != 0 else low
    mel_for_bwe = vocoder._compute_mel(low_padded).transpose(2, 3)  # (B, C, T_frames, mel_bins), as fed to bwe_generator
    taps["vocoder_mel_for_bwe"] = mel_for_bwe.detach().float().clone().contiguous()

print("REF mel", tuple(mel.shape), "mean|x|", mel.abs().mean().item())
print("REF waveform", tuple(waveform.shape), "mean|x|", waveform.abs().mean().item(),
      "peak", waveform.abs().max().item())
for k, v in sorted(taps.items()):
    print(f"REF tap {k} {tuple(v.shape)} mean|x| {v.abs().mean().item():.5f}")

dump = {"latent": latent.clone(), "mel": mel.detach().clone().contiguous(),
        "waveform": waveform.detach().clone().contiguous()}
dump.update(taps)
save_file(dump, OUT)
print("dumped", OUT)
