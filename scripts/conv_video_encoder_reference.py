"""Reference intermediates for the convolutional video encoder.

Runs Lightricks' own VideoEncoder on a fixed pixel input and dumps the
tensors our Swift port must reproduce. CPU float32: this is ground truth, not
a benchmark.

Every retake, every i2v conditioning image, and every LipDub video reference
goes through this encoder. Deterministic: no timestep conditioning, no
attention on this checkpoint's config, and encoding uses only the mean of the
latent distribution (no sampling), so there's no RNG to match across
frameworks.

    PYTHONPATH=<ltx-core>/src python3 scripts/conv_video_encoder_reference.py <checkpoint> out.safetensors
"""
import json
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

from ltx_core.model.video_vae.model_configurator import _prepare_video_encoder_kwargs
from ltx_core.model.video_vae.video_vae import VideoEncoder

CKPT = sys.argv[1]
OUT = sys.argv[2]

with open(CKPT, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
config = json.loads(header["__metadata__"]["config"])["vae"]

model = VideoEncoder(**_prepare_video_encoder_kwargs(config))
model.eval().to(torch.float32)

raw = load_file(CKPT)
mapped = {}
for k, v in raw.items():
    if k.startswith("encoder."):
        mapped[k[len("encoder."):]] = v.to(torch.float32)
    elif k.startswith("per_channel_statistics."):
        mapped[k] = v.to(torch.float32)

missing, unexpected = model.load_state_dict(mapped, strict=False)
print("MISSING:", len(missing), missing[:6])
print("UNEXPECTED:", len(unexpected), unexpected[:6])
assert not missing, "reference load incomplete — comparison would be meaningless"

torch.manual_seed(0)
# 8n+1 frames, divisible-by-32 spatial (patch_size=4 then 4 downsample stages
# = /32 total) — small but valid: 9 frames, 32x32 pixels -> (1, 128, 2, 1, 1) latent.
pixels = torch.rand(1, 3, 9, 32, 32, dtype=torch.float32) * 2 - 1  # [-1, 1]

taps = {}
def tap(name):
    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        taps[name] = t.detach().float().clone().contiguous()
    return hook

model.conv_in.register_forward_hook(tap("conv_in"))
for i, block in enumerate(model.down_blocks):
    block.register_forward_hook(tap(f"down_blocks_{i}"))

# conv_out's raw 129-channel output, before the mean/logvar split — this is
# what the Swift port's callAsFunction returns directly (it normalizes
# outside the encoder, reusing the decoder's already-loaded per-channel
# statistics — see LTXPipeline.encodeVideo). Comparing against this, not
# forward()'s final per_channel_statistics.normalize(means), keeps the
# comparison at the same module boundary as the Swift port.
raw_means = {}
def tap_raw_means(_m, _inp, out):
    raw_means["value"] = out[:, :128, ...].detach().float().clone().contiguous()
model.conv_out.register_forward_hook(tap_raw_means)
model.conv_out.register_forward_hook(tap("conv_out"))

with torch.no_grad():
    out = model(pixels)

print("REF output (normalized means)", tuple(out.shape), "mean|x|", out.abs().mean().item(), "std", out.std().item())
print("REF raw_means (pre-normalize)", tuple(raw_means["value"].shape),
      "mean|x|", raw_means["value"].abs().mean().item())
dump = {"pixels": pixels.clone(), "output": out.detach().clone().contiguous(),
        "raw_means": raw_means["value"]}
dump.update(taps)
for k, v in sorted(taps.items()):
    print(f"REF tap {k} {tuple(v.shape)} mean|x| {v.abs().mean().item():.5f}")
save_file(dump, OUT)
print("dumped", OUT)
