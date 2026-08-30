"""Reference intermediates for the convolutional video decoder.

Runs Lightricks' own ConvVideoDecoder on a fixed latent and dumps the
tensors our Swift port must reproduce. CPU float32: this is ground truth, not
a benchmark.

This is the *default* decode path — every clip this repo has ever produced
went through it (the diffusion decoder, already covered by
diffvae_reference.py, is opt-in via --diffvae). Unlike the diffusion decoder
it has no attention and no timestep conditioning on the real checkpoint
(config: `timestep_conditioning: false`), so the forward pass is a single
deterministic call — no noise injection, no RNG to match across frameworks.

    PYTHONPATH=<ltx-core>/src python3 scripts/conv_video_decoder_reference.py <checkpoint> out.safetensors
"""
import json
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

from ltx_core.model.video_vae.model_configurator import _build_conv_video_decoder

CKPT = sys.argv[1]
OUT = sys.argv[2]

with open(CKPT, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
config = json.loads(header["__metadata__"]["config"])["vae"]
assert not config.get("timestep_conditioning", False), (
    "checkpoint has timestep_conditioning=true — the decoder injects random "
    "noise per forward pass and this reference (deterministic, no RNG "
    "matching) would not be comparable"
)

model = _build_conv_video_decoder(config)
model.eval().to(torch.float32)

raw = load_file(CKPT)
mapped = {}
for k, v in raw.items():
    if k.startswith("decoder."):
        mapped[k[len("decoder."):]] = v.to(torch.float32)
    elif k.startswith("per_channel_statistics."):
        mapped[k] = v.to(torch.float32)

missing, unexpected = model.load_state_dict(mapped, strict=False)
print("MISSING:", len(missing), missing[:6])
print("UNEXPECTED:", len(unexpected), unexpected[:6])
assert not missing, "reference load incomplete — comparison would be meaningless"

torch.manual_seed(0)
latent = torch.randn(1, 128, 3, 8, 8, dtype=torch.float32)

taps = {}
def tap(name):
    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        taps[name] = t.detach().float().clone().contiguous()
    return hook

model.conv_in.register_forward_hook(tap("conv_in"))
for i, block in enumerate(model.up_blocks):
    block.register_forward_hook(tap(f"up_blocks_{i}"))
model.conv_out.register_forward_hook(tap("conv_out"))

with torch.no_grad():
    out = model(latent)

print("REF output", tuple(out.shape), "mean|x|", out.abs().mean().item(), "std", out.std().item())
dump = {"latent": latent.clone(), "output": out.detach().clone().contiguous()}
dump.update(taps)
for k, v in sorted(taps.items()):
    print(f"REF tap {k} {tuple(v.shape)} mean|x| {v.abs().mean().item():.5f}")
save_file(dump, OUT)
print("dumped", OUT)
