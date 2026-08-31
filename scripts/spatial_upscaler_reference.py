"""Reference intermediates for the latent spatial upscaler (CPU float32).

Sub-task 6 of issue #57's breakdown (sub-tasks 1-5: video VAE decoder/encoder,
text connector, audio VAE + vocoder, dual-stream transformer — PRs #76, #77,
#78, #80, #82). `ltx_core.model.upsampler.model.LatentUpsampler` is config-
driven (spatial_upsample/temporal_upsample/rational_resampler flags read from
the checkpoint), the same class `temporal_upscaler_reference.py` already
covers — this script is its spatial-mode sibling, extended with per-stage
taps: the resampler is exactly where a dimension-order bug (which axis pairs
with which pixel-shuffle factor) can be wrong without changing the output
*shape* at all, per docs/knowledge/pitfalls/conv-decoder-wrong-spatial-padding.md's
sub-task-1 precedent.

    PYTHONPATH=<ltx-core>/src python3 scripts/spatial_upscaler_reference.py <ckpt> out.safetensors
"""
import json
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

from ltx_core.model.upsampler.model import LatentUpsampler

CKPT, OUT = sys.argv[1], sys.argv[2]
with open(CKPT, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
cfg = json.loads(header["__metadata__"]["config"])
print("config:", {k: v for k, v in cfg.items() if k != "_class_name"})

model = LatentUpsampler(
    in_channels=cfg["in_channels"], mid_channels=cfg["mid_channels"],
    num_blocks_per_stage=cfg["num_blocks_per_stage"], dims=cfg["dims"],
    spatial_upsample=cfg["spatial_upsample"], temporal_upsample=cfg["temporal_upsample"],
    spatial_scale=cfg.get("spatial_scale", 2.0),
    rational_resampler=cfg.get("rational_resampler", False),
)
missing, unexpected = model.load_state_dict(
    {k: v.to(torch.float32) for k, v in load_file(CKPT).items()}, strict=False)
print("MISSING:", len(missing), missing[:4], "UNEXPECTED:", len(unexpected), unexpected[:4])
assert not missing, "reference upsampler load incomplete — comparison would be meaningless"
model.eval().to(torch.float32)

torch.manual_seed(0)
# Batch=2, not 1: the resampler tap folds (batch, frames) into one axis on the
# Swift side to compare against this per-frame (batch*frames, ...) tap — with
# batch=1 that fold is bit-identical to a swapped fold order, so it can't
# actually catch an N/D mix-up. Batch=2 makes the two orderings diverge.
latent = torch.randn(2, cfg["in_channels"], 3, 8, 8, dtype=torch.float32)

intermediates = {}
def capture(name):
    def hook(_module, _inputs, output):
        intermediates[name] = output.detach().float().clone().contiguous()
    return hook

model.initial_conv.register_forward_hook(capture("initial_conv"))
for i, block in enumerate(model.res_blocks):
    block.register_forward_hook(capture(f"res_block{i}"))
model.upsampler.register_forward_hook(capture("upsampler"))
for i, block in enumerate(model.post_upsample_res_blocks):
    block.register_forward_hook(capture(f"post_res_block{i}"))
model.final_conv.register_forward_hook(capture("final_conv"))

with torch.no_grad():
    out = model(latent)
print("REF out", tuple(out.shape), "mean|x|", out.abs().mean().item())

dump = {"latent": latent.clone(), "output": out.detach().clone().contiguous()}
dump.update(intermediates)
save_file(dump, OUT)
print("dumped", OUT)
