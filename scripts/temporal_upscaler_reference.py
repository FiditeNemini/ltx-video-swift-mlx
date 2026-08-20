"""Reference output for the latent temporal upsampler (CPU float32)."""
import json, struct, sys, torch
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
)
missing, unexpected = model.load_state_dict(
    {k: v.to(torch.float32) for k, v in load_file(CKPT).items()}, strict=False)
print("MISSING:", len(missing), missing[:4], "UNEXPECTED:", len(unexpected), unexpected[:4])
assert not missing
model.eval().to(torch.float32)

torch.manual_seed(0)
latent = torch.randn(1, 128, 3, 8, 8, dtype=torch.float32)
with torch.no_grad():
    out = model(latent)
print("REF out", tuple(out.shape), "mean|x|", out.abs().mean().item())
save_file({"latent": latent.clone(), "output": out.detach().clone().contiguous()}, OUT)
print("dumped", OUT)
