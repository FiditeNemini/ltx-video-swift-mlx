"""Reference intermediates for the diffusion video decoder.

Runs Lightricks' own DiffusionVideoDecoder on a fixed latent and dumps the
tensors our Swift port must reproduce. CPU float32: this is ground truth, not
a benchmark.
"""
import json, sys, torch
from safetensors.torch import load_file, save_file
from ltx_core.model.video_vae.model_configurator import _build_diffusion_video_decoder

CKPT = sys.argv[1]
OUT = sys.argv[2]

with open(CKPT, "rb") as f:
    import struct
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
config = json.loads(header["__metadata__"]["config"])["vae"]

model = _build_diffusion_video_decoder(config)
model.eval().to(torch.float32)

# No NATTEN on this machine (CUDA-only); use the reference's own eager fallback,
# which is exactly what our Swift port mirrors.
from ltx_core.model.video_vae.transformer.fallback_na import fallback_na_attention
eager = fallback_na_attention()
for mod in model.modules():
    if hasattr(mod, "attention_function"):
        mod.attention_function = eager

raw = load_file(CKPT)
mapped = {}
for k, v in raw.items():
    if k.startswith("decoder."):
        mapped[k[len("decoder."):]] = v.to(torch.float32)
    elif k.startswith("per_channel_statistics."):
        mapped[k] = v.to(torch.float32)

# their QKVProjections owns three linears; the checkpoint fuses them
expanded = {}
for k, v in mapped.items():
    if k.endswith("qkv.weight") or k.endswith("qkv.bias"):
        leaf = "weight" if k.endswith(".weight") else "bias"
        prefix = k[: -len(leaf)]
        d = v.shape[0] // 3
        expanded[f"{prefix}to_q.{leaf}"] = v[:d].clone()
        expanded[f"{prefix}to_k.{leaf}"] = v[d:2*d].clone()
        expanded[f"{prefix}to_v.{leaf}"] = v[2*d:].clone()
    elif k.startswith("t_embedder.mlp."):
        # PixArt timestep embedder: mlp.0/.2 are linear_1/linear_2 (SiLU between)
        idx = "linear_1" if k.startswith("t_embedder.mlp.0.") else "linear_2"
        leaf = k.rsplit(".", 1)[1]
        expanded[f"t_embedder.timestep_embedder.{idx}.{leaf}"] = v
    else:
        expanded[k] = v

missing, unexpected = model.load_state_dict(expanded, strict=False)
print("MISSING:", len(missing), missing[:6])
print("UNEXPECTED:", len(unexpected), unexpected[:6])
assert not missing, "reference load incomplete — comparison would be meaningless"

torch.manual_seed(0)
latent = torch.randn(1, 128, 3, 8, 8, dtype=torch.float32)

taps = {}
def tap(name):
    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        taps[name] = t.detach().float().contiguous()
    return hook

b0 = model.det_stages[0][0]
model.conv_in.register_forward_hook(tap("conv_in"))
b0.norm1.register_forward_hook(tap("b0_norm1"))
b0.attn.register_forward_hook(tap("b0_attn"))
b0.register_forward_hook(tap("b0_out"))
model.det_stages[0][1].register_forward_hook(tap("b1_out"))
model.upsamples[0].register_forward_hook(tap("up0_out"))

with torch.no_grad():
    ctx13 = model.forward_stages_1_to_3(latent, drop_leading_frame=True)
    ctx = model.forward_stage_4(ctx13, drop_leading_frame=True, pad_trailing=False)
    t = torch.tensor([1.0])
    x_t = torch.zeros(1, 3, ctx.shape[1], ctx.shape[2] * 4, ctx.shape[3] * 4)
    cx = model._context_and_x_for_diff_step(ctx, x_t)
    pred = model.forward_diff_step(cx, t)

print("REF ctx", tuple(ctx.shape), "mean|x|", ctx.abs().mean().item(), "std", ctx.std().item())
print("REF pred", tuple(pred.shape), "mean|x|", pred.abs().mean().item(), "std", pred.std().item())
dump = {"latent": latent, "context": ctx.contiguous(), "prediction": pred.contiguous()}
dump.update(taps)
for k, v in sorted(taps.items()):
    print(f"REF tap {k} {tuple(v.shape)} mean|x| {v.abs().mean().item():.5f}")
save_file(dump, OUT)
print("dumped", OUT)
