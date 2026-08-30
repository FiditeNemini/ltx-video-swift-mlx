"""Reference intermediates for the LTX-2.5 text connector.

Runs Lightricks' own FeatureExtractorV2 and Embeddings1DConnector on
synthetic Gemma hidden states (CPU float32) and dumps the tensors our Swift
port must reproduce. Every prompt goes through this pipeline: 49 Gemma
hidden-state layers -> per-token RMS norm + concat -> video_aggregate_embed
(188160 -> 4096) -> 8-layer 1D transformer with SPLIT RoPE, learnable
registers replacing padded tokens, and gated attention -> final RMS norm.

Bypasses the real ~24 GB Gemma 4 model on purpose — this tests the connector
math (already validated in isolation against the reference hidden states in
Gemma4TextEncoderE2ETests), not Gemma itself. Deterministic given fixed
hidden states and a fixed seed.

    PYTHONPATH=<ltx-core>/src python3 scripts/connector_reference.py \
      <gemma4-with-proj-bf16.safetensors> <unified-transformer-bf16.safetensors> out.safetensors
"""
import sys

import torch
from safetensors.torch import load_file, save_file

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2

GEMMA_PROJ_CKPT = sys.argv[1]
UNIFIED_CKPT = sys.argv[2]
OUT = sys.argv[3]

# Real checkpoint values (verified against ltx-2.5-22b-dev-transformer-bf16.safetensors'
# metadata.config.transformer): connector_num_attention_heads=32,
# connector_attention_head_dim=128, connector_num_layers=8,
# connector_num_learnable_registers=128, connector_apply_gated_attention=true,
# connector_positional_embedding_max_pos=[4096], frequencies_precision=float64,
# rope_type=split. video_aggregate_embed output dim (num_attention_heads *
# attention_head_dim) = 4096.
HIDDEN_DIM = 3840
NUM_GEMMA_LAYERS = 49
VIDEO_INNER_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = 128
CONNECTOR_LAYERS = 8
NUM_REGISTERS = 128

# --- Feature extractor ---
video_lin = torch.nn.Linear(HIDDEN_DIM * NUM_GEMMA_LAYERS, VIDEO_INNER_DIM, bias=True)
# embedding_dim is the *per-layer* Gemma hidden size (3840), not the flattened
# D*L (188160) — ltx_core's encoder_configurator.py sets it from
# gemma_text_config.hidden_size, used only for the rescale factor
# sqrt(target_dim / embedding_dim). Confirmed against that file directly;
# getting this wrong scales the whole feature-extractor output by sqrt(49)=7x.
fe = FeatureExtractorV2(video_aggregate_embed=video_lin, embedding_dim=HIDDEN_DIM)
fe.eval().to(torch.float32)

proj_raw = load_file(GEMMA_PROJ_CKPT)
fe_missing, fe_unexpected = fe.load_state_dict(
    {
        "video_aggregate_embed.weight": proj_raw["text_embedding_projection.video_aggregate_embed.weight"].to(torch.float32),
        "video_aggregate_embed.bias": proj_raw["text_embedding_projection.video_aggregate_embed.bias"].to(torch.float32),
    },
    strict=False,
)
print("FE MISSING:", fe_missing)
assert not fe_missing, "feature extractor load incomplete"

# --- Connector ---
connector = Embeddings1DConnector(
    attention_head_dim=HEAD_DIM,
    num_attention_heads=NUM_HEADS,
    num_layers=CONNECTOR_LAYERS,
    positional_embedding_max_pos=[4096],
    num_learnable_registers=NUM_REGISTERS,
    rope_type=LTXRopeType.SPLIT,
    double_precision_rope=True,
    apply_gated_attention=True,
)
connector.eval().to(torch.float32)

unified_raw = load_file(UNIFIED_CKPT)
prefix = "model.diffusion_model.video_embeddings_connector."
conn_sd = {k[len(prefix):]: v.to(torch.float32) for k, v in unified_raw.items() if k.startswith(prefix)}
conn_missing, conn_unexpected = connector.load_state_dict(conn_sd, strict=False)
print("CONNECTOR MISSING:", conn_missing)
print("CONNECTOR UNEXPECTED:", conn_unexpected)
assert not conn_missing, "connector load incomplete"

torch.manual_seed(0)
B, T = 1, 1024
hidden_states = [torch.randn(B, T, HIDDEN_DIM, dtype=torch.float32) for _ in range(NUM_GEMMA_LAYERS)]
# Left-padding: this repo's actual convention (Gemma4TextEncoder.encode:
# `mask = [0]*padding + [1]*ids.count`) — a short prompt leaves most of the
# 1024-token window padded at the *front*, valid tokens at the tail. Not an
# arbitrary choice: the register-replacement bug behaves differently by
# padding side (see docs/knowledge/pitfalls/connector-register-replacement-reorders-tokens.md).
attention_mask = torch.ones(B, T, dtype=torch.float32)
attention_mask[:, :200] = 0.0

taps = {}
def tap(name):
    def hook(_m, _inp, out):
        taps[name] = out.detach().float().clone().contiguous()
    return hook
for i, block in enumerate(connector.transformer_1d_blocks):
    block.register_forward_hook(tap(f"block_{i}"))

with torch.no_grad():
    video_feat, _ = fe(hidden_states, attention_mask)
    additive_mask = (1.0 - attention_mask)[:, None, None, :] * torch.finfo(torch.float32).min
    # Also capture the connector's post-register-replacement hidden state and
    # mask, so the Swift side can feed the transformer blocks the exact same
    # input and isolate register-replacement from the block math.
    post_register_hidden, post_register_mask = connector._replace_padded_with_learnable_registers(
        video_feat, additive_mask
    )
    connector_out, out_mask = connector(video_feat, additive_attention_mask=additive_mask)

print("REF feature_extractor output", tuple(video_feat.shape), "mean|x|", video_feat.abs().mean().item())
print("REF connector output", tuple(connector_out.shape), "mean|x|", connector_out.abs().mean().item(),
      "std", connector_out.std().item())
for k, v in sorted(taps.items()):
    print(f"REF tap {k} {tuple(v.shape)} mean|x| {v.abs().mean().item():.5f}")

dump = {
    "hidden_states": torch.stack(hidden_states, dim=-1).clone(),  # [B, T, D, L]
    "attention_mask": attention_mask.clone(),
    "feature_extractor_output": video_feat.detach().clone().contiguous(),
    "connector_output": connector_out.detach().clone().contiguous(),
    "post_register_hidden": post_register_hidden.detach().clone().contiguous(),
    "post_register_mask": post_register_mask.detach().clone().contiguous(),
}
dump.update(taps)
save_file(dump, OUT)
print("dumped", OUT)
