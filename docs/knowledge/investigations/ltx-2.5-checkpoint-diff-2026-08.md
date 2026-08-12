---
type: Investigation
title: What LTX-2.5 actually changes (August 2026)
description: Tensor-level diff of the LTX-2.5 checkpoints against LTX-2.3, read from the safetensors headers. The DiT is unchanged apart from two flags; the video/audio VAEs and the latent upscaler are bit-for-bit the same architecture; the whole cost of the port sits in the Gemma 4 text encoder.
tags: [ltx-2.5, architecture, checkpoints, weight-loading, text-encoder]
timestamp: 2026-08-12T00:00:00Z
---

LTX-2.5 shipped on 11 August 2026. Before scoping the port, every component's
safetensors header (tensor names, shapes and embedded `__metadata__`) was read
by HTTP range request and diffed against the local
`ltx-2.3-22b-distilled.safetensors`. The conclusion is narrower than the
release notes suggest: **the diffusion transformer is the same model with two
flags flipped, and three of the five weight-carrying components are unchanged
architectures.** The port cost is concentrated in one place — the text encoder.

Method, reusable: the first 8 bytes of a safetensors file are the header
length, so `Range: bytes=0-7` then `Range: bytes=8-N` yields the complete
tensor manifest plus `__metadata__` for a few hundred KB, on a 42 GB file, in
one second. Lightricks stamps the full model config into `__metadata__["config"]`,
so the architecture is knowable without downloading a single weight.

# The transformer: two config keys, 97 tensors

`__metadata__["config"]["transformer"]` for
`ltx-2.5-22b-distilled-transformer-bf16.safetensors` is identical to LTX-2.3's
apart from two added keys:

| Key | 2.3 | 2.5 |
|---|---|---|
| `ff_bias` | absent (⇒ `true`) | `false` |
| `use_keyframes_abs_pos_embedding` | absent (⇒ `false`) | `true` |

Everything else matches exactly: 48 layers, 32 × 128 heads, `in/out_channels`
128, `cross_attention_dim` 4096, `caption_channels` 3840, RoPE θ = 10000 with
`max_pos [20, 2048, 2048]` and `float64` frequency precision, `rope_type`
`split`, gated attention on, `cross_attention_adaln` on,
`caption_proj_before_connector` on, 8-layer connector with 128 learnable
registers, `use_middle_indices_grid`, `timestep_scale_multiplier` 1000. The
**frame cap and resolution constraints are therefore unchanged**, including
[the 481-frame RoPE bound](/docs/knowledge/decisions/frame-cap-481-rope-range.md).

The tensor manifest agrees. Normalising block indices, the whole diff over the
`model.diffusion_model.*` namespace is:

* removed: `transformer_blocks.N.ff.net.0.proj.bias` and
  `transformer_blocks.N.ff.net.2.bias` — 96 tensors, all 48 blocks
* added: `keyframes_abs_pos_embedding`, shape `[1, 4096]`

No shape changed anywhere else. The connector blocks (`video_embeddings_connector`,
`audio_embeddings_connector`, 129 tensors each) **keep** their FFN biases — only
the DiT blocks lose them.

Consequence for this port: `LTXFeedForward` must build bias-free Linears for a
2.5 checkpoint. Building them with biases would leave 96 vectors at MLX's random
initialisation inside the forward pass — the loader reports unmatched keys, but
nothing forces the model to refuse them. Hence `LTXTransformerConfig.ffBias`,
threaded from `LTXTransformerConfig.ltx25` down to the blocks.

`keyframes_abs_pos_embedding` is a learned marker added **only** to *generated*
keyframe slots — the interior keyframes the DFR pipeline invents. The reference
implementation never marks ordinary image, first/last-frame or keyframe-index
conditioning (`ltx_core/conditioning/types/keyframe_slots.py`), so
[our append-based keyframe path](/docs/knowledge/pitfalls/keyframes-append-not-inject.md)
needs nothing for it.

# Unchanged components (verified tensor-for-tensor)

| Component | Result |
|---|---|
| Conv video VAE (`ltx-2.5-video-vae-conv-bf16`) | 170 tensors, **identical names and shapes** to the `vae.*` block of the 2.3 unified file |
| Audio VAE + vocoder (`ltx-2.5-audio-vae-bf16`) | 1329 tensors, **identical** to 2.3's `audio_vae.*` + `vocoder.*` |
| Latent spatial upscaler ×2 | 24 distinct tensor patterns, **identical shapes** to `ltx-2.3-spatial-upscaler-x2` |
| Sigma schedules | `DISTILLED_SIGMA_VALUES` and the stage-2 subset are byte-identical to ours |

So `VideoDecoder`, `AudioVAE`, `LTX2Vocoder` and `SpatialUpscaler` load 2.5
weights without a line of change. The new **latent temporal upscaler ×2** is
the same architecture at 512 channels instead of 1024.

# What is genuinely new

* **Text encoder — the whole cost.** 2.3 pairs with stock Gemma 3 12B, which is
  why a community 4-bit MLX build works. 2.5 ships `gemma4-12b-ltx-v1`: a
  `gemma4_unified` derivative (48 layers, hidden 3840, MLP 15360, `head_dim`
  256 sliding / `global_head_dim` 512 full, `attention_k_eq_v: true` so the 8
  full-attention layers carry no `v_proj`, per-layer `layer_scalar`,
  `partial_rotary_factor` 0.25 on full-attention RoPE). It exists **only**
  inside the 26 GB LTX file — no community quantisation to fall back on. The
  LTX-side projections are unchanged in shape
  (`video_aggregate_embed` `[4096, 188160]` = 49 hidden states × 3840, same as
  2.3), so the Feature-Extractor-V2 / per-token-RMS path carries over intact;
  only the LM underneath differs. The checkpoint declares its pairing in
  `__metadata__["gemma_source_checkpoint"]`, and upstream refuses a mismatched
  Gemma root outright — a wrong encoder is garbage, not degraded output.
* **Diffusion video decoder ("DiffVAE")** — a genuinely new decoder
  (`det_stages`, `diff_blocks`, `t_embedder`, `shared_adaln`, `type_emb`,
  396 tensors) replacing VAE reconstruction. The conv decoder ships alongside
  it and is the compatible path.
* **Duration head** — 15 tensors, 3.8 MB: attention pooler + per-modality
  projections predicting clip length from the caption.
* **Pixel-space spatial upscaler** — an IC-LoRA (rank 32, attn1/attn2 q/k/v/out
  + both ff layers, 48 blocks) in its own gated repo, with metadata
  `reference_downscale_factor: 2` **and a new `reference_spatial_scale_factor: 2`**
  that our IC-LoRA reference builder does not read yet.

# Distribution changes that break URLs

Checked live, August 2026: upstream renames files in place and withdraws old
revisions, so hard-coded filenames rot silently.

* `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` → **`…-IC-LoRA-DubIt`**, and
  `ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors` →
  `ltx-2.3-22b-ic-lora-dubit-0.9.safetensors`. The repo id 307-redirects; the
  old **filename** 404s. (Upstream also removed `LipDubPipeline` in favour of
  `DubItPipeline`.)
* `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` was withdrawn; only `-1.1`
  resolves. Both of our downloads had been broken for anyone without a warm
  cache.
* Every LTX-2.5 repo is **gated** (`gated: auto` — accept the licence, then
  send a token), where `Lightricks/LTX-2.3` is open. The IC-LoRA repos are
  gated in both generations.

Hence `ModelDownloader` now resolves names through `LTXAuxiliaryModel`, accepts
previously-shipped filenames when reading the cache, and maps 401/403/404 to
messages naming the repo and its licence page.

# Practical read

A 2.5 port is not a rewrite. In dependency order: Gemma 4 text encoder (the
real work — `gemma-4-swift-mlx` already implements `gemma4_unified` with
`global_head_dim`, K=V and `layer_scalar`), then split-checkpoint loading,
then `ffBias` (done), then the optional extras — duration head, temporal
upscaler, pixel upscaler IC-LoRA, DiffVAE. Upstream states that most 2.3
LoRAs and IC-LoRAs run unchanged on 2.5, which the tensor diff supports:
LoRAs patch weights, and no weight shape moved.
