---
type: Pitfall
title: VAE decoder DepthToSpace blocks must use residual=false
description: residual=true in the decoder's D2S upsamplers was the root cause of the visible pixel grid on every generated frame.
tags: [vae, decoder, grid-artifact, root-cause]
timestamp: 2026-07-16T00:00:00Z
---

The LTX-2.3 VAE config names decoder upsampler modes `compress_all` /
`compress_space` / `compress_time` — **without** the `_res` suffix. Only the
encoder uses `_res` variants (residual=true). The Swift port initially applied
residual=true in the decoder's `VAEDepthToSpaceUpsample3d` blocks too, which
produced a visible grid pattern on every frame (grid metric @4px = 1.10;
after the fix = 0.98, smooth). Root-caused March 2026.

# The defense

- Decoder D2S: `residual: false`. Encoder `_res` blocks: `residual: true`.
  The config suffix is the contract — parse it, don't assume symmetry.
- Related decoder facts worth not re-deriving: 9 flat `up_blocks`
  (5 ResBlockGroup + 4 D2S), `conv_out` → 48 channels → unpatchify(4) → RGB,
  `output_frames = 8 * (latent_frames - 1) + 1`, **no timestep conditioning**
  (`timestep_conditioning: false` — no scale_shift_table or noise injection
  in the decoder). **Correction (2026-08-30):** both encoder and decoder pad
  with `.zeros` (`vae.spatial_padding_mode` is one flat top-level checkpoint
  key, shared by both) — "decoder with `.reflect`" here was the framework's
  fallback-when-missing default, misread as the checkpoint's actual value; see
  [conv-decoder-wrong-spatial-padding.md](conv-decoder-wrong-spatial-padding.md).

# Citations

[1] Grid-artifact investigation, March 2026 (grid metric before/after in the
PR that landed the fix).
