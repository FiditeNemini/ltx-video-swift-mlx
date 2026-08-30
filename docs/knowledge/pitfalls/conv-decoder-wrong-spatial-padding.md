---
type: Pitfall
title: The conv VAE decoder padded every conv with reflect instead of zeros
description: Every generated clip's default decode path used the wrong spatial padding mode — 17-27% relative error against the reference decoder, collapsing to ~1e-6 once fixed.
tags: [vae, decoder, padding, root-cause, parity-harness]
timestamp: 2026-08-30T00:00:00Z
---

`ConvVideoDecoder`'s five convolution sites (`conv_in`, `conv_out`, the two
`ResnetBlock3D` convs, and the `DepthToSpaceUpsample` conv) all defaulted to
`spatialPaddingMode: .reflect` in the Swift port
(`Sources/LTXVideo/Models/VAE/VideoDecoder.swift` /
`VideoConvolution.swift`). None of the five call sites overrode it, so every
conv in the decoder padded with `.reflect` — and this is the *default* decode
path: the one every clip this repo has ever produced went through (the
diffusion decoder is opt-in via `--diffvae`).

The real checkpoint (`ltx-2.5-video-vae-conv-bf16.safetensors`, and by
extension LTX-2.3's — [investigated as unchanged between the two
versions](/docs/knowledge/investigations/ltx-2.5-checkpoint-diff-2026-08.md))
carries `vae.spatial_padding_mode: "zeros"` in its metadata. `ltx_core`'s
`_build_conv_video_decoder` only falls back to `"reflect"` when the config key
is *absent* — it is present here, and set to `"zeros"`. [The D2S-residual
pitfall](decoder-d2s-residual-false.md) had documented "decoder pads with
`.reflect`" as an established fact; that was the code's fallback default
misread as the checkpoint's actual value, never checked against the real
config. That line is now corrected.

Found by [`ConvVAEDecoderParityTests`](../../../Tests/LTXVideoTests/ConvVAEDecoderParityTests.swift)
(issue #57), which runs the port against Lightricks' own `ConvVideoDecoder` on
a fixed small latent (`scripts/conv_video_decoder_reference.py`). Relative
error against the reference was 17% at `conv_in` alone and grew to 27% by
`up_blocks_2`, settling around 8-10% at the final output — small enough on a
typical production-size latent (interior pixels, unaffected by border
padding, dominate the average) to read as "looks right" in every generation
this repo has shipped, and exactly the class of bug the harness exists to
catch. After passing `spatialPaddingMode: .zeros` at all five call sites,
every tap point (including the final output) matches the reference to
2-8e-6 — bf16-checkpoint-cast-to-float32 noise, not a further defect.

# The defense

- A framework default is not the checkpoint's value. `ltx_core`'s
  `config.get("spatial_padding_mode", "reflect")` is a fallback for a
  *missing* key — always read the real checkpoint's metadata before porting a
  default, the way `scripts/*_reference.py` do (`json.loads(header["__metadata__"]["config"])`).
  This file's own decoder blocks have no `_class_name` variance to hide
  behind: `vae.spatial_padding_mode` is one flat top-level key, shared by
  encoder and decoder.
- A relative-error metric on a small test latent is far more sensitive to
  border-padding bugs than production output at typical resolutions — that is
  a feature of the harness, not a sign the bug does not matter. A padding
  mismatch degrades every border/edge region of every frame, which visual
  spot-checks reliably miss but an element-wise reference does not.
- [`docs/knowledge/pitfalls/decoder-d2s-residual-false.md`](decoder-d2s-residual-false.md)
  previously stated "decoder with `.reflect`" — corrected there; if you find
  another doc repeating it, fix it too.

# Citations

[1] `ConvVAEDecoderParityTests.outputMatchesReference` / `bisectFirstDivergence`,
measured relative error 0.098 (output) / 0.17-0.27 (per-layer) before the fix,
2.3e-6-7.7e-6 after, against `scripts/conv_video_decoder_reference.py` run on
CPU float32 via Lightricks' own `ConvVideoDecoder` (2026-08-30).
