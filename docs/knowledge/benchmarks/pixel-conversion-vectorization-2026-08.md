---
type: Benchmark
title: Vectorized RGBA pixel conversion vs. the scalar loop it replaced
description: loadVideo on a 121-frame 768x512 clip went from ~8.15s to ~1.0s (~8x) once the RGBA-to-float conversion moved from a scalar Swift loop to a vectorized MLX op.
tags: [benchmarks, performance, m3-max, pixel-conversion, vae]
timestamp: 2026-08-30T00:00:00Z
---

`loadImage`/`loadVideo` (`Sources/LTXVideo/Pipeline/LatentUtils.swift`)
converted RGBA pixel buffers to normalized `[-1,1]` floats with a scalar
Swift loop, three writes per pixel — `loadVideo` ran it per frame with three
individual `[Float].append` calls per pixel. `loadVideo` sits on the
critical path of every retake, every LipDub video reference, and every i2v
run.

# Measurement

Timed `loadVideo` on the repo's example clip
(`docs/examples/lipdub/lipdub-teaser-french-ours-768x512-121f.mp4`, 121
frames @ 768x512) with an uncommitted harness, 3 runs after a warm-up, on an
M3 Max:

| | mean time |
|---|---|
| Before (scalar loop) | ~8.15 s |
| After (vectorized) | ~1.0 s |

**~8x faster.** The remaining ~1 s is AVFoundation frame-decode/resize
overhead (`AVAssetImageGenerator`), not pixel conversion — the scalar loop
was genuinely the dominant cost before this change, and after it the floor
is decode, not conversion. A further win there would need batched decode
(`AVAssetImageGenerator.images(for:)`, available since macOS 13) or a
reused `CGContext` instead of one alloc/free per frame — not attempted here.

# The fix

Both functions now build one `MLXArray` directly from the raw RGBA bytes and
do the byte→float normalization (`/127.5 - 1.0`) as a single vectorized MLX
op, slicing off the alpha channel, instead of a per-pixel Swift loop. Output
shapes, value range, and channel order are unchanged and locked by an
exact-value test (`Tests/LTXVideoTests/LatentUtilsTests.swift`).

# Citations

[1] Ad-hoc timing harness (uncommitted), `loadVideo(from:width:height:numFrames:)`
on the repo's example clip, M3 Max, 3 runs after warm-up (2026-08-30).
