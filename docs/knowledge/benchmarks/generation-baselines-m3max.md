---
type: Benchmark
title: Generation baselines on M3 Max 96 GB (Release, warm caches)
description: Healthy wall-clock numbers measured July 2026 — anything far above them is contention/thermal/misconfiguration, not the engine.
tags: [benchmarks, performance, m3-max, distilled]
timestamp: 2026-07-16T00:00:00Z
---

All numbers: `distilled` model, two-stage pipeline, Release binaries from
xcodebuild, model weights already on disk, M3 Max 96 GB, July 2026. These are
single-run wall-clock observations, not averaged benchmarks — treat as sanity
anchors with ±30% noise (thermal state matters, see below).

# Wall-clock anchors

| Operation | Measured |
|---|---|
| Model load, warm page cache | ~25 s – 2 min |
| Model load, first ever (download/convert) | ~8 min |
| `generate` 512×512×9 (beacon demo) | ~159 s generation |
| `generate` 512×512×25 | ~4 min total |
| `generate` 512×512×481 (20 s video, the cap) | ~1507 s (~25 min) |
| `lipdub` 384×256×121 (video ref + target audio) | 335 s and 829 s observed |
| `lipdub` 384×256×33 (E2E, back-to-back pair) | 67 s → 131 s and 234 s → 522 s |

# LTX-2.5 anchors (August 2026)

Same machine, `2.5-distilled`, i2v, two-stage, weights on an external SSD.

| Operation | Measured |
|---|---|
| Model load (transformer + Gemma 4 encoder), qint8 | 69 s |
| `generate` 768×512×121, qint8 | 330 s |
| `generate` 1280×832×121, bf16 | 1332 s |

Two things to expect when reading these against 2.3. The load number covers a
**24 GB bf16 text encoder** that 2.3 does not pay for — its Gemma 3 arrives
pre-quantized from `mlx-community`, while 2.5's derivative exists in bf16 only,
so `--transformer-quant` has to carry the encoder too. And the 1280×832 stage 2
runs ~42 000 latent tokens, roughly 4× the 768×512 stage 2, which is most of the
gap between those two rows — quantization accounts for far less of it.

# Known variance sources

- **Back-to-back runs**: the second of two consecutive LipDub runs measured
  ~2× the first in two independent sessions, with identical work and after
  the cache fix — thermal throttling is the working hypothesis. Don't read a
  slow second segment as a regression without a cooldown control.
- **Debug builds**: MLX in Debug is drastically slower; never benchmark them
  (and SPM binaries don't run at all — see
  [the metallib pitfall](/docs/knowledge/pitfalls/spm-binary-no-metallib.md)).
- The `--beacon` flag makes runs observable in SiliconScope (phase + step
  progress) at negligible cost — useful when attributing slowness.
