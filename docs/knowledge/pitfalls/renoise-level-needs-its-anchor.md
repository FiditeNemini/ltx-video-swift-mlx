---
type: Pitfall
title: A renoise level copied from upstream needs upstream's anchoring too
description: The temporal round renoises to sigma 0.975, which keeps ~2% of the source — viable there only because every tile is anchored on keyframe seams. Ported without the anchors, the same number redraws the subject. Measured 14.4 dB vs 20.1 dB identity at 0.725.
tags: [renoise, identity, temporal, dfr, parameters, root-cause]
timestamp: 2026-08-20T00:00:00Z
---

Upstream's DFR temporal round starts its refinement at sigma 0.975. Taking
that number into a single-window port produced smooth motion and **a
different car** — the identity-loss mode already recorded for the IC-LoRA
upscale chain, where a sigma-0.909 renoise only preserved the subject when
adapter *and* reference stayed active.

At sigma 0.975 the renoised latent is `0.975 * noise + 0.025 * source`. The
subject survives that only if something else pins it: in DFR, every tile
carries hard keyframe conditionings on its seams. The port had none, so the
model redrew from the prompt.

Measured on the bench clip, frame 0 against the source:

| refinement starts at | identity | wall time |
|---|---|---|
| 0.975 (upstream's) | 14.4 dB | 458 s |
| 0.725 | **20.1 dB** | 239 s |

# Resolved: anchoring beats the sigma trade-off

Anchoring the refinement on the source's own frames — one latent frame in
four, appended as guide tokens — restores upstream's level *with* its
prerequisite. Measured on the same clip, frame 0 against the source:

| variant | identity | motion/frame | sharpness |
|---|---|---|---|
| sigma 0.975, no anchors | 14.4 dB | 7.51 | 6.17 |
| sigma 0.725, no anchors | 20.1 dB | 7.91 | 6.61 |
| **sigma 0.975 + anchors** | **26.7 dB** | **7.98** | **7.57** |

It wins on all three at once, which is the point: the anchors hold the
subject, which frees the high noise level to invent motion instead of
spending itself on not drifting. It is no longer a trade-off.

One detail that had to be exact: an anchor built from a frame *of the
sequence being denoised* must land on that frame's own grid coordinate,
`(8i - 3) / fps`, which is not expressible as `(pixel + 0.5) / fps`.
Rounding to the nearest pixel would offset every anchor by half a frame, so
`buildKeyframeGuideToken` gained a variant taking the coordinate directly.

# The defense

- `interpolateTemporally` anchors every 4th source frame by default and runs
  at 0.975; `--anchor-every 0` disables it, `--renoise-from` still exposes the
  level.
- General lesson, and this is the second instance in one session (the other
  being upstream's diffusion decoder as a default): **a parameter borrowed
  from a reference implementation is only valid together with its
  prerequisites.** Copy the number, copy what makes it work.

# Citations

[1] Bench runs 2026-08-20, `interpolate` on the 121-frame 2CV clip, seed 42.
[2] Prior instance: [[iclora-stage2-keeps-adapter-and-reference]].
