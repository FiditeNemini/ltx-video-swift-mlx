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

# The defense

- `interpolateTemporally` starts at 0.725 by default and exposes
  `renoiseFrom`, with the trade-off stated in the help text: lower keeps the
  subject, higher invents more motion and can redraw it.
- The real fix is keyframe anchoring — the guide-token machinery this package
  already has for keyframes. That is the missing half of DFR.
- General lesson, and this is the second instance in one session (the other
  being upstream's diffusion decoder as a default): **a parameter borrowed
  from a reference implementation is only valid together with its
  prerequisites.** Copy the number, copy what makes it work.

# Citations

[1] Bench runs 2026-08-20, `interpolate` on the 121-frame 2CV clip, seed 42.
[2] Prior instance: [[iclora-stage2-keeps-adapter-and-reference]].
