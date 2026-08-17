---
type: Decision
title: IC-LoRA upscaling keeps adapter AND reference through stage 2
description: Deliberate divergence from ic_lora.py, which refines on the bare transformer. Measured over seven runs across both generations — subject identity survives the sigma-0.909 renoise only when both are active; either alone, or neither, swaps the subject.
tags: [iclora, upscaler, two-stage, identity, ltx-2.5, ltx-2.3]
timestamp: 2026-08-13T00:00:00Z
---

`ic_lora.py` builds its stage 2 with `loras=()` and images-only conditioning:
bare transformer, no video reference. This port keeps the adapter fused and the
reference appended through stage 2 instead.

# The evidence (seven runs, both generations, same source/seed)

| Final stage runs with | Subject identity |
|---|---|
| adapter + reference (single-stage 2.5, single-stage 2.3, two-stage 2.3, two-stage 2.5) | **held, 4/4** |
| adapter only | lost |
| reference only | lost |
| neither | lost |

# Why

The inter-stage renoise (`lerp` to sigma 0.909) keeps ~9% of the upscaled
latent — enough for position, scale and composition, not for fine identity. The
refinement therefore needs an anchor, and the reference tokens only anchor when
the adapter trained to read them is active: the base model was never trained
with appended reference tokens and effectively ignores them. Both halves
together, or the subject is reinvented.

Whether upstream tolerates the drift or its real-world ComfyUI workflow supplies
an anchor invisible in `ltx-pipelines` is unknown; what is recorded here is
measured on this port.

# The measurement lesson

Centroid, subject scale and high-frequency energy were all blind to the failure:
every broken run scored well on them (position within 2%, texture near-native).
The discriminating check was categorical, not statistical — *is it the same car
between stage 1 and the final output* — and `upscale --stage-one PATH` exists so
that check takes one glance. Cost of learning this: the one configuration that
preserved identity was abandoned to fix an HF-energy deviation nobody had
complained about.
