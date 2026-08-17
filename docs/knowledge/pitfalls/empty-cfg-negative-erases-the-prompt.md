---
type: Pitfall
title: CFG against an empty negative erases the prompt on from-scratch dev runs
description: The dev paths inherited "" as the CFG negative from the community MLX port; upstream guides against DEFAULT_NEGATIVE_PROMPT. At cfg 3.0 from pure noise the empty negative wiped a 14-second choreography that the two-stage siblings kept — one A/B run apart.
tags: [cfg, negative-prompt, dev-checkpoint, single-stage, root-cause]
timestamp: 2026-08-16T00:00:00Z
---

`ti2vid_one_stage.py` (and every upstream CFG call site) guides against
`DEFAULT_NEGATIVE_PROMPT` — ~1.2k characters of trained artifact tags
("has_subtitles, has_blurbox, transition from black, blurry, camera shake,
…"). The community MLX port used `""`, and both of our dev paths inherited
that: the retake loop (where it survived validation, because retake starts
from an existing video whose structure anchors the trajectory) and
`generateVideoDev` (where it did not).

Measured on the 2CV bench, same seed/prompt/frames, single variable flipped
(commit `4c44f0ce`): with `""`, the 30-step single-stage run held a static
scene for 14 s — lift-off, hover, launch and fire trails all absent — while
A'8/B'/C' (two-stage, no CFG) rendered them. With the official negative, the
same run renders the full choreography.

The mechanism: CFG computes `cond + (scale−1)(cond − neg)`. Against the
trained artifact-tag negative, that direction means "away from artifacts";
against the empty-prompt encoding it means "away from the unconditional
mean", which at scale 3.0 over 30 steps drowns the prompt's own content.
Distilled two-stage paths never see this (cfg 1.0, no negative pass).

# The defense

- `LTXPipeline.defaultNegativePrompt` carries the upstream text verbatim;
  `generateVideoDev` encodes it for the CFG pass.
- The retake dev path still uses `""` (validated behaviour, weaker exposure);
  if retake quality on dev checkpoints ever disappoints, this is the first
  knob to try.
- General lesson: a guidance *direction* is part of the trained contract,
  like an activation variant. "Empty negative" is not a neutral default —
  it is a different vector.

# Citations

[1] A/B runs p5-d.mp4 (empty) vs p5-d2.mp4 (official), 2026-08-15/16,
    seed 42, 337 frames, 30 steps, CFG 3.0 + STG [28] + rescale 0.7.
[2] Lightricks/LTX-2 `packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py`.
