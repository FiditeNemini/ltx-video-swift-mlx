---
type: Pitfall
title: LipDub segments cap at ~233 frames, not 481 — the audio reference doubles the span
description: The audio reference occupies NEGATIVE RoPE positions, so the audio stream spans twice the segment duration against the same 20 s window. Past ~9.9 s the lip-sync drifts by a constant lag.
tags: [lipdub, rope, frames, audio-reference, segmentation]
timestamp: 2026-07-26T00:00:00Z
---

[The 481-frame cap](/docs/knowledge/decisions/frame-cap-481-rope-range.md) is
correct for `generateVideo` / `generateRetake`: video RoPE coordinates run
`0 → numFrames/24` seconds against `maxPos[0] = 20`.

**LipDub does not get that budget.** `buildAudioReference` shifts the whole
reference into negative time so it sits "before" the target
(`AudioReference.swift`): the reference ends at `-0.04 s` and starts at
`-(refDuration + 0.04)`. The audio stream therefore spans

```
[-(refDur + 0.04), targetDur]   ≈ 2 × segment duration
```

against the same `audioMaxPos = 20 s` normalization. A 15.7 s segment asks
the audio RoPE to cover ~31 s.

# Measured

Same source image, same voice, same seed, same LoRA — only the segment length
changed (image mode, French TTS at 24 fps):

| Segment | Audio span | Lip-sync |
|---|---|---|
| 377 frames (15.7 s) | ~31.4 s | constant **~0.75 s lag**; the audio pause at 2.67 s appears at 3.4 s |
| 233 frames (9.7 s) | ~19.5 s | in sync (mouth opens at 0.21 s for a 0 s onset) |

The bound: `2 × duration + 0.04 ≤ 20 s` → **9.98 s → 233 frames** in `8n+1`.

# The defense

- Split dialogue at **233 frames** per LipDub segment (the app-side
  segmenter, not the framework, owns the split), and chain segments with
  `continuationTailPath` — see
  [the continuation decision](/docs/knowledge/decisions/lipdub-continuation-anchor.md).
- `generateLipDub` prints a WARNING when the computed span exceeds
  `audioMaxPos`, naming the largest safe frame count.
- Cut segments **inside speech pauses**: both sides of the seam then have a
  closed mouth, which hides any residual discontinuity. Verified on a
  3-segment / 19 s chain — seams invisible.
- The old 257-frame cap (~10.7 s), removed in PR #36 as invented, happened to
  sit near the correct LipDub bound. Raising it to 481 is right for the other
  pipelines but exposed this one.
