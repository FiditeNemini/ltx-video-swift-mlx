---
type: Pitfall
title: LipDub overwrites anything crossing the mouth
description: The IC-LoRA regenerates the mouth region without modelling occlusion, so a hand, a microphone or a headset band passing in front of the face gets painted over by the new lips. Visible in the teaser shipped with this repo.
tags: [lipdub, artifacts, reference-video, integration]
timestamp: 2026-08-12T00:00:00Z
---

LipDub redraws the mouth to match the target audio. It does not reason about
depth: whatever sits **in front of** the mouth in the reference frame is
regenerated away, because the model treats that image region as mouth to be
repainted rather than as an object occluding it.

Reproducible in the example shipped with this repo —
`docs/examples/lipdub/lipdub-teaser-french-ours-768x512-121f.mp4`, frames 52–64.
The subject lifts a headset in front of his face; the band crosses the lower
face and the regenerated lips are composited over it, so the arch appears to
pass *behind* the mouth. That clip predates the August 2026 vocoder work
entirely, so this is a property of the method, not a regression.

# What it means when choosing reference footage

* Prefer shots where **nothing crosses the lower face** for the segment's
  duration: no hand gestures near the chin, no held product, no microphone in
  frame at mouth height, no headset being put on.
* The artifact is *local*: the rest of the frame is untouched, so a clip that
  only briefly violates this is usually salvageable by cutting the segment
  boundary around the crossing rather than discarding the take.
* It scales with how much of the mouth the object covers. A thin mic stand edge
  is often invisible; a headset band spanning the lips is not.

# Diagnosing it

If a LipDub output shows an object "melting" or passing behind the face, check
the *reference* frames at that timestamp before suspecting the pipeline. The
question to ask is whether something crosses the mouth there — not whether the
LoRA fused, the vocoder changed, or the audio drifted.

That distinction cost a round trip once: a warmer colour grade and this artifact
were both attributed to a vocoder change that, by construction, only runs after
video decoding. Colour differences between two LipDub outputs are generation
variance (different seed, different prompt, and a reference that is itself a
generation); the occlusion artifact is this pitfall.
