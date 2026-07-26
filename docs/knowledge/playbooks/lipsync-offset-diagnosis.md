---
type: Playbook
title: Diagnosing a lip-sync offset or wrong mouth shapes
description: The ordered checklist that attributes a bad LipDub to its real cause — prompt, audio channels, speech windows, or alignment — before suspecting the model.
tags: [lipdub, diagnosis, lip-sync, playbook]
timestamp: 2026-07-16T00:00:00Z
---

Bad LipDub output has six known causes with distinct signatures. Check in
this order — each step is cheaper than the next.

# 0. Prompt text vs spoken audio (sync measurements are MEANINGLESS)

Before measuring anything: transcribe the reference audio (any STT) and
compare it with the prompt's dialogue. The generated speech follows the
**prompt**, so a mismatch guarantees desync and voids every timing number.
This wasted a day in July 2026. See
[the prompt pitfall](/docs/knowledge/pitfalls/lipdub-prompt-needs-dialogue.md).

# 0b. Segment length (CONSTANT lag, grows with duration)

Over ~233 frames (9.9 s) the audio reference's negative RoPE positions push
the span past the 20 s window: measured constant ~0.75 s lag at 377 frames,
in sync at 233. `generateLipDub` warns. See
[the segment-bound pitfall](/docs/knowledge/pitfalls/lipdub-segment-bound-233.md).

# 1. Prompt format (wrong mouth SHAPES, sync irrelevant)

The prompt must contain `speaking in <LANG> saying: "<literal dialogue>"`.
Scene-only prompts produce structurally wrong poses (wide smile on neutral
speech). See [the prompt pitfall](/docs/knowledge/pitfalls/lipdub-prompt-needs-dialogue.md).
Also check language: a prompt language mismatching the audio degrades sync
(user-verified).

# 2. Audio channel handling (mouth moves in WRONG DIRECTIONS)

If the reference audio went through any mono downmix (`ffmpeg -ac 1`, forced
`AVNumberOfChannelsKey: 1`), the AudioVAE features are garbage. See
[the stereo pitfall](/docs/knowledge/pitfalls/audio-must-stay-stereo.md).

# 3. Speech-window detection (constant OFFSET / late attack)

Run with `--debug` and read the alignment log:

```
[lipdub] source speech window: 0.200s..4.580s (4.380s)
[lipdub] target speech window: 0.290s..4.050s (3.760s)
[lipdub] time-stretch rate=0.858 (pitch preserved)
```

Red flags: a window spanning the whole clip (detection found no boundaries),
or a `rate` far from `target speech / source speech`. Cross-check the audio
with ffmpeg:

```bash
ffmpeg -i target.wav -af silencedetect=n=-35dB:d=0.2 -f null -   # boundaries?
ffmpeg -i target.wav -af "atrim=0:0.3,astats=metadata=1" -f null - 2>&1 | grep "RMS level"  # noise floor
```

A noise floor above -35 dB (enrolled voices) is handled since PR #36 by the
floor-relative threshold — see
[the thresholds decision](/docs/knowledge/decisions/speech-window-noise-floor.md).
If the windows are wrong anyway, tune `thresholdDB`/`noiseFloorMarginDB` on
`alignTargetToSource` and file what you learned here.

# 4. Fusion state (everything looks right but output is degraded)

On consecutive runs in one process, verify the fusion log says either
`LoRA fused: 1344 / 1344 layer-pairs (100.0%)` or
`LoRA already fused (same file) — reusing fused transformer`. A doubled
delta or a stale file should be impossible since PR #36 (guards throw), but
if you see burned/saturated output, check
[the double-delta pitfall](/docs/knowledge/pitfalls/lora-refusion-double-delta.md).

# 5. The voice itself sounds wrong (timbre, not timing)

If sync is fine but the voice is unrecognisable, the loss is almost certainly
upstream of LTX. Measure **F0 vs H2** (energy at the fundamental against the
second harmonic, median over voiced frames) on the reference *and* on the
generated audio: a healthy voice sits near 0 dB, and the decoder transmits
that ratio nearly unchanged. A thin reference in, a thin voice out. Do not
trust a plain F0 tracker here — with a depleted fundamental it locks onto
2×F0 and reports a plausible wrong number. Full attribution chain in
[the custom-voice investigation](/docs/knowledge/investigations/custom-voice-timbre-chain-2026-07.md).

# When all of them pass

The residual is the known audio-anchored vs pose-anchored trade-off — see
[the AdaLN investigation](/docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md)
for the quantitative comparison method (Pearson audio-envelope vs
mouth-openness) before concluding anything.
