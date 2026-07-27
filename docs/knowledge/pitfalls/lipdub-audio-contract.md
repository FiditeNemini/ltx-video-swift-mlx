---
type: Pitfall
title: The LipDub audio contract — what to feed it, and which track to ship
description: LipDub denoises video and audio jointly, so the lips follow the audio the model produced, not the file you passed in. Six rules, each with the measurement that established it.
tags: [lipdub, audio, integration, contract]
timestamp: 2026-07-26T00:00:00Z
---

LipDub is not "video generation with an audio input". `LTX2Transformer`
denoises a **video stream and an audio stream jointly**, with cross-attention
at every step: the reference audio conditions the run, and the model produces
its own audio, which the lips are locked to by construction. Every rule below
follows from that one fact, and each was established by measurement during
the July 2026 custom-voice campaign
([investigation](/docs/knowledge/investigations/custom-voice-timbre-chain-2026-07.md)).

# 1. Ship the generated audio, never substitute the input

`VideoGenerationResult.audioWaveform` (+ `audioSampleRate`) is the track the
mouth is synchronised with, and `VideoExporter` already muxes it — natively,
via `AVAssetWriter` + `AVMutableComposition`, no external tool. Replacing it
with the original TTS file reintroduces exactly the offset the pipeline
avoided: the model *re-times* its copy (measured wander ±0.2–0.9 s depending
on how fluent the reference is), so the substituted track drifts against the
lips even though the generation itself was in sync.

The temptation is real — the generated voice is a resynthesis and never a
sample-exact copy — but the fix is upstream (rules 4–5), not a swap.

# 2. The prompt must carry the verbatim transcript

The generated speech follows the prompt's dialogue. A prompt that disagrees
with the reference audio makes the model speak the *prompt*, and no sync is
possible. See
[the prompt pitfall](/docs/knowledge/pitfalls/lipdub-prompt-needs-dialogue.md).
Callers usually have the exact text already (it drove the TTS); when they
don't, transcribe before generating.

# 3. Segments cap at ~233 frames

The audio reference sits at negative RoPE positions, doubling the span. See
[the segment-bound pitfall](/docs/knowledge/pitfalls/lipdub-segment-bound-233.md).
Cut segments **inside speech pauses**: both sides of the seam then have a
closed mouth, which hides any residual discontinuity.

# 4. The reference's own defects are reproduced

Whatever the reference sounds like is what the model copies. Measured on the
same pipeline, same seed:

| Reference | Timing fidelity of the copy |
|---|---|
| Preset TTS, fluent, ~−24 dB RMS, natural noise floor | pause reproduced within **5 ms** |
| Enrolled voice, choppy, −38 dB RMS, digital-zero silences | rhythm reinvented, words altered |

Target: level comparable to the shipped presets (~−24 dB RMS), a **natural
noise floor** (not digital zeros), and fluent delivery. Note the direction of
that last one — a hard-gated reference is *worse* than a slightly noisy one,
because exact zeros are out of distribution.

Caveat on that last point, from the Voxtral side (July 2026): a TTS reference
may carry exact zeros that came from **the codec, not from a gate**, and their
proportion varies **3.4 %–10.5 % between generations of the same voice**. So
"the reference has digital zeros" is not by itself evidence of a gating bug
upstream, and nothing downstream may assume a natural floor is present. Our
own silence detection already survives this — see rule 7.

# 5. Normalising the level afterwards does not fix a bad reference

Tempting shortcut, measured and rejected: taking a −38 dB reference and
raising it +13 dB before the run left the re-timing unchanged (start
+0.95 s — the worst measured). Gain multiplies the zeros and the choppiness
along with the speech. The repair belongs where the audio is produced.

# 6. Judge timbre with F0 vs H2, not a pitch tracker

Energy at the fundamental against the second harmonic, median over voiced
frames: a healthy voice sits near 0 dB, and the decoder transmits that ratio
nearly unchanged (measured −15.1 → −14.5 dB on a thin input; −7.3 → −5.8 dB
on a healthy one, i.e. slightly *improved*). A plain F0 tracker is
misleading here — with a depleted fundamental both autocorrelation and HPS
lock onto 2×F0 and report a plausible wrong number (199 Hz for an 88 Hz
voice, in the measured case).

Do not chase the last few dB. Between a clean mic recording and its cloned
synthesis, ~5 dB of fundamental is lost **inherently**: the enrolled codes are
faithful, the loss happens in generation (established by the Voxtral team,
July 2026). A synthesis landing a few dB under its own reference is normal and
not a defect to escalate.

# 7. Do not assume a natural noise floor is present

Related to rule 4, but it constrains *us*, not the caller: a TTS reference can
contain exact digital zeros produced by the codec, in a proportion that varies
**3.4 %–10.5 % between generations**. Any analysis that estimates a floor must
therefore treat "no floor at all" as a normal input.

`AudioPreprocessor.detectSpeechWindow` already does: the 10th-percentile frame
RMS is only trusted when it is non-zero **and** at least 15 dB below the
loudest frame; otherwise the absolute −35 dBFS threshold takes over. A
generation full of zeros degrades to that fallback instead of producing a
nonsense window. Keep that guard if the detector is ever rewritten.

# 8. Quantization of the upstream TTS is not a LipDub concern

For the record, because the opposite was briefly (and wrongly) recommended
from a single observation: on cloned voices Voxtral **q6 beats bf16** —
99.4 % vs 96.5 % coverage, RTF 1.47 vs 3.44, 3.5 GB vs 8 GB. A dropped word in
one q6 synthesis is generation variance, not a quantization effect. Nothing in
the LipDub path cares which variant produced the reference; judge a reference
by rules 4 and 6, never by its provenance.

# What the framework does NOT do

No external binary is invoked anywhere in generation or export: frames go out
through `AVAssetWriter`, audio through a hand-built WAV writer and
`AVAssetWriter`, and the two are muxed by `AVMutableComposition`. Since the
continuation anchor reads its tail natively, a LipDub workflow needs no
external tool at all.
