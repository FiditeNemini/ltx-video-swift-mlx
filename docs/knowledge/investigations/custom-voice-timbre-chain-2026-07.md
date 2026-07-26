---
type: Investigation
title: Custom-voice LipDub — where the timbre and the sync are actually lost (July 2026)
description: A four-day attribution campaign across Voxtral enrollment and LTX LipDub. The decoder was innocent; the losses were an over-long segment, a prompt/audio mismatch, and an upstream high-pass that destroyed the fundamental.
tags: [lipdub, voice-clone, audio, attribution, voxtral]
timestamp: 2026-07-26T00:00:00Z
---

Reported symptom: LipDub runs driven by an enrolled ("custom") voice came out
with lips lagging the voice, and the generated voice was unrecognisable —
while the same pipeline with a preset TTS voice had been fine.

Four causes, in the order they were isolated. Only the first two are
framework-side.

# 1. Segment length (constant lag) — framework

The failing runs were single 377-frame segments (15.7 s), newly possible
after the 257 → 481 cap change. The audio reference sits at negative RoPE
positions, so the span is doubled: ~31 s against a 20 s window. Constant
0.75 s lag; the same source at 233 frames is in sync. See
[the segment-bound pitfall](/docs/knowledge/pitfalls/lipdub-segment-bound-233.md).

# 2. Prompt text ≠ spoken audio (nonsense sync) — integration

Several diagnostic runs used a prompt reconstructed by ear instead of the
audio's real transcript. The generated audio follows the **prompt**, so those
runs compared two different sentences and every timing number was void. See
[the prompt pitfall](/docs/knowledge/pitfalls/lipdub-prompt-needs-dialogue.md).
This invalidated a batch of intermediate measurements — the ones surviving
are those where prompt and audio provably agree.

# 3. The LTX audio decoder — innocent (measured)

Three independent measurements clear the AudioVAE + vocoder chain:

| Probe | Input → generated |
|---|---|
| Energy above 8 kHz | −21.3 → −21.8 dB (the top of the spectrum is not cut) |
| Fundamental strength (F0 vs H2) | −15.1 → −14.5 dB (a thin input stays thin — faithfully) |
| Per-run spectral delta | varies run to run → no fixed decoder colouration |

On the final, healthy input the decoder even *improved* the fundamental
(−7.3 → −5.8 dB) and reproduced F0 exactly (93.8 Hz in, 93.8 Hz out). LTX
**resynthesises** rather than copies, so a small timbre difference is
inherent and not tunable — but it is not where the voice was lost.

# 4. The enrolled voice itself — upstream (mlx-voxtral-swift)

The voice embedding is an audio *prefix* the model continues, so every
characteristic of the prepared reference is learned. Three defects, all in
`VoxtralVoiceEnrollment.prepareReference`:

- **The 70 Hz high-pass destroyed the fundamental.** Implemented as
  `x − lowPass(x)` with a 64-tap kernel; at 24 kHz that resolves ~375 Hz, so
  the real response was −27 dB at 100 Hz, −23.9 dB at 120 Hz. Reproduced by
  applying it to a healthy preset voice: F0/H2 +0.9 → −10.5 dB.
- No loudness normalisation (references at −38 dB RMS → −38 dB syntheses).
- A hard gate writing exact zeros, learned as chopped micro-gaps.

Plus a silent NaN divergence in the optimiser that saved the corrupt
embedding (→ 198 s of runaway babble). Fixed upstream in
[mlx-voxtral-swift#44](https://github.com/VincentGourbin/mlx-voxtral-swift/pull/44)
(Butterworth biquad run forward–backward, active-RMS normalisation, soft
gate, divergence guard); follow-ups tracked in issue #45.

After the fix, a voice enrolled from a clean mic recording carried its pitch
exactly (87.9 Hz source → 87.9 Hz synthesis) at preset-comparable level.

# Method notes worth keeping

- **F0 vs H2** (energy at the fundamental against the second harmonic,
  median over voiced frames) is the discriminating metric for "thin" voices.
  A plain F0 tracker is not: with a depleted fundamental, autocorrelation and
  HPS both lock onto 2×F0 and report a plausible-looking wrong number.
- Always transcribe both the reference and the generated audio (STT) before
  measuring sync — it catches prompt/audio mismatches instantly.
- Attribution needs a **healthy control**: the preset-voice run is what
  proved the pipeline could hit 5 ms accuracy, and therefore that the custom
  path — not the model — was at fault.
