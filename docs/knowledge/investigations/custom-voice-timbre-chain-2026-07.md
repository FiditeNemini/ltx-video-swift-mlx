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

> **August 2026 caveat.** The probes below compare an input against its
> generation *through the same decoder*, so a deficiency common to both largely
> cancels — and the decoder in question was the wrong one. LTX-2.3 bundles a
> BigVGAN v2 + bandwidth-extension vocoder that this package was not running; the
> stage actually used left 1–4 kHz some 56 dB below total. See
> [the vocoder pitfall](/docs/knowledge/pitfalls/wrong-vocoder-lost-the-midrange.md).
> These measurements stand as taken; the conclusions attributed upstream deserve
> a re-run with the correct vocoder before being treated as settled.


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

# Closed out by the Voxtral team (26 July)

Three findings that settle the loose ends, two of which correct what this
investigation had assumed:

- **q6 is the right variant for cloned voices**, not bf16: 99.4 % vs 96.5 %
  coverage, RTF 1.47 vs 3.44, 3.5 GB vs 8 GB. A dropped word observed once
  during this campaign was generation variance; recommending bf16 from that
  single sample was wrong and has been withdrawn upstream.
- **The exact digital zeros come from the codec**, not from the enrollment
  gate, and vary **3.4 %–10.5 % between generations of the same voice**. They
  are therefore not a per-embedding property and no consumer may assume a
  natural floor. Ours does not — see rule 7 of
  [the audio contract](/docs/knowledge/pitfalls/lipdub-audio-contract.md).
- **The residual ~5 dB of fundamental is inherent**: the enrolled codes are
  faithful, generation is where it is lost. No upstream fix is pending, and a
  synthesis a few dB under its own reference is expected.

The lesson on our side is the first one: an n=1 observation was written into a
recommendation on someone else's tracker, and it pointed the wrong way. Sample
size belongs in the finding, or the finding does not get filed.

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
