---
type: Pitfall
title: The LTX-2 vocoder was decoding LTX-2.3 / 2.5 latents, and it cost the top octave
description: This package loaded a 194-tensor vocoder from Lightricks/LTX-2 while both LTX-2.3 and LTX-2.5 ship a 667+557-tensor BigVGAN v2 with bandwidth extension. Same latent space, so it decoded plausibly — losing 18 dB above 12 kHz and everything above 16 kHz.
tags: [audio, vocoder, ltx-2.5, ltx-2.3, lipdub, measurement]
timestamp: 2026-08-12T00:00:00Z
---

`loadAudioModels` downloaded `Lightricks/LTX-2`'s standalone vocoder
(`conv_in`, `resnets`, `upsamplers` — 194 tensors, 24 kHz out). Both LTX-2.3 and
LTX-2.5 bundle a different one: BigVGAN v2 (`conv_pre`, AMP blocks, SnakeBeta
activations, anti-aliased resampling — 667 tensors) followed by a
bandwidth-extension generator (557 tensors), 48 kHz out.

Measured, not inferred:

* the two vocoders share **zero key names**
* the 2.3-bundled and 2.5-bundled vocoders are **byte-identical** (1227 tensors,
  zero differing values) — one implementation serves both
* the audio VAE is **byte-identical** across LTX-2, 2.3 and 2.5 (100 common
  tensors, zero differences)

That last point is why this survived so long. The latent space is shared and the
LTX-2 vocoder was legitimately trained on it, so it decoded into *plausible*
audio rather than noise. Nothing failed; the load reported success.

# What it actually costs

A/B on one generation — same prompt, same seed, same 337 frames, only the
vocoder differs. Band energy relative to total:

| Band | LTX-2 vocoder (24 kHz) | Checkpoint's vocoder (48 kHz) | Δ |
|---|---|---|---|
| 0–1 kHz | −0.5 dB | −0.7 dB | −0.2 |
| 1–4 kHz | −10.8 dB | −9.6 dB | +1.2 |
| 4–8 kHz | −15.5 dB | −14.7 dB | +0.8 |
| 8–12 kHz | −23.8 dB | −26.0 dB | −2.1 |
| 12–16 kHz | −54.0 dB | −35.9 dB | **+18.0** |
| 16–24 kHz | absent (24 kHz ceiling) | −51.0 dB | — |

Confirmed on speech through a 2.3 LipDub run against the shipped teaser
(same pipeline, same reference video): 1–4 kHz −11.7 → −10.7 dB, 12–16 kHz
−52.2 → −37.9 dB. **The effect is the top octave, not the midrange.**

So the mismatch cost air and detail, not intelligibility. Worth fixing — the
bandwidth-extension stage exists precisely to produce that band — but existing
LipDub output was not broken by it.

# A measurement trap this exposed, twice

The first comparison run here reported a *40 dB* gap in 1–4 kHz and concluded
the midrange had been missing all along. It was wrong: it compared a 337-frame
generation against a **121-frame** one. Different frame counts are different
generations, so the audio content differed — a long steady drone against a
compressed action scene — and the content difference was attributed to the
vocoder.

Same failure mode as changing duration and audio in one run earlier the same
day, and again that evening when a pixel upscaler's framing was called "shifted"
from eyeballed crops and a bounding box that a single stray red pixel inflates —
a mass-percentile measure then showed subject scale and position tracking the
source to within 2%.

**When A/B-ing a stage, hold everything else fixed** (same prompt, same seed,
same frame count) **and measure with a statistic that outliers cannot move**. If
the old path is no longer reachable, keep a sample of its output rather than
re-generating at different settings. Three times in one day, an uncontrolled
comparison produced a confident wrong conclusion; each time the fix was cheaper
than the retraction.

# Porting notes worth keeping

* **float32, deliberately.** Upstream documents that bf16 accumulation across the
  ~108 sequential convolutions degrades spectral metrics by 40–90%. This is the
  one stage where the pipeline departs from its bf16 default.
* **No filter is recomputed.** Every Kaiser kernel and DFT basis is a persistent
  buffer in the checkpoint, so loading them verbatim removes any window-convention
  drift. The single exception is the BWE skip resampler (`persistent=False`
  upstream), derived in Swift and named explicitly in the loader's whitelist so a
  genuinely missing filter still fails.
* **Base upsampling is ×160**, not ×320: `5·2·2·2·2·2`, one sample per 10 ms hop
  at 16 kHz, which is exactly the audio VAE's `hopLength`. The BWE stage then
  triples the rate to 48 kHz.
* Snake α/β are stored in **log scale**, so a checkpoint value of 0 means α = 1.
