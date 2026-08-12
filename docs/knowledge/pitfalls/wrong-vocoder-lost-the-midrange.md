---
type: Pitfall
title: The LTX-2 vocoder was decoding LTX-2.3 / 2.5 latents, and it cost 40 dB of midrange
description: This package loaded a 194-tensor vocoder from Lightricks/LTX-2 while both LTX-2.3 and LTX-2.5 ship a 667+557-tensor BigVGAN v2 with bandwidth extension. Same latent space, so it produced plausible audio — with the 1–4 kHz band sitting 56 dB below total.
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

That last point is why this survived so long. The latent space is shared, and the
LTX-2 vocoder was legitimately trained on it, so it decoded that space into
*plausible* audio rather than noise. Nothing failed; the load reported success.

# What it actually cost

Band energy relative to total, same generated clip, first 4 s:

| Band | LTX-2 vocoder (24 kHz) | Checkpoint's vocoder (48 kHz) |
|---|---|---|
| 0–1 kHz | 0 dB | 0 dB |
| **1–4 kHz** | **−56.6 dB** | **−16.2 dB** |
| 4–8 kHz | −67.5 dB | −20.3 dB |
| 8–12 kHz | −67.4 dB | −35.7 dB |
| 12–16 kHz | −79.4 dB | −44.4 dB |
| 16–24 kHz | absent | −59.1 dB |

Essentially everything below 1 kHz and nothing above. Not a degraded signal — a
missing one, in the band where a sound becomes recognisable.

# Consequence for the LipDub record

The vocoder is identical in the 2.3 bundle, so **every LipDub track this package
produced went through the amputated stage**. The
[July 2026 timbre investigation](/docs/knowledge/investigations/custom-voice-timbre-chain-2026-07.md)
cleared the audio decoder on three probes, one of which was energy above 8 kHz
(−21.3 → −21.8 dB, input vs generated). That measurement stands as taken — it
compared an input against its generation through the same decoder, so a decoder
deficiency common to both largely cancels out. It did not, and could not, see the
1–4 kHz hole.

**Open**: the custom-voice conclusions that were attributed upstream deserve a
re-measurement with the correct vocoder before being treated as settled.

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
