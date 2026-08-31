---
type: Pitfall
title: The vocoder's own float32 policy only ever cast the runtime input, never its checkpoint weights
description: Every real LipDub/audio generation ran BigVGAN's ~108-conv chain with bf16 conv weights despite a comment claiming float32 execution; 2-9% relative error, collapsing to ~1e-5 once the loader casts the parameters too.
tags: [audio, vocoder, bigvgan, precision, root-cause, parity-harness, high-impact]
timestamp: 2026-08-31T00:00:00Z
---

`LTXVocoderWithBWE.swift` carries an explicit comment: "Both stages run in
**float32**. The reference is explicit that bf16 accumulation across the
~108 sequential convolutions degrades spectral metrics by 40-90%, and this
is the one place in the pipeline where that matters enough to break from the
bf16 default." The code backing that claim only cast the *runtime activation*
handed to the base vocoder (`melSpectrogram.asType(.float32)` in
`callAsFunction`) — every `Conv1d`/`ConvTransposed1d` parameter loaded by
`BigVGANWeightLoader.load` stayed at the checkpoint's native bf16
(`ltx-2.5-audio-vae-bf16.safetensors`). MLX's conv ops accept the mixed
dtypes without erroring, so nothing signaled the gap — but the intended
float32 execution never actually happened for the weights, only the first
activation.

Lightricks' own reference forces this the other way: `VocoderWithBWE.forward`
runs its entire pass — weights included — under `torch.autocast(dtype=torch.float32)`
(or `model.float()` on backends that don't support that autocast), which is
exactly the departure from bf16 the Swift comment describes but doesn't
implement.

# The defense

- A comment describing a precision policy is a claim about the *code*, not
  documentation of *intent* — verify it by checking what dtype the weights
  actually end up at after loading, not just what the runtime call site
  casts. Same failure shape as
  [the connector's reordering bug](connector-register-replacement-reorders-tokens.md):
  a stated behavior that the implementation didn't fully carry out.
- The bug was invisible without an element-wise reference: the base vocoder's
  own comment already predicted "40-90% degradation" from bf16 accumulation,
  so a ~2-9% relative error read as *plausible* accumulated bf16 noise rather
  than a fixable bug, until [`AudioVAEVocoderParityTests`](../../../Tests/LTXVideoTests/AudioVAEVocoderParityTests.swift)
  (issue #57 sub-task 4) bisected the audio decode chain and showed the AudioVAE
  decoder upstream of the vocoder — same bf16-checkpoint weights, far fewer
  sequential convs — held to ~1e-7. The order-of-magnitude difference between
  a few conv layers and a ~108-conv chain, both loaded identically, was the
  signal that depth-dependent bf16 accumulation (not a "known" precision
  ceiling) was doing the damage, and that damage was fixable by actually
  finishing the float32 cast.
- The bandwidth-extension residual's small output magnitude (mean |x| ≈ 7.6e-4,
  roughly 17× smaller than the base vocoder's ≈1.3e-2) makes its *relative*
  error read far worse than the base vocoder's for the same absolute noise —
  8.8% vs 2.0% before the fix, for what is the same underlying precision bug.
  Don't judge two taps of very different magnitude by relative error alone
  without checking the scale each is relative to.

# Fix

`BigVGANWeightLoader.load` now casts every checkpoint value to float32 before
the single `model.update(parameters:)` call:

```swift
for key in updates.keys { updates[key] = updates[key]!.asType(.float32) }
model.update(parameters: ModuleParameters.unflattened(updates))
```

# Real end-to-end sanity check

Built the Release binary at both the pre-fix commit (a `git worktree` at
`dd90c67d`) and the fix, and ran the identical real generation on each —
`generate "A woman singing a short melody in a sunlit room, close-up of her
mouth" --model 2.5-distilled --audio --frames 49 --width 512 --height 512
--seed 42`, same weights, same seed. Unlike
[wrong-vocoder-lost-the-top-octave.md](wrong-vocoder-lost-the-top-octave.md)
— a *structural* bug (the wrong module entirely) that showed up as missing
frequency bands — this is a *precision* bug (bf16 rounding noise compounding
through the conv chain), and the two read very differently on a real
waveform: per-band FFT energy across 0-24 kHz matched to within 0.02-0.04 dB
in every band except the top octave (20-24 kHz, where absolute energy is
three orders of magnitude below the bass band and dB is correspondingly
noisy), and RMS/peak were unchanged to 4 decimal places. The two waveforms
still differ sample-by-sample — mean absolute difference relative to the
fixed waveform's own mean magnitude is 1.56%, the same order as the
pre-fix `waveform` parity error (2.05e-2) measured against the Python
reference — but that difference reads as *added grain*, not missing content:
plausible enough on a single listen that this class of bug can live
undetected exactly as long as it did.

# Citations

[1] `AudioVAEVocoderParityTests.bisectFirstDivergence` /
`waveformOutputMatchesReference`, against `scripts/audio_vae_reference.py`
run on CPU float32 via Lightricks' own `AudioDecoder` + `VocoderWithBWE`
(2026-08-31). Measured relative error, before → after the weight cast:
`vocoder_base` 1.97e-2 → 1.30e-5, `vocoder_bwe_residual` 8.83e-2 → 1.13e-4,
full `waveform` 2.05e-2 → 1.48e-5. The `AudioDecoder` taps upstream (same
bf16-checkpoint weights, no cast) held at ~1e-7 throughout, unaffected since
its chain is far shorter.
