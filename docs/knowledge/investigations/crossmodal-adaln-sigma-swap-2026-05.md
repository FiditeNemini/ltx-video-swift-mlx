---
type: Investigation
title: LipDub mouth-modulation failure — cross-modal AdaLN sigma swap (May 2026)
description: Each cross-modal AdaLN was fed its OWN modality's sigma instead of the OTHER's, and the gate input was scaled 1000× wrong; two plausible hypotheses (RoPE negatives, LoRA delta) were numerically refuted along the way. Two follow-ups (2026-08-31) — the scale/shift half of the May fix was backwards, and scale/shift was also wrongly collapsed from per-token to one broadcast value — see the updates at the end.
tags: [lipdub, adaln, cross-modal, debugging, root-cause]
timestamp: 2026-07-16T00:00:00Z
---

LipDub produced wrong mouth modulation despite a provably correct LoRA fusion
and video anchoring. The campaign that root-caused it is worth keeping
because two *very* plausible hypotheses were expensively refuted first.

# The bug (two-fold, in LTX2Transformer.forward)

1. **Wrong sigma source**: each cross-modal AdaLN
   (`avCrossAttn{Video|Audio}{ScaleShift|Gate}`) was fed its own modality's
   timesteps. The Python reference feeds the **opposite** modality's scalar
   sigma (`video_preprocessor.prepare(video, audio)` /
   `audio_preprocessor.prepare(audio, video)` — `ltx-core model.py:402-403`).
   **Correction (2026-08-31, see the updates below): this is only true for the
   GATE half of each pair. The SCALE/SHIFT half wants the modality's OWN
   sigma** — the May fix below pointed both the same way, which correctly
   fixed the gate but broke scale/shift.
2. **Missing `av_ca_factor` on the gate**: Python scales the GATE AdaLN input
   by `av_ca_timestep_scale_multiplier / timestep_scale_multiplier` (= 1/1000
   with defaults). We passed `sigma × 1000`; correct is `sigma × 1`.

Why only LipDub expressed it: T2V+audio has equal sigmas (swap is a no-op);
I2V+audio's frame-0 AdaLN(0) lands on a latent that conditioning overwrites
anyway. LipDub has BOTH per-token timesteps and σ=0 reference tokens — the
bug fully expressed there.

# Hypotheses refuted (with parity harnesses)

- **RoPE on negative positions**: `precomputeFreqsCisDoublePrecision`
  matches Python to 3e-7 on the extended audio grid ([-2.0, +1.95]) —
  `RoPENegativePositionTests`.
- **LoRA delta direction**: Swift `getDelta` matches PEFT `(B @ A)`
  byte-for-byte (max abs diff 9.1e-7) — `LoRADeltaParityTests`.

# Quantitative validation of the fix

Pearson correlation of audio envelope vs mouth openness over 121 frames on
the Lightricks teaser: source (ground truth) +0.165/+0.298 (lagged);
Lightricks-FR −0.047/+0.056; **ours with the fix +0.140/+0.148**. On
end-of-clip silence our pipeline closes the mouth (openness 4.84 vs their
6.47 still open). Residual trade-off, understood and accepted: our output is
*audio-anchored*, Lightricks is *pose-anchored* — the remaining gap is pose
preservation, not lip-sync, and is numerical accumulation, not a code bug.

# Related

The [stereo pitfall](/docs/knowledge/pitfalls/audio-must-stay-stereo.md) was
root-caused in the same campaign.

# Update (2026-08-31): the "wrong sigma source" fix above was half-backwards

Issue #57 sub-task 5 built the first *element-wise* reference for this exact
code — a small `LTXModel(model_type=AudioVideo)` run through Lightricks' own
forward pass, with **deliberately different sigmas per stream** (0.7 video,
0.3 audio; equal sigmas make a swap a no-op, which is exactly why this needed
a harness rather than eyeballing a real generation). It found that bullet 1
above was itself wrong: only the **gate** AdaLNs
(`av_ca_{a2v,v2a}_gate_adaln_single`) take the *opposite* modality's sigma.
The **scale/shift** AdaLNs (`av_ca_{video,audio}_scale_shift_adaln_single`)
take the modality's **own** sigma — the May fix pointed both pairs the same
way, so it corrected the gate (previously own-sigma, now fixed) but broke the
scale/shift pair (previously already correct via the original bug's
symmetry, now cross-sigma and wrong).

Measured directly against each AdaLN module in isolation
(`DualStreamAudioParityTests.crossModalAdaLNInputsMatchReference`):

| module | own-sigma error | cross-sigma error | reference uses |
|---|---|---|---|
| `av_ca_video_scale_shift_adaln_single` | 3.1e-6 | 0.545 | own |
| `av_ca_audio_scale_shift_adaln_single` | 7.9e-7 | 0.262 | own |
| `av_ca_a2v_gate_adaln_single` | 0.066 | 1.5e-7 | cross |
| `av_ca_v2a_gate_adaln_single` | 0.015 | 3.4e-8 | cross |

On the full forward pass this dropped the video/audio output relative error
from 3.6e-3 / 8.2e-3 to 2.1e-6 / 1.2e-6 — both were already under the
harness's 2% pass/fail threshold even with the bug present, which is exactly
the plan's warning that a combined-output threshold is not sensitive enough
here: the fix would have shipped unverified on output magnitude alone.

**Real end-to-end sanity check**: `retake --modality audio` (video frozen at
σ=0, audio denoising from σ=1 — the same divergent-sigma shape as LipDub) on
the same source clip and seed, before and after, at the real 22B scale: RMS
0.045 → 0.0097 (−4.6×), peak 0.125 → 0.040, and the 0-2 kHz band down 16.6 dB.
A real, large, audible difference in exactly the scenario this bug requires
divergent sigma to express — consistent with the May investigation's own
account of why plain T2V+audio (matched sigma) never showed it.

Fix: `Sources/LTXVideo/Models/Transformer/LTX2Transformer.swift`, swap only
the scale/shift assignment back to own-modality sigma, leaving the gate
assignment (already fixed in May) untouched.

# Update 2 (2026-08-31): scale/shift was also collapsed to one broadcast value

A `/code-review`'s gap-sweep pass on the PR carrying the fix above caught a
second, independent defect in the same lines, verified directly against
`ltx_core.model.transformer.transformer_args._prepare_cross_attention_timestep`:
the scale/shift AdaLN input isn't just "the wrong sigma" — even after
pointing it at the *own* modality, the port fed it `videoTimesteps.max(axis:
1)`, a single value collapsed from the modality's own **per-token**
timesteps and broadcast identically over every token. The reference keeps it
per-token: it flattens `modality.timesteps` (shape `(B, T)`) into the AdaLN
call and reshapes the output back to `(B, T, 4, D)` — one genuinely distinct
scale/shift value per token, not one value shared by all of them. Only the
GATE input is a true batch scalar (`Modality.sigma`, `(B,)`); collapsing
*that* to `.max(axis: 1)` is a legitimate approximation in the absence of a
separate sigma parameter, since real denoising tokens all carry the current
step's sigma while conditioning/guide tokens sit below it, and `.max` recovers
it exactly.

This matters precisely where `buildExtendedTimestep`
(`Sources/LTXVideo/Pipeline/AppendedGuideTokens.swift`) already builds
genuinely non-uniform per-token timesteps for production use — real denoising
tokens at the schedule's sigma, appended guide/keyframe tokens pinned at
0 — i.e. every keyframe, IC-LoRA and LipDub-audio-reference generation. The
original harness's fixture used uniform per-token timesteps throughout
(matching a plain T2V+audio prompt), which cannot distinguish a correctly
per-token AdaLN call from one collapsed to a single broadcast value — the
same "equal inputs make a swap a no-op" blind spot as the original bug, one
level down.

**Fix**: `AudioTransformerArgs`' `crossVideoScaleShift`/`crossAudioScaleShift`
fields now carry the genuinely per-token `(B, T, 4, D)` embedding, split out
from a new, separate `crossVideoGate`/`crossAudioGate` field carrying the
scalar-broadcast `(B, 1, 1, D)` gate embedding — the two could not stay fused
into one `(B, 1, 5, D)` tensor (the pre-fix design) once scale/shift stopped
being uniform across tokens. `LTX2TransformerBlock`'s cross-modal phase reads
the two fields separately instead of slicing indices out of one combined
tensor.

**Verification**: `scripts/transformer_reference.py`'s "av" branch and
`DualStreamAudioParityTests` were both updated to use non-uniform per-token
timesteps (half the video tokens and one of five audio tokens held below the
modality's active sigma). Reverting the fix locally and re-running confirmed
the new fixture actually catches it: `videoAndAudioOutputsMatchReferenceSeparately`
goes from ~1e-6 (clean) to 1.1e-3 / 5.4e-3 (collapsed-to-scalar) — still
under this suite's old 2% threshold, so that threshold was tightened to
2e-4 (matching `TransformerParityTests`'s video-only precision, appropriate
for a pure float32 synthetic model with no legitimate large noise source).

# Update 3: on the plan's own warning about output-only thresholds

Both regressions found in this file — the own/cross sigma swap and the
per-token collapse — measured *below* the family's usual 2% pass/fail
threshold on the full forward pass, even while a synthetic, deliberately
adversarial input (divergent sigmas, non-uniform per-token timesteps) was
already in use. A parity suite whose isolated per-module checks are tight
(this file's checks run at ~1e-6) is what actually caught both; the combined
end-to-end number alone would have shipped both.
