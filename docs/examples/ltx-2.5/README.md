# LTX-2.5 bring-up — validated results (August 2026)

Every clip here was generated on-device by this package and user-validated.
All runs: 768×512, bf16 unless noted, conditioning image `conditioning-frame.png`
(frame 0 of an LTX-2.5 reference generation). The measured findings behind these
clips live in [`docs/knowledge/`](../../knowledge/index.md).

| File | What it evidences | Settings |
|---|---|---|
| `generate-25-i2v-audio-337f.mp4` | LTX-2.5 i2v two-stage with audio, at the duration the checkpoint's own head predicts (14.04 s vs 13.375 s from the reference service) | 2.5-distilled, 337f, seed 42, BigVGAN vocoder 48 kHz |
| `generate-23-control.mp4` | LTX-2.3 two-stage unchanged after the 2.5 work — same prompt/image/seed as the 2.5 run | distilled (2.3), 121f, seed 42 |
| `lora23-arcshot-on-25.mp4` | A 2.3-trained camera LoRA (attention-only, 384 modules) produces its arc on the 2.5 checkpoint; prompt requested no motion | 2.5-distilled, 121f, seed 7 |
| `transition-23.mp4` / `transition-25.mp4` | A 2.3-trained transition LoRA (attention + FFN, 576 modules) works on both generations — same keyframes, prompt, seed; only the checkpoint differs. FFN is the one layer where 2.5 diverges structurally (bias-free), so this closes the cross-generation LoRA question behaviourally | joyfox/LTX-2.3-Transition-LORA, keyframes `transition-video-A` last frame → `transition-video-B` first frame, trigger `zhuanchang`, seed 300 |
| `transition-video-A.mp4` / `transition-video-B.mp4` | The two source clips for the transition | distilled (2.3), seeds 100 / 200 |
| `transition-compare-strip.png` | Frames 0/30/60/90/120, top 2.3 / bottom 2.5 |  |
| `upscale-source-384x256.mp4` → `upscale-25-stage1.mp4` → `upscale-25-final-768x512.mp4` | The pixel spatial upscaler chain on 2.5: 8-step stage 1 with the IC-LoRA at source resolution, latent upscale, 3-step refinement. Subject identity holds end-to-end — see the [stage-2 decision](../../knowledge/decisions/iclora-stage2-keeps-adapter-and-reference.md) for why adapter and reference both stay active | 2.5-distilled, x2 1.0 adapter, seed 42 |
| `lipdub-23-bigvgan-vocoder.mp4` | LipDub (2.3) through the checkpoint's real vocoder — 48 kHz, BigVGAN + bandwidth extension | distilled (2.3), 121f, seed 42 |

Known cosmetic caveats, deliberate (they document real behaviour):
- The transition clips open on a dark blurred close-up: that is genuinely the
  last frame of video A, which drifts at its end. Keyframes anchor only the
  endpoints — check hinge frames before using them as anchors.
- `transition-video-A`'s car is not held mid-transition for the same reason:
  the frame-0 anchor was unreadable, so mid-clip content came from the prompt.
