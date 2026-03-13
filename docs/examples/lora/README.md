# LoRA Inference — LTX-2.3 Camera LoRA (Arcshot)

Applying a custom LoRA during video generation to control camera movement.

## How It Works

LoRA weights are **fused into the transformer** after model loading and before denoising. The fusion modifies attention projection weights (to_q, to_k, to_v, to_out for self-attention and cross-attention) across all 48 transformer blocks — 384 weight updates total.

The `--lora` flag accepts any `.safetensors` LoRA file compatible with LTX-2.3 22B. The `--lora-scale` parameter (default: 1.0) controls the strength of the adaptation.

```
W' = W + scale * (B @ A)
```

## LoRA Source

Camera LoRAs from [squareyards/LTX-2.3-camera-lora](https://huggingface.co/squareyards/LTX-2.3-camera-lora):
- `ltx-2.3-22b-lora-camera-arcshot.safetensors` (192 MB, arc shot camera movement)
- `ltx-2.3-22b-lora-camera-kenburn.safetensors` (384 MB, Ken Burns effect)

---

## Comparison: With vs Without LoRA

Same prompt, same seed, same input image — the only difference is the arcshot LoRA.

### Input Image

A red Citroën 2CV at 768x512.

![Input image](../image-to-video/input_sample.jpg)

### Prompt

```
arc shot, camera moving from left to right around the subject,
a red vintage car parked on a gravel road surrounded by green foliage
```

### With Arcshot LoRA

```bash
ltx-video generate \
    "arc shot, camera moving from left to right around the subject, a red vintage car parked on a gravel road surrounded by green foliage" \
    --image input_768x512.png \
    --lora ~/Library/Caches/ltx-video-mlx/loras/ltx-2.3-22b-lora-camera-arcshot.safetensors \
    -w 768 -h 512 -f 121 --seed 42 --enhance-prompt \
    -o i2v-arcshot-v2-lora.mp4
```

[![With LoRA](arcshot-lora-thumb.png)](https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/lora/i2v-arcshot-v2-lora.mp4)

*Click to download and play. The camera orbits around the car (arc shot movement).*

### Without LoRA

```bash
ltx-video generate \
    "arc shot, camera moving from left to right around the subject, a red vintage car parked on a gravel road surrounded by green foliage" \
    --image input_768x512.png \
    -w 768 -h 512 -f 121 --seed 42 --enhance-prompt \
    -o i2v-arcshot-v2-nolora.mp4
```

[![Without LoRA](arcshot-nolora-thumb.png)](https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/lora/i2v-arcshot-v2-nolora.mp4)

*Click to download and play. Standard camera behavior without the LoRA.*

---

## Parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 768x512 (two-stage: 384x256 → 768x512) |
| Frames | 121 (5.0s at 24fps) |
| Steps | 8 (stage 1) + 3 (stage 2) = 11 total |
| Seed | 42 |
| LoRA scale | 1.0 |
| LoRA layers fused | 384 (8 per block × 48 blocks) |
| Prompt enhancement | Yes (multimodal I2V via Gemma VLM) |
| Inference time | ~430s (M3 Max 96GB) |

## CLI Usage

```bash
# Any LTX-2.3 compatible LoRA
ltx-video generate "your prompt" \
    --lora /path/to/lora.safetensors \
    --lora-scale 1.0

# With image conditioning
ltx-video generate "your prompt" \
    --image input.png \
    --lora /path/to/lora.safetensors

# Adjust strength (0.5 = subtle, 1.0 = full effect)
ltx-video generate "your prompt" \
    --lora /path/to/lora.safetensors \
    --lora-scale 0.5
```

## Hardware

- Apple Silicon M3 Max 96GB
- macOS 26.3 (Tahoe)
