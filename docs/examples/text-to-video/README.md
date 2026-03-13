# Text-to-Video — LTX-2.3 Distilled Two-Stage Pipeline

First validated use case of the LTX-2.3 Swift/MLX port.

## Pipeline Architecture

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Prompt Enhancement (optional)                  │
│  Gemma 3 12B VLM (4-bit QAT, ~7.5 GB)          │
│  Rewrites short prompt into detailed scene      │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Text Encoding                                  │
│  Gemma 3 12B → 49 hidden states (layers 1-47    │
│    + norm(layer_47))                            │
│  → Feature Extractor V2 (RMS norm + rescale)    │
│  → Connector (attention + MLP, 3840 → 4096)     │
│  Output: text embeddings [1, 1024, 4096]        │
└───────────────┬─────────────────────────────────┘
                │  ◄── Gemma unloaded from memory
                ▼
┌─────────────────────────────────────────────────┐
│  Stage 1 — Half Resolution Denoising            │
│  LTX-2.3 Transformer (48 blocks, 22B params)    │
│  + Distilled LoRA (rank 384, fused at load)     │
│  8 Euler steps, predefined sigmas               │
│  No CFG (guidance scale = 1.0)                  │
│  Input: noise at W/2 × H/2                      │
│  Output: latents [1, 128, T_lat, H/2, W/2]     │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Spatial Upscaler (2x)                          │
│  Denormalize → Upscale → Renormalize → AdaIN    │
│  Output: latents [1, 128, T_lat, H, W]          │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Stage 2 — Full Resolution Refinement           │
│  Same transformer + LoRA                        │
│  3 Euler steps, sigmas [0.909, 0.725, 0.422]    │
│  Re-noise upscaled latent with σ=0.909          │
│  Output: refined latents at full resolution     │
└───────────────┬─────────────────────────────────┘
                │  ◄── Transformer unloaded from memory
                ▼
┌─────────────────────────────────────────────────┐
│  VAE Decoder                                    │
│  128ch latent → 9 up_blocks (ResBlocks +        │
│  DepthToSpace upsampling) → 48ch → unpatchify   │
│  → 3ch RGB frames                               │
│  Temporal tiling for long videos (>64 frames)   │
│  Output: [B, 3, T_frames, H, W]                │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  MP4 Export (AVFoundation, 24fps)               │
└─────────────────────────────────────────────────┘
```

### Key Model Components

| Component | Source | Size |
|-----------|--------|------|
| Gemma 3 12B VLM | `mlx-community/gemma-3-12b-it-qat-4bit` | ~7.5 GB |
| LTX-2.3 Distilled (unified) | `Lightricks/LTX-2` → `ltx-2.3-22b-distilled.safetensors` | ~22 GB |
| Distilled LoRA | `Lightricks/LTX-2` → `ltx-2-19b-distilled-lora-384.safetensors` | ~1.5 GB |
| Spatial Upscaler | `Lightricks/LTX-2` → `latent_upsampler/diffusion_pytorch_model.safetensors` | ~50 MB |

All weights are auto-downloaded on first run.

---

## Examples

### 1. Quick Test — 768x512, 9 frames

Short generation to validate the pipeline.

```bash
ltx-video generate \
    "A beaver building a dam in a peaceful forest stream, golden hour lighting" \
    -w 768 -h 512 -f 9 \
    --seed 1953802378 --enhance-prompt \
    -o t2v-768x512-9f.mp4
```

| Parameter | Value |
|-----------|-------|
| Resolution | 768x512 (stage 1: 384x256) |
| Frames | 9 (0.4s at 24fps) |
| Steps | 8 (stage 1) + 3 (stage 2) = 11 total |
| Seed | 1953802378 |
| Prompt enhancement | Yes (Gemma 3 12B) |
| Inference time | *work in progress — full benchmark pending LTX 2.3 adaptation* |

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/text-to-video/t2v-768x512-9f.mp4" controls width="512"></video>

---

### 2. Full Generation — 1024x576, 10 seconds

Full-length high-resolution generation matching the HuggingFace Space output quality.

```bash
ltx-video generate \
    "A beaver building a dam in a peaceful forest stream, golden hour lighting" \
    -w 1024 -h 576 -f 241 \
    --seed 1953802378 --enhance-prompt \
    -o t2v-1024x576-10s.mp4
```

| Parameter | Value |
|-----------|-------|
| Resolution | 1024x576 (stage 1: 512x288) |
| Frames | 241 (10.0s at 24fps) |
| Steps | 8 (stage 1) + 3 (stage 2) = 11 total |
| Seed | 1953802378 |
| Prompt enhancement | Yes (Gemma 3 12B) |
| Inference time | *work in progress — full benchmark pending LTX 2.3 adaptation* |

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/text-to-video/t2v-1024x576-10s.mp4" controls width="640"></video>

---

## Hardware

- Apple Silicon M3 Max 96GB
- macOS 15 (Sequoia)
- Inference times will be benchmarked after full LTX 2.3 adaptation is complete
