# Image-to-Video — LTX-2.3 Distilled Two-Stage Pipeline

Second validated use case: generate a video from a single input image.

## Pipeline Architecture

```mermaid
flowchart TD
    A["Input Image + Text Prompt"] --> B

    subgraph enhance ["Prompt Enhancement (optional)"]
        B["Gemma 3 12B VLM\n4-bit QAT · multimodal I2V\nanalyzes image + rewrites prompt"]
    end

    B --> C

    subgraph encoding ["Text Encoding"]
        C["Gemma 3 12B → 49 hidden states"] --> D["Feature Extractor V2 + Connector\n→ text embeddings 1024 × 4096"]
    end

    D -->|"⚡ Gemma unloaded"| E

    subgraph imageenc ["Image Conditioning"]
        E["VAE Encoder\nimage → latent frame 0\n+ conditioning mask (frame 0 = clean)"]
    end

    E --> F

    subgraph stage1 ["Stage 1 — Half Resolution"]
        F["LTX-2.3 Transformer (22B)\n+ Distilled LoRA · 8 Euler steps\nframe 0 frozen (per-token timestep)\nnoise at W/2 × H/2"]
    end

    F --> G

    subgraph upscale ["Spatial Upscaler"]
        G["Denormalize → 2x Upscale → Renormalize → AdaIN"]
    end

    G --> H

    subgraph stage2 ["Stage 2 — Full Resolution Refinement"]
        H["Same Transformer + LoRA\n3 Euler steps · σ = 0.909 → 0.725 → 0.422\nframe 0 re-encoded at full resolution"]
    end

    H -->|"⚡ Transformer unloaded"| I

    subgraph decode ["VAE Decoder"]
        I["128ch → 9 up_blocks → 3ch RGB\ntemporal tiling for long videos"]
    end

    I --> J["MP4 Export · 24fps"]

    style enhance fill:#f3e8ff,stroke:#7c3aed
    style encoding fill:#e0f2fe,stroke:#0284c7
    style imageenc fill:#dbeafe,stroke:#2563eb
    style stage1 fill:#fef3c7,stroke:#d97706
    style upscale fill:#d1fae5,stroke:#059669
    style stage2 fill:#fef3c7,stroke:#d97706
    style decode fill:#fee2e2,stroke:#dc2626
```

### Difference from Text-to-Video

The I2V pipeline adds **image conditioning**: the input image is VAE-encoded into latent frame 0, which is kept clean (not noised) via a **conditioning mask** with per-token timesteps. The transformer denoises all other frames while keeping frame 0 frozen, ensuring the output video starts from the exact input image.

---

## Examples

### Input Image

A red Citroën 2CV (resized from 1920x1280 to 768x512 for pipeline input).

![Input image](input_sample.jpg)

### 1. Quick Test — 768x512, 9 frames

```bash
ltx-video generate \
    "The red vintage car's wheels fold up underneath, flames burst from the rear, and the car lifts off the ground into the sky like in Back to the Future, dramatic lighting, cinematic" \
    --image input_768x512.png \
    -w 768 -h 512 -f 9 \
    --seed 42 --enhance-prompt \
    -o i2v-768x512-9f.mp4
```

| Parameter | Value |
|-----------|-------|
| Resolution | 768x512 (stage 1: 384x256) |
| Frames | 9 (0.4s at 24fps) |
| Steps | 8 (stage 1) + 3 (stage 2) = 11 total |
| Seed | 42 |
| Prompt enhancement | Yes (multimodal I2V) |
| Inference time | ~39s (excl. model loading) |

[![I2V 768x512 preview](i2v-768x512-9f-thumb.png)](https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/image-to-video/i2v-768x512-9f.mp4)

*Click the image to download and play the video.*

---

### 2. Full Generation — 1024x576, 10 seconds

```bash
ltx-video generate \
    "The red vintage car's wheels fold up underneath, flames burst from the rear, and the car lifts off the ground into the sky like in Back to the Future, dramatic lighting, cinematic" \
    --image input_768x512.png \
    -w 1024 -h 576 -f 241 \
    --seed 42 --enhance-prompt \
    -o i2v-1024x576-10s.mp4
```

| Parameter | Value |
|-----------|-------|
| Resolution | 1024x576 (stage 1: 512x288) |
| Frames | 241 (10.0s at 24fps) |
| Steps | 8 (stage 1) + 3 (stage 2) = 11 total |
| Seed | 42 |
| Prompt enhancement | Yes (multimodal I2V) |
| Inference time | ~755s (~12.5 min, excl. model loading) |

[![I2V 1024x576 10s preview](i2v-1024x576-10s-thumb.png)](https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/image-to-video/i2v-1024x576-10s.mp4)

*Click the image to download and play the video.*

---

## Hardware

- Apple Silicon M3 Max 96GB
- macOS 26.3 (Tahoe)
- Inference times measured March 2026 (macOS 26.3, Release build)
