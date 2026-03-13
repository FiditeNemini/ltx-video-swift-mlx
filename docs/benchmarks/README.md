# Benchmarks — LTX-2.3 Swift/MLX on Apple Silicon

## Hardware

- **Apple M3 Max** — 16-core CPU, 40-core GPU, 96 GB unified memory
- **macOS 26.3** (Tahoe)
- Release build (Xcode, `-O` optimization)

## Test Configuration

All benchmarks use the same generation parameters:

| Parameter | Value |
|-----------|-------|
| Pipeline | Two-stage distilled (I2V + Audio) |
| Resolution | 1024x576 (stage 1: 512x288) |
| Frames | 241 (10.0s at 24fps) |
| Steps | 8 (stage 1) + 3 (stage 2) = 11 total |
| Seed | 42 |
| CFG | 1.0 (distilled, no classifier-free guidance) |
| Audio | Dual video/audio denoising |
| Prompt enhancement | Disabled (raw prompt) |

```bash
ltx-video generate \
    "A chic woman walks towards a red vintage car, opens the door, gets inside \
    and sits down. She starts the engine which roars like a lawnmower, then the \
    car drives away down the road." \
    --image input_768x512.png \
    -w 1024 -h 576 -f 241 --seed 42 --audio \
    --transformer-quant {bf16|qint8|int4} \
    --profile -o output.mp4
```

---

## Quantization Comparison

On-the-fly quantization (`--transformer-quant`) converts the LTX2Transformer weights after loading. The VAE, text encoder, and audio models remain in their native precision.

### Inference Time

| | **bf16** | **qint8** | **int4** |
|---|---|---|---|
| Stage 1 — half-res (8 steps) | 248s | 409s | 408s |
| Stage 2 — full-res (3 steps) | 766s | 918s | 778s |
| **Total generation** | **1145s** | **1458s** | **1294s** |
| Ratio vs bf16 | 1.0x | 1.27x slower | 1.13x slower |

### GPU Memory

| | **bf16** | **qint8** | **int4** |
|---|---|---|---|
| Peak GPU memory | 54.8 GB | 44.6 GB | 38.4 GB |
| Mean GPU (denoising) | 49.7 GB | 32.7 GB | 23.7 GB |
| Peak savings vs bf16 | — | -19% | -30% |
| Mean savings vs bf16 | — | -34% | **-52%** |

### Audio Quality

| | **bf16** | **qint8** | **int4** |
|---|---|---|---|
| Peak level | -11.7 dBFS | -12.2 dBFS | -11.9 dBFS |
| RMS level | -32.7 dBFS | -32.9 dBFS | -32.0 dBFS |

Audio levels are consistent across all quantization levels — no quality degradation detected.

---

## Analysis

### When to use each quantization level

| Config | Best for | Trade-off |
|--------|----------|-----------|
| **bf16** | 96 GB+ machines, maximum speed | Fastest inference, highest memory |
| **qint8** | 64 GB machines | ~27% slower, saves ~10 GB peak |
| **int4** | 32-64 GB machines | ~13% slower, halves denoising memory |

### Why quantization is slower on M3 Max 96GB

On machines with ample unified memory, the LTX2Transformer (22B parameters) fits comfortably in bf16 (~44 GB weights). Quantization adds **dequantization overhead** at every forward pass without alleviating a memory bandwidth bottleneck:

1. **bf16**: Direct matrix multiplication — no conversion needed
2. **qint8/int4**: Each `QuantizedLinear` must dequantize weights before matmul

The primary benefit of quantization is **memory reduction**, not speed. On a 64 GB Mac, bf16 would cause memory pressure and swapping, making qint8/int4 faster in practice due to avoided swap overhead.

### Memory breakdown (bf16)

| Component | Estimated Size |
|-----------|---------------|
| LTX2Transformer (22B params) | ~44 GB |
| VAE decoder | ~0.5 GB |
| Text encoder (Gemma 4-bit) | ~7.5 GB |
| Audio VAE + Vocoder | ~0.2 GB |
| Activations + KV cache | ~2-5 GB |
| **Total** | **~54-57 GB** |

With 3-phase memory management (unload Gemma before denoising, unload transformer before VAE), only the active phase's models are in memory.

---

## Per-Step Timing Detail

### Stage 1 — Half Resolution (512x288, 31 latent frames)

| Step | bf16 | qint8 | int4 |
|------|------|-------|------|
| 1 (σ=1.000→0.994) | 27.1s | 46.1s | 48.6s |
| 2 (σ=0.994→0.988) | 29.9s | 40.2s | 46.3s |
| 3 (σ=0.988→0.981) | 29.7s | 41.3s | 44.5s |
| 4 (σ=0.981→0.975) | 27.7s | 42.1s | 45.3s |
| 5 (σ=0.975→0.909) | 32.5s | 46.5s | 47.0s |
| 6 (σ=0.909→0.725) | 34.2s | 45.2s | 50.9s |
| 7 (σ=0.725→0.422) | 32.8s | 74.9s | 58.2s |
| 8 (σ=0.422→0.000) | 34.1s | 72.5s | 67.4s |
| **Total** | **248s** | **409s** | **408s** |

### Stage 2 — Full Resolution (1024x576, 31 latent frames)

| Step | bf16 | qint8 | int4 |
|------|------|-------|------|
| 1 (σ=0.909→0.725) | 329.0s | 266.3s | 255.4s |
| 2 (σ=0.725→0.422) | 185.2s | 344.7s | 240.8s |
| 3 (σ=0.422→0.000) | 250.0s | 306.1s | 280.6s |
| **Total** | **766s** | **918s** | **778s** |

Note: Step timing variability is expected due to MLX lazy evaluation, memory pressure, and thermal throttling during long runs.
