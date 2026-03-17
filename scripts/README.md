# Scripts

## benchmark.sh — Performance Benchmark Suite

Runs all inference pipelines (T2V, I2V, retake, audio) with fixed seeds and generates a shareable `RESULTS.md`.

### Quick start

```bash
# 1. Build in Release mode
xcodebuild -scheme ltx-video -configuration Release \
    -derivedDataPath .xcodebuild -destination 'platform=macOS' build

# 2. Download models (first time only, ~30GB)
.xcodebuild/Build/Products/Release/ltx-video download

# 3. Run benchmarks
./scripts/benchmark.sh --quick              # ~10 min (9-frame tests only)
./scripts/benchmark.sh                      # ~1 hour (includes 10-second videos)
./scripts/benchmark.sh --quick --skip-audio # ~6 min (no audio model loading)
```

### Output

Results are saved to `benchmarks/<timestamp>/`:

```
benchmarks/20260317_143022/
  RESULTS.md              <- Share this! (paste into GitHub issue)
  benchmark.log           <- Full console log
  t2v-768x512-9f.txt      <- Profiling details per test
  t2v-768x512-9f.mp4      <- Generated video
  i2v-768x512-9f.txt
  i2v-768x512-9f.mp4
  ...
```

### Share your results

Copy `RESULTS.md` to clipboard and create a benchmark issue:

```bash
cat benchmarks/*/RESULTS.md | pbcopy
# Then: https://github.com/VincentGourbin/ltx-video-swift-mlx/issues/new?template=benchmark.yml
```

This helps us track performance across different Apple Silicon chips and memory configurations.

### Options

| Flag | Description |
|------|-------------|
| `--quick` | Only run 9-frame tests (skip 241-frame / 10-second videos) |
| `--skip-audio` | Skip audio pipeline test (requires separate audio model download) |
| `--help` | Show usage |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BINARY` | `.xcodebuild/Build/Products/Release/ltx-video` | Path to the CLI binary |
| `SEED` | `42` | Random seed for reproducibility |
| `COOLDOWN` | `120` | Seconds between tests (thermal throttling mitigation) |

### Reference times (M3 Max 96GB)

| Test | Time |
|------|------|
| T2V 768x512 9f | ~33s |
| T2V 1024x576 241f | ~895s (~15 min) |
| I2V 768x512 9f | ~39s |
| I2V 1024x576 241f | ~755s (~12.5 min) |
| Retake 768x512 9f | ~22s |
| Audio I2V 768x512 9f | ~58s |
