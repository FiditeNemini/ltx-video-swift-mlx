#!/bin/bash
# benchmark.sh — Full inference benchmark suite for LTX-2.3 Swift/MLX
#
# Runs all supported pipelines and reports timing + memory.
# Results are saved to a timestamped directory with a shareable RESULTS.md.
#
# Usage:
#   ./scripts/benchmark.sh                    # Run all tests
#   ./scripts/benchmark.sh --quick            # Quick tests only (9 frames)
#   ./scripts/benchmark.sh --skip-audio       # Skip audio tests (require audio models)
#   ./scripts/benchmark.sh --quick --skip-audio
#
# Prerequisites:
#   1. Build in Release mode:
#      xcodebuild -scheme ltx-video -configuration Release \
#          -derivedDataPath .xcodebuild -destination 'platform=macOS' build
#
#   2. Download models (first run only):
#      .xcodebuild/Build/Products/Release/ltx-video download
#
# Output:
#   benchmarks/<timestamp>/RESULTS.md    <- Share this file (or paste into a GitHub issue)
#   benchmarks/<timestamp>/*.txt         <- Full profiling logs per test
#   benchmarks/<timestamp>/*.mp4         <- Generated videos
#
# Hardware: Apple Silicon Mac with >=36GB unified memory (96GB recommended for 10s videos)

set -euo pipefail

# --- Configuration ---
BINARY="${BINARY:-.xcodebuild/Build/Products/Release/ltx-video}"
SEED=42
COOLDOWN=120  # seconds between tests to avoid thermal throttling
IMAGE="docs/examples/image-to-video/input_768x512.png"
QUICK_ONLY=false
SKIP_AUDIO=false

for arg in "$@"; do
    case $arg in
        --quick) QUICK_ONLY=true ;;
        --skip-audio) SKIP_AUDIO=true ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg (use --help)"; exit 1 ;;
    esac
done

# --- Output directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="benchmarks/${TIMESTAMP}"
mkdir -p "$OUTDIR"

# --- System info ---
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')
RAM=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
MACOS=$(sw_vers -productVersion)
VERSION=$("$BINARY" --version 2>&1 || echo "unknown")

# --- Helpers ---
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/benchmark.log"; }
cooldown() { log "Cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

# Collect results for RESULTS.md
declare -a RESULT_NAMES=()
declare -a RESULT_TIMES=()
declare -a RESULT_MEMS=()
declare -a RESULT_DENOISE=()
declare -a RESULT_VAE=()
declare -a RESULT_TEXT=()

run_bench() {
    local name="$1"
    local output="$OUTDIR/${name}.mp4"
    shift
    log "=== $name ==="
    log "Command: $BINARY $*"

    local start_time=$(date +%s)
    "$BINARY" "$@" -o "$output" --profile --seed "$SEED" 2>&1 | tee "$OUTDIR/${name}.txt" || true
    local end_time=$(date +%s)
    local wall_time=$((end_time - start_time))

    # Extract key metrics
    local gen_time=$(grep -E "(Generation|Retake) completed in" "$OUTDIR/${name}.txt" | grep -oE "[0-9]+\.[0-9]+s" | head -1)
    local peak_mem=$(grep "Peak GPU memory" "$OUTDIR/${name}.txt" | grep -oE "[0-9]+ MB" | head -1)
    local denoise=$(grep "Denoising" "$OUTDIR/${name}.txt" | grep -oE "[0-9]+\.[0-9]+s" | head -1)
    local vae=$(grep "VAE Decoding" "$OUTDIR/${name}.txt" | grep -oE "[0-9]+\.[0-9]+s" | head -1)
    local text=$(grep "Text Encoding" "$OUTDIR/${name}.txt" | grep -oE "[0-9]+\.[0-9]+s" | head -1)

    log "Result: ${gen_time:-N/A} (wall: ${wall_time}s), peak memory: ${peak_mem:-N/A}"
    log ""

    # Store for summary
    RESULT_NAMES+=("$name")
    RESULT_TIMES+=("${gen_time:-N/A}")
    RESULT_MEMS+=("${peak_mem:-N/A}")
    RESULT_DENOISE+=("${denoise:-N/A}")
    RESULT_VAE+=("${vae:-N/A}")
    RESULT_TEXT+=("${text:-N/A}")
}

# --- Sanity check ---
if [ ! -x "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Build first:"
    echo "  xcodebuild -scheme ltx-video -configuration Release -derivedDataPath .xcodebuild -destination 'platform=macOS' build"
    exit 1
fi

log "LTX-2.3 Benchmark Suite"
log "Hardware: $CHIP, ${RAM} GB, macOS $MACOS"
log "Mode: $(if $QUICK_ONLY; then echo 'quick (9 frames only)'; else echo 'full'; fi)"
log "Output: $OUTDIR/"
log ""

# ============================================================
# 1. TEXT-TO-VIDEO
# ============================================================

run_bench "t2v-768x512-9f" generate \
    "A beaver building a dam in a peaceful forest stream, golden hour lighting" \
    -w 768 -h 512 -f 9

if ! $QUICK_ONLY; then
    cooldown
    run_bench "t2v-1024x576-241f" generate \
        "A beaver building a dam in a peaceful forest stream, golden hour lighting" \
        -w 1024 -h 576 -f 241
fi

# ============================================================
# 2. IMAGE-TO-VIDEO
# ============================================================

if [ -f "$IMAGE" ]; then
    cooldown
    run_bench "i2v-768x512-9f" generate \
        "The scene comes alive with gentle motion, camera slowly panning right, leaves swaying" \
        --image "$IMAGE" \
        -w 768 -h 512 -f 9

    if ! $QUICK_ONLY; then
        cooldown
        local_image="$OUTDIR/input_1024x576.png"
        ffmpeg -y -i "$IMAGE" -vf "scale=1024:576" "$local_image" 2>/dev/null
        run_bench "i2v-1024x576-241f" generate \
            "The scene comes alive with gentle motion, camera slowly panning right, leaves swaying in the breeze, cinematic" \
            --image "$local_image" \
            -w 1024 -h 576 -f 241
    fi
else
    log "SKIP I2V: input image not found at $IMAGE"
fi

# ============================================================
# 3. RETAKE
# ============================================================

T2V_SOURCE="$OUTDIR/t2v-768x512-9f.mp4"
if [ -f "$T2V_SOURCE" ]; then
    cooldown
    run_bench "retake-768x512-9f" retake \
        "A fluffy orange cat in a peaceful forest stream, golden hour lighting" \
        --video "$T2V_SOURCE" \
        --strength 0.8 \
        -w 768 -h 512 -f 9
else
    log "SKIP Retake: source video not available"
fi

# ============================================================
# 4. AUDIO
# ============================================================

if ! $SKIP_AUDIO && [ -f "$IMAGE" ]; then
    cooldown
    run_bench "audio-i2v-768x512-9f" generate \
        "A chic woman walks towards a red vintage car, opens the door, gets inside. She starts the engine which roars." \
        --image "$IMAGE" --audio \
        -w 768 -h 512 -f 9
else
    if $SKIP_AUDIO; then
        log "SKIP Audio: --skip-audio flag"
    else
        log "SKIP Audio: input image not found"
    fi
fi

# ============================================================
# Generate RESULTS.md (shareable)
# ============================================================

RESULTS="$OUTDIR/RESULTS.md"

cat > "$RESULTS" <<HEADER
# LTX-2.3 Benchmark Results

## System

| | |
|---|---|
| **Chip** | $CHIP |
| **Memory** | ${RAM} GB unified |
| **macOS** | $MACOS |
| **Version** | $VERSION |
| **Date** | $(date +%Y-%m-%d) |
| **Mode** | $(if $QUICK_ONLY; then echo 'quick (9 frames)'; else echo 'full'; fi) |
| **Seed** | $SEED |

## Results

| Test | Total | Denoising | VAE Decode | Text Encode | Peak Memory |
|------|-------|-----------|------------|-------------|-------------|
HEADER

for i in "${!RESULT_NAMES[@]}"; do
    printf "| %s | %s | %s | %s | %s | %s |\n" \
        "${RESULT_NAMES[$i]}" \
        "${RESULT_TIMES[$i]}" \
        "${RESULT_DENOISE[$i]}" \
        "${RESULT_VAE[$i]}" \
        "${RESULT_TEXT[$i]}" \
        "${RESULT_MEMS[$i]}" >> "$RESULTS"
done

cat >> "$RESULTS" <<'FOOTER'

## How to reproduce

```bash
git clone https://github.com/VincentGourbin/ltx-video-swift-mlx.git
cd ltx-video-swift-mlx
xcodebuild -scheme ltx-video -configuration Release \
    -derivedDataPath .xcodebuild -destination 'platform=macOS' build
./scripts/benchmark.sh --quick    # or without --quick for full suite
```

## Share your results

Paste this file into a [benchmark issue](https://github.com/VincentGourbin/ltx-video-swift-mlx/issues/new?template=benchmark.yml) to help us track performance across hardware.
FOOTER

# Print summary
log ""
log "========================================"
log "RESULTS"
log "========================================"
cat "$RESULTS" | tee -a "$OUTDIR/benchmark.log"
log ""
log "========================================"
log "Share: $RESULTS"
log "  or: cat $RESULTS | pbcopy"
log "Done."
