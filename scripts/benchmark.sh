#!/bin/bash
# benchmark.sh — Full inference benchmark suite for LTX-2.3 Swift/MLX
#
# Runs all supported pipelines and reports timing + memory.
# Results are saved to a timestamped directory.
#
# Usage:
#   ./scripts/benchmark.sh                    # Run all tests
#   ./scripts/benchmark.sh --quick            # Quick tests only (9 frames)
#   ./scripts/benchmark.sh --skip-audio       # Skip audio tests (require loadAudioModels)
#
# Prerequisites:
#   - Release build: xcodebuild -scheme ltx-video -configuration Release \
#       -derivedDataPath .xcodebuild -destination 'platform=macOS' build
#   - Models downloaded: ltx-video download
#
# Hardware: Apple Silicon Mac with ≥36GB unified memory (96GB recommended for 10s videos)

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
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Output directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="benchmarks/${TIMESTAMP}"
mkdir -p "$OUTDIR"

# --- Helpers ---
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/benchmark.log"; }
cooldown() { log "Cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

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

    # Extract key metrics from output (match both "Generation completed" and "Retake completed")
    local gen_time=$(grep -E "(Generation|Retake) completed in" "$OUTDIR/${name}.txt" | grep -oE "[0-9]+\.[0-9]+s" | head -1)
    local peak_mem=$(grep "Peak GPU memory" "$OUTDIR/${name}.txt" | grep -oE "[0-9]+ MB" | head -1)
    log "Result: ${gen_time:-N/A} (wall: ${wall_time}s), peak memory: ${peak_mem:-N/A}"
    log ""
}

# --- Sanity check ---
if [ ! -x "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Build first: xcodebuild -scheme ltx-video -configuration Release -derivedDataPath .xcodebuild -destination 'platform=macOS' build"
    exit 1
fi

log "LTX-2.3 Benchmark Suite"
log "Binary: $BINARY"
log "Seed: $SEED"
log "Output: $OUTDIR/"
log "Mode: $(if $QUICK_ONLY; then echo 'quick (9 frames only)'; else echo 'full'; fi)"
log "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
log "Memory: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"
log "macOS: $(sw_vers -productVersion)"
log ""

# ============================================================
# 1. TEXT-TO-VIDEO
# ============================================================

# T2V Quick: 768x512, 9 frames
run_bench "t2v-768x512-9f" generate \
    "A beaver building a dam in a peaceful forest stream, golden hour lighting" \
    -w 768 -h 512 -f 9

if ! $QUICK_ONLY; then
    cooldown
    # T2V Full: 1024x576, 241 frames (10s)
    run_bench "t2v-1024x576-241f" generate \
        "A beaver building a dam in a peaceful forest stream, golden hour lighting" \
        -w 1024 -h 576 -f 241
fi

# ============================================================
# 2. IMAGE-TO-VIDEO
# ============================================================

if [ -f "$IMAGE" ]; then
    cooldown
    # I2V Quick: 768x512, 9 frames
    run_bench "i2v-768x512-9f" generate \
        "The scene comes alive with gentle motion, camera slowly panning right, leaves swaying" \
        --image "$IMAGE" \
        -w 768 -h 512 -f 9

    if ! $QUICK_ONLY; then
        cooldown
        # I2V Full: 1024x576, 241 frames (10s)
        # Resize input image to match target resolution
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
# 3. RETAKE (requires a source video)
# ============================================================

# Use the T2V output as retake source
T2V_SOURCE="$OUTDIR/t2v-768x512-9f.mp4"
if [ -f "$T2V_SOURCE" ]; then
    cooldown
    # Retake: full, 768x512, 9 frames
    run_bench "retake-full-768x512-9f" retake \
        "A fluffy orange cat in a peaceful forest stream, golden hour lighting" \
        --video "$T2V_SOURCE" \
        --strength 0.8 \
        -w 768 -h 512 -f 9
else
    log "SKIP Retake: source video not available"
fi

# ============================================================
# 4. AUDIO (I2V + audio, requires audio models)
# ============================================================

if ! $SKIP_AUDIO && [ -f "$IMAGE" ]; then
    cooldown
    # Audio: I2V + audio, 768x512, 9 frames
    run_bench "audio-i2v-768x512-9f" generate \
        "A chic woman walks towards a red vintage car, opens the door, gets inside. She starts the engine which roars." \
        --image "$IMAGE" --audio \
        -w 768 -h 512 -f 9
else
    if $SKIP_AUDIO; then
        log "SKIP Audio: --skip-audio flag set"
    else
        log "SKIP Audio: input image not found at $IMAGE"
    fi
fi

# ============================================================
# Summary
# ============================================================

log "========================================"
log "BENCHMARK SUMMARY"
log "========================================"
for f in "$OUTDIR"/*.txt; do
    name=$(basename "$f" .txt)
    gen_time=$(grep -E "(Generation|Retake) completed in" "$f" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+s" | head -1)
    peak_mem=$(grep "Peak GPU memory" "$f" 2>/dev/null | grep -oE "[0-9]+ MB" | head -1)
    if [ -n "$gen_time" ]; then
        printf "  %-30s %10s  peak: %s\n" "$name" "$gen_time" "${peak_mem:-N/A}" | tee -a "$OUTDIR/benchmark.log"
    fi
done
log "========================================"
log "Full logs: $OUTDIR/"
log "Done."
