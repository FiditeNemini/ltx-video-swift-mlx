// LoRAUnfuseE2ETests.swift — Gated check that unfusing actually restores weights
// Copyright 2026
//
// Motivated by a live observation: a run whose stage 2 unfused the adapter
// produced output bit-identical to a run that kept it fused. Either the unfuse
// restores nothing, or something upstream never calls it. This test isolates the
// mechanics: fuse a real adapter into a randomly-initialised transformer (key
// paths are all that matter — no 46 GB load), then unfuse, then compare weights
// elementwise against their pre-fuse values.
//
// Gated on the adapter file being cached locally.

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

/// Any cached LoRA with transformer_blocks targets will do.
private let cachedAdapterPath: String? = [
    NSHomeDirectory() + "/Library/Caches/ltx-video-mlx/ltx23-lora-pixel-upscaler-x2/ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors",
    "/Users/vincent/Pictures/FluxforgeStudio/LoRAs/ltx-2.3-22b-lora-camera-arcshot.safetensors",
].first { FileManager.default.fileExists(atPath: $0) }

@Suite("LoRA unfuse restores weights (gated on cached adapter)",
       .enabled(if: cachedAdapterPath != nil),
       .serialized)
struct LoRAUnfuseE2ETests {

    @Test func fuseThenUnfuseIsAnExactRoundTrip() throws {
        let path = cachedAdapterPath!
        // Two blocks are enough: the adapter's block-0/1 layers fuse, and the
        // mechanics under test (capture → restore) are identical per layer. A full
        // 48-layer model at random init would mean snapshotting ~44 GB for nothing.
        var tiny = LTXTransformerConfig.ltx23
        tiny.numLayers = 2
        let model = LTXTransformer(config: tiny)
        eval(model.parameters())

        // Snapshot every parameter before fusing. Copied via `+ 0` so a hypothetical
        // in-place mutation of the originals cannot fool the comparison.
        let before = Dictionary(uniqueKeysWithValues:
            model.parameters().flattened().map { ($0.0, $0.1 + 0) })
        eval(Array(before.values))

        let (originals, result) = try (model as MLXNN.Module).fuseLoRA(from: path, scale: 1.0)
        #expect(result.modifiedLayerCount > 0, "the adapter must actually fuse")

        // Fusion must have changed at least one weight, or the test proves nothing.
        // Summed into one array so a single GPU sync answers for all ~1400 parameters.
        let afterFuse = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let fuseDelta = before.map { MLX.abs(afterFuse[$0.key]! - $0.value).max() }
        let fuseChanged = MLX.stacked(fuseDelta).max().item(Float.self)
        #expect(fuseChanged > 0, "fusion changed no weights — nothing to unfuse")

        // The regression that motivated this test: Module.update mutates parameter
        // arrays IN PLACE, so originals captured as bare references silently became
        // the fused values and unfuse restored exactly what it was meant to undo.
        // The capture must therefore be a copy — pinned here explicitly, because a
        // reference capture makes the round-trip below pass trivially against a
        // contaminated baseline in any test that snapshots without copying.
        let probeKey = "transformer_blocks.0.attn1.to_q.weight"
        if let captured = originals[probeKey], let pristine = before[probeKey] {
            let capturedDelta = MLX.abs(captured - pristine).max().item(Float.self)
            #expect(capturedDelta == 0,
                    "captured originals are contaminated (Δmax=\(capturedDelta)) — reference capture, not a copy")
        }
        _ = afterFuse   // aliases the live (mutated-in-place) arrays; unusable as a baseline

        (model as MLXNN.Module).unfuseLoRA(originalWeights: originals)

        // Every parameter must be back, elementwise — one sync for the global answer,
        // then a per-key pass only if it fails, to name the culprits.
        let after = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let keys = Array(before.keys)
        let residuals = keys.map { MLX.abs(after[$0]! - before[$0]!).max() }
        let stacked = MLX.stacked(residuals)
        MLX.eval(stacked)
        let worst = stacked.max().item(Float.self)
        if worst > 0 {
            let values = stacked.asArray(Float.self)
            let culprits = zip(keys, values).filter { $0.1 > 0 }
                .sorted { $0.1 > $1.1 }.prefix(5)
                .map { "\($0.0) (Δmax=\($0.1))" }
            let count = values.filter { $0 > 0 }.count
            #expect(Bool(false), "unfuse left \(count) weights fused: \(culprits)")
        }
    }
}
