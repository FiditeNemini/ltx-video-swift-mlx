//
//  LipDubLoRAScaleTests.swift
//  ltx-video-swift-mlx
//
//  Guards the `lipdubLoRAScale` contract: the scale multiplies the fused
//  delta (W' = W + scale · B·A), and a non-positive scale is refused because
//  it would run base weights that never saw the appended reference tokens.
//
//  The scale-vs-reuse guard (same file, different scale → throw) needs a
//  loaded 22B transformer and lives in the gated LipDubReuseE2ETests.
//

import Testing
import Foundation
@preconcurrency import MLX
@testable import LTXVideo

@Suite("LipDub LoRA scale")
struct LipDubLoRAScaleTests {

    /// A synthetic LoRA file with one A/B pair, so the delta is exactly
    /// `up @ down` before scaling.
    private func writeSyntheticLoRA() throws -> String {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lipdub-scale-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("lora.safetensors")

        // rank 2, 4×4 layer: down (2,4), up (4,2) → delta (4,4)
        let down = MLXArray((0 ..< 8).map { Float($0) * 0.1 }, [2, 4])
        let up = MLXArray((0 ..< 8).map { Float($0) * 0.2 }, [4, 2])
        let base = "diffusion_model.transformer_blocks.0.attn1.to_q"
        try MLX.save(arrays: [
            base + ".lora_A.weight": down,
            base + ".lora_B.weight": up,
        ], url: url)
        return url.path
    }

    /// The delta must scale linearly: at 0.5 it is exactly half of the 1.0
    /// delta. This is the property the new parameter relies on.
    @Test func deltaScalesLinearly() throws {
        let path = try writeSyntheticLoRA()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let key = "diffusion_model.transformer_blocks.0.attn1.to_q"

        let full = try LoRALoader.load(from: path, config: LoRAConfig(weightsPath: path, scale: 1.0))
        let half = try LoRALoader.load(from: path, config: LoRAConfig(weightsPath: path, scale: 0.5))

        guard let dFull = full.getDelta(for: key), let dHalf = half.getDelta(for: key) else {
            Issue.record("no delta for \(key)")
            return
        }
        let diff = MLX.abs(dHalf - dFull * MLXArray(Float(0.5))).max().item(Float.self)
        #expect(diff < 1e-6, "scale must multiply the delta linearly (max diff \(diff))")

        // And 1.0 is not a no-op — the delta is non-zero, so the test above
        // cannot pass trivially.
        #expect(MLX.abs(dFull).max().item(Float.self) > 0)
    }

    /// No `.alpha` in the file → effectiveScale is 1.0, so the caller's scale
    /// is the whole story. (The shipped LipDub IC-LoRA has no alpha keys.)
    @Test func absentAlphaMeansUnitEffectiveScale() throws {
        let path = try writeSyntheticLoRA()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let weights = try LoRALoader.load(from: path, config: LoRAConfig(weightsPath: path, scale: 1.0))
        let layer = try #require(weights.layers.first)
        #expect(layer.alpha == nil)
        #expect(layer.effectiveScale == 1.0)
    }

    /// A non-positive scale is refused. Exercised through the validator itself:
    /// inside `generateLipDub` this check necessarily sits after the
    /// models-loaded guard, so calling it on an unloaded pipeline would throw
    /// `modelNotLoaded` and pass without touching this rule at all.
    @Test func nonPositiveScaleThrows() throws {
        for bad in [Float(0), -0.5, -1] {
            #expect(throws: LTXError.self) {
                _ = try LTXPipeline.validateLipDubLoRAScale(bad)
            }
        }
        // And the thrown error is the configuration one, with the actionable text.
        do {
            _ = try LTXPipeline.validateLipDubLoRAScale(0)
            Issue.record("scale 0 did not throw")
        } catch let error as LTXError {
            guard case .invalidConfiguration(let message) = error else {
                Issue.record("wrong LTXError case: \(error)")
                return
            }
            #expect(message.contains("lipdubLoRAScale must be > 0"))
        }
    }

    /// In-range scales are accepted silently; out-of-range ones are accepted but
    /// warn, because this LoRA carries the conditioning rather than a style.
    @Test func warnsOnlyOutsideTheSaneRange() throws {
        for ok in [Float(0.5), 0.8, 1.0, 1.2, 1.5] {
            #expect(try LTXPipeline.validateLipDubLoRAScale(ok) == nil, "\(ok) should not warn")
        }
        for far in [Float(0.1), 0.49, 1.51, 3.0] {
            let warning = try LTXPipeline.validateLipDubLoRAScale(far)
            #expect(warning != nil, "\(far) should warn")
            #expect(warning?.contains("far from the published 1.0") == true)
        }
    }
}
