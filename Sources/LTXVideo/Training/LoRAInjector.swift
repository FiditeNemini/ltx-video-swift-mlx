// LoRAInjector.swift - Inject LoRA layers into a frozen transformer
// Copyright 2026

import Foundation
@preconcurrency import MLX
import MLXNN

/// Injects trainable LoRA layers into a frozen transformer model.
///
/// Uses the same pattern as MLX's `quantize()`: walks `leafModules().flattened()`,
/// creates replacement modules, and applies them via `model.update(modules:)`.
///
/// After injection, only `loraA` and `loraB` parameters are trainable.
///
/// Target layers (matching ostris `LTX2VideoTransformer3DModel`):
/// - `attn1.to_q/k/v/to_out` (self-attention video)
/// - `attn2.to_q/k/v/to_out` (cross-attention)
/// - `ff.project_in.proj`, `ff.project_out` (FFN)
/// - Audio: `audio_attn*`, `audio_ff*`, `audio_to_video_attn`, `video_to_audio_attn`
struct LoRAInjector {

    /// Result of LoRA injection
    struct InjectionResult {
        /// Number of Linear layers replaced with LoRALinear
        let injectedCount: Int
        /// Paths of injected layers
        let injectedPaths: [String]
    }

    /// Inject LoRA layers into a transformer model.
    ///
    /// Process:
    /// 1. Freeze all model parameters
    /// 2. Find all matching Linear leaf modules
    /// 3. Replace them with LoRALinear wrappers
    /// 4. Only loraA/loraB are trainable after injection
    ///
    /// - Parameters:
    ///   - model: The transformer model to inject into
    ///   - rank: LoRA rank (default: 16)
    ///   - alpha: LoRA alpha (default: same as rank)
    ///   - includeAudio: Whether to target audio layers too
    ///   - includeFFN: Whether to target FFN layers
    /// - Returns: Injection result with count and paths
    @discardableResult
    static func inject(
        into model: Module,
        rank: Int = 16,
        alpha: Float? = nil,
        includeAudio: Bool = false,
        includeFFN: Bool = true
    ) -> InjectionResult {
        // Step 1: Freeze the entire model
        model.freeze()

        // Step 2: Find and replace Linear modules (same pattern as MLX quantize())
        let updates = model.leafModules().flattened().compactMap { (path, module) -> (String, Module)? in
            guard let linear = module as? Linear else { return nil }
            guard shouldTarget(path: path, includeAudio: includeAudio, includeFFN: includeFFN) else {
                return nil
            }
            let loraLinear = LoRALinear(base: linear, rank: rank, alpha: alpha)
            return (path, loraLinear)
        }

        // Step 3: Apply replacements
        if !updates.isEmpty {
            model.update(modules: ModuleChildren.unflattened(updates))
        }

        let injectedPaths = updates.map { $0.0 }

        // Step 4: Verify — only LoRA parameters should be trainable
        let trainableCount = model.trainableParameters().flattenedValues().count
        LTXDebug.log("LoRA injected: \(injectedPaths.count) layers, \(trainableCount) trainable parameter tensors")

        return InjectionResult(
            injectedCount: injectedPaths.count,
            injectedPaths: injectedPaths
        )
    }

    /// Extract LoRA weights from an injected model for saving.
    ///
    /// Returns a dictionary mapping layer paths to (loraA, loraB) pairs.
    static func extractLoRAWeights(from model: Module) -> [(path: String, down: MLXArray, up: MLXArray)] {
        var results: [(path: String, down: MLXArray, up: MLXArray)] = []

        for (path, module) in model.leafModules().flattened() {
            if let loraLinear = module as? LoRALinear {
                results.append((path: path, down: loraLinear.loraA, up: loraLinear.loraB))
            }
        }

        return results
    }

    // MARK: - Private

    /// Extract the last dot-separated component from a module path
    private static func lastDotComponent(_ path: String) -> String {
        if let lastDot = path.lastIndex(of: ".") {
            return String(path[path.index(after: lastDot)...])
        }
        return path
    }

    /// Check if a module path should be targeted for LoRA injection
    private static func shouldTarget(path: String, includeAudio: Bool, includeFFN: Bool) -> Bool {
        // Must be inside a transformer block
        guard path.contains("transformer_blocks") else { return false }

        let last = lastDotComponent(path)

        // Attention projections (video)
        if path.contains("attn1.") || path.contains("attn2.") {
            if ["to_q", "to_k", "to_v", "to_out"].contains(last) {
                return true
            }
        }

        // FFN projections (video)
        if includeFFN {
            if path.contains(".ff.project_in.") && last == "proj" {
                return true
            }
            if last == "project_out" && path.contains(".ff.") {
                return true
            }
        }

        // Audio targets
        if includeAudio {
            // Audio attention
            if path.contains("audio_attn1.") || path.contains("audio_attn2.") {
                if ["to_q", "to_k", "to_v", "to_out"].contains(last) {
                    return true
                }
            }
            // Audio FFN
            if includeFFN {
                if path.contains(".audio_ff.project_in.") && last == "proj" {
                    return true
                }
                if last == "project_out" && path.contains(".audio_ff.") {
                    return true
                }
            }
            // Cross-modal attention (audio ↔ video)
            if path.contains("audio_to_video_attn.") || path.contains("video_to_audio_attn.") {
                if ["to_q", "to_k", "to_v", "to_out"].contains(last) {
                    return true
                }
            }
        }

        return false
    }
}
