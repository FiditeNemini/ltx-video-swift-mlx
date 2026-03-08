// LoRALinear.swift - Trainable LoRA layer wrapping a frozen Linear
// Copyright 2026

import Foundation
@preconcurrency import MLX
import MLXNN

/// A trainable LoRA layer that wraps a frozen Linear module.
///
/// Forward: `base(x) + scale * (x @ A.T) @ B.T`
///
/// Where:
/// - `base` is the frozen original Linear (weights are not trainable)
/// - `A` (loraA) has shape `(rank, inFeatures)` — initialized with Kaiming uniform
/// - `B` (loraB) has shape `(outFeatures, rank)` — initialized with zeros
/// - `scale = alpha / rank`
///
/// Only `loraA` and `loraB` appear in `trainableParameters()`.
class LoRALinear: Module {
    /// The frozen base linear layer
    let base: Linear

    /// LoRA down projection: (rank, inFeatures)
    @ParameterInfo(key: "lora_down") var loraA: MLXArray

    /// LoRA up projection: (outFeatures, rank)
    @ParameterInfo(key: "lora_up") var loraB: MLXArray

    /// Scaling factor: alpha / rank
    let scale: Float

    /// LoRA rank
    let rank: Int

    /// Alpha value
    let alpha: Float

    /// Input features of the base linear
    let inFeatures: Int

    /// Output features of the base linear
    let outFeatures: Int

    init(base: Linear, rank: Int = 16, alpha: Float? = nil) {
        self.base = base
        self.rank = rank
        self.alpha = alpha ?? Float(rank)
        self.scale = self.alpha / Float(rank)

        // Infer dimensions from base weight
        // Linear weight shape is (outFeatures, inFeatures) in MLX
        let weight = base.weight
        self.outFeatures = weight.dim(0)
        self.inFeatures = weight.dim(1)

        // Initialize A with Kaiming uniform, B with zeros
        // Kaiming uniform: U(-bound, bound) where bound = sqrt(1/inFeatures)
        let bound = Float(1.0 / Double(self.inFeatures).squareRoot())
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -bound, high: bound,
            [rank, self.inFeatures]
        ).asType(.float32)
        self._loraB.wrappedValue = MLXArray.zeros([self.outFeatures, rank]).asType(.float32)

        super.init()

        // Freeze the base linear so its weights don't appear in trainableParameters()
        self.base.freeze()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base output (frozen)
        let baseOutput = base(x)

        // LoRA delta: scale * (x @ A.T) @ B.T
        let xFloat = x.asType(.float32)
        let loraOut = MLX.matmul(MLX.matmul(xFloat, loraA.T), loraB.T) * scale

        return baseOutput + loraOut.asType(baseOutput.dtype)
    }
}
