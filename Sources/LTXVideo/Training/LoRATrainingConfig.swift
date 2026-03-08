// LoRATrainingConfig.swift - Training Configuration
// Copyright 2026

import Foundation

/// Configuration for LoRA training
public struct LoRATrainingConfig: Sendable {
    /// LoRA rank (low-rank dimension)
    public var rank: Int

    /// LoRA alpha (scaling factor). Default: same as rank
    public var alpha: Float

    /// Learning rate
    public var learningRate: Float

    /// AdamW weight decay
    public var weightDecay: Float

    /// Maximum training steps
    public var maxSteps: Int

    /// Save checkpoint every N steps
    public var saveEvery: Int

    /// Video width (must be divisible by 32)
    public var width: Int

    /// Video height (must be divisible by 32)
    public var height: Int

    /// Number of frames (must be 8n+1)
    public var numFrames: Int

    /// Model variant to train on
    public var model: String

    /// Whether to include audio LoRA targets
    public var includeAudio: Bool

    /// Audio loss weight relative to video loss
    public var audioLossWeight: Float

    /// Whether to include FFN layers in LoRA targets
    public var includeFFN: Bool

    /// Transformer quantization for QLoRA (bf16, qint8, int4)
    public var transformerQuant: String

    /// Gradient accumulation steps
    public var gradientAccumulationSteps: Int

    /// LR warmup steps
    public var warmupSteps: Int

    /// Seed for reproducibility
    public var seed: UInt64?

    /// HuggingFace token
    public var hfToken: String?

    /// Custom models directory
    public var modelsDir: String?

    /// Gemma model path
    public var gemmaPath: String?

    /// LTX weights path
    public var ltxWeightsPath: String?

    // MARK: - Memory Presets

    /// Preset for 32GB systems
    public static let compact = LoRATrainingConfig(
        rank: 8, alpha: 8, learningRate: 1e-4,
        width: 256, height: 256, numFrames: 9,
        transformerQuant: "int4"
    )

    /// Preset for 64GB systems
    public static let balanced = LoRATrainingConfig(
        rank: 16, alpha: 16, learningRate: 1e-4,
        width: 384, height: 384, numFrames: 17,
        transformerQuant: "qint8"
    )

    /// Preset for 96GB systems
    public static let quality = LoRATrainingConfig(
        rank: 32, alpha: 32, learningRate: 1e-4,
        width: 512, height: 512, numFrames: 25,
        transformerQuant: "bf16"
    )

    /// Preset for 192GB systems
    public static let max = LoRATrainingConfig(
        rank: 64, alpha: 64, learningRate: 1e-4,
        width: 768, height: 512, numFrames: 41,
        transformerQuant: "bf16"
    )

    public init(
        rank: Int = 16,
        alpha: Float? = nil,
        learningRate: Float = 1e-4,
        weightDecay: Float = 0.01,
        maxSteps: Int = 2000,
        saveEvery: Int = 250,
        width: Int = 256,
        height: Int = 256,
        numFrames: Int = 9,
        model: String = "dev",
        includeAudio: Bool = false,
        audioLossWeight: Float = 0.5,
        includeFFN: Bool = true,
        transformerQuant: String = "bf16",
        gradientAccumulationSteps: Int = 1,
        warmupSteps: Int = 100,
        seed: UInt64? = nil,
        hfToken: String? = nil,
        modelsDir: String? = nil,
        gemmaPath: String? = nil,
        ltxWeightsPath: String? = nil
    ) {
        self.rank = rank
        self.alpha = alpha ?? Float(rank)
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.maxSteps = maxSteps
        self.saveEvery = saveEvery
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.model = model
        self.includeAudio = includeAudio
        self.audioLossWeight = audioLossWeight
        self.includeFFN = includeFFN
        self.transformerQuant = transformerQuant
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.warmupSteps = warmupSteps
        self.seed = seed
        self.hfToken = hfToken
        self.modelsDir = modelsDir
        self.gemmaPath = gemmaPath
        self.ltxWeightsPath = ltxWeightsPath
    }

    /// Validate configuration
    public func validate() throws {
        guard (numFrames - 1) % 8 == 0 else {
            throw TrainingError.invalidConfig("Frame count must be 8n+1, got \(numFrames)")
        }
        guard width % 32 == 0 && height % 32 == 0 else {
            throw TrainingError.invalidConfig("Width and height must be divisible by 32, got \(width)x\(height)")
        }
        guard rank > 0 && rank <= 512 else {
            throw TrainingError.invalidConfig("Rank must be between 1 and 512, got \(rank)")
        }
        guard learningRate > 0 && learningRate < 1 else {
            throw TrainingError.invalidConfig("Learning rate must be between 0 and 1, got \(learningRate)")
        }
    }
}

/// Training-specific errors
public enum TrainingError: Error, LocalizedError {
    case invalidConfig(String)
    case datasetError(String)
    case modelError(String)
    case checkpointError(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let msg): return "Invalid training config: \(msg)"
        case .datasetError(let msg): return "Dataset error: \(msg)"
        case .modelError(let msg): return "Model error: \(msg)"
        case .checkpointError(let msg): return "Checkpoint error: \(msg)"
        }
    }
}
