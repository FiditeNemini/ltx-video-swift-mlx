// TrainCommand.swift - CLI command for LoRA training
// Copyright 2026

import ArgumentParser
import Foundation
import LTXVideo

struct Train: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Train a LoRA adapter on a dataset of videos + captions"
    )

    @Argument(help: "Path to dataset directory (video.mp4 + video.txt pairs)")
    var dataset: String

    @Option(name: .shortAndLong, help: "Output directory for LoRA weights and checkpoints")
    var output: String = "./lora-output"

    @Option(name: .shortAndLong, help: "Model variant: dev or distilled")
    var model: String = "dev"

    @Option(name: .long, help: "LoRA rank (default: 16)")
    var rank: Int = 16

    @Option(name: .long, help: "LoRA alpha (default: same as rank)")
    var alpha: Float?

    @Option(name: .long, help: "Learning rate (default: 1e-4)")
    var lr: Float = 1e-4

    @Option(name: .long, help: "Weight decay (default: 0.01)")
    var weightDecay: Float = 0.01

    @Option(name: .long, help: "Maximum training steps (default: 2000)")
    var steps: Int = 2000

    @Option(name: .long, help: "Save checkpoint every N steps (default: 250)")
    var saveEvery: Int = 250

    @Option(name: .shortAndLong, help: "Video width (must be divisible by 32)")
    var width: Int = 256

    @Option(name: .shortAndLong, help: "Video height (must be divisible by 32)")
    var height: Int = 256

    @Option(name: .shortAndLong, help: "Number of frames (must be 8n+1)")
    var frames: Int = 9

    @Option(name: .long, help: "Transformer quantization: bf16 (default), qint8, int4")
    var transformerQuant: String = "bf16"

    @Option(name: .long, help: "Gradient accumulation steps (default: 1)")
    var gradAccum: Int = 1

    @Option(name: .long, help: "LR warmup steps (default: 100)")
    var warmupSteps: Int = 100

    @Option(name: .long, help: "Random seed")
    var seed: UInt64?

    @Option(name: .long, help: "HuggingFace token")
    var hfToken: String?

    @Option(name: .long, help: "Custom models directory")
    var modelsDir: String?

    @Option(name: .long, help: "Path to local Gemma3 model directory")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to unified LTX-2 weights file")
    var ltxWeights: String?

    @Flag(name: .long, help: "Train on audio+video (dual transformer)")
    var audio: Bool = false

    @Option(name: .long, help: "Audio loss weight (default: 0.5)")
    var audioLossWeight: Float = 0.5

    @Flag(name: .long, inversion: .prefixedNo, help: "Include FFN layers in LoRA targets (default: yes)")
    var includeFFN: Bool = true

    @Option(name: .long, help: "Memory preset: compact (32GB), balanced (64GB), quality (96GB), max (192GB)")
    var preset: String?

    mutating func run() async throws {
        print("LTX-2 LoRA Training")
        print("===================")

        // Apply preset if specified
        var config: LoRATrainingConfig
        if let presetName = preset {
            switch presetName {
            case "compact": config = .compact
            case "balanced": config = .balanced
            case "quality": config = .quality
            case "max": config = .max
            default:
                throw ValidationError("Unknown preset: \(presetName). Use: compact, balanced, quality, max")
            }
            print("Preset: \(presetName)")
        } else {
            config = LoRATrainingConfig()
        }

        // Override with explicit flags
        config.rank = rank
        if let a = alpha { config.alpha = a }
        config.learningRate = lr
        config.weightDecay = weightDecay
        config.maxSteps = steps
        config.saveEvery = saveEvery
        config.width = width
        config.height = height
        config.numFrames = frames
        config.model = model
        config.includeAudio = audio
        config.audioLossWeight = audioLossWeight
        config.includeFFN = includeFFN
        config.transformerQuant = transformerQuant
        config.gradientAccumulationSteps = gradAccum
        config.warmupSteps = warmupSteps
        config.seed = seed
        config.hfToken = hfToken
        config.modelsDir = modelsDir
        config.gemmaPath = gemmaPath
        config.ltxWeightsPath = ltxWeights

        print("Dataset: \(dataset)")
        print("Output: \(output)")
        print("Model: \(model)")
        print("Resolution: \(width)x\(height), Frames: \(frames)")
        print("Rank: \(rank), Alpha: \(config.alpha), LR: \(lr)")
        print("Steps: \(steps), Save every: \(saveEvery)")
        print("Quant: \(transformerQuant)")
        if audio { print("Audio: enabled (weight: \(audioLossWeight))") }
        print()

        let trainer = LoRATrainer(
            config: config,
            datasetPath: dataset,
            outputDir: output
        )

        try await trainer.train { progress in
            if progress.step % 10 == 0 || progress.step == 1 {
                print(progress.status)
                fflush(stdout)
            }
        }
    }
}
