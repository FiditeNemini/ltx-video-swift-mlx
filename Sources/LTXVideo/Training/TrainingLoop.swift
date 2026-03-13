// TrainingLoop.swift - Main LoRA training loop
// Copyright 2026

import Foundation
@preconcurrency import MLX
import MLXRandom
import MLXNN
import MLXOptimizers

/// Training progress callback
public typealias TrainingProgressCallback = @Sendable (TrainingProgress) -> Void

/// Training progress information
public struct TrainingProgress: Sendable {
    public let step: Int
    public let totalSteps: Int
    public let loss: Float
    public let learningRate: Float
    public let elapsedSeconds: Double
    public let samplesPerSecond: Double

    public var status: String {
        let pct = Int(Double(step) / Double(totalSteps) * 100)
        return "Step \(step)/\(totalSteps) [\(pct)%] loss=\(String(format: "%.6f", loss)) lr=\(String(format: "%.2e", learningRate))"
    }
}

/// Main LoRA training loop for LTX-2 transformers.
///
/// Orchestrates the full training pipeline:
/// 1. Load models (VAE + Gemma + transformer)
/// 2. Build latent cache (encode all videos + captions)
/// 3. Unload VAE + Gemma (free memory)
/// 4. Inject LoRA into transformer
/// 5. Training loop with AdamW + flow-matching loss
/// 6. Save checkpoints
public class LoRATrainer {
    let config: LoRATrainingConfig
    let datasetPath: String
    let outputDir: String

    public init(config: LoRATrainingConfig, datasetPath: String, outputDir: String) {
        self.config = config
        self.datasetPath = datasetPath
        self.outputDir = outputDir
    }

    /// Run the full training pipeline
    public func train(
        onProgress: TrainingProgressCallback? = nil
    ) async throws {
        try config.validate()

        let startTime = Date()

        // Create output directory
        let fm = FileManager.default
        if !fm.fileExists(atPath: outputDir) {
            try fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        }

        // Configure models directory
        if let dir = config.modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }

        // Set seed if specified
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        // Step 1: Load dataset
        print("Loading dataset from \(datasetPath)...")
        let dataset = try VideoDataset(
            directory: datasetPath,
            width: config.width,
            height: config.height,
            numFrames: config.numFrames,
            extractAudio: config.includeAudio
        )
        print("Found \(dataset.count) training samples")

        // Step 2: Create pipeline and load models
        guard let modelVariant = LTXModel(rawValue: config.model) else {
            throw TrainingError.invalidConfig("Invalid model variant: \(config.model)")
        }

        guard let quantOption = TransformerQuantization(rawValue: config.transformerQuant) else {
            throw TrainingError.invalidConfig("Invalid quantization: \(config.transformerQuant)")
        }

        let pipeline = LTXPipeline(
            model: modelVariant,
            quantization: LTXQuantizationConfig(transformer: quantOption),
            hfToken: config.hfToken
        )

        print("Loading models...")
        try await pipeline.loadModels(
            progressCallback: { progress in
                print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            },
            gemmaModelPath: config.gemmaPath,
            ltxWeightsPath: config.ltxWeightsPath
        )

        if config.includeAudio {
            print("Loading audio models...")
            try await pipeline.loadAudioModels { progress in
                print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            }
        }

        // Step 3: Build latent cache
        let cacheDir = (outputDir as NSString).appendingPathComponent("latent_cache")
        let cache = LatentCache(cacheDir: cacheDir, includeAudio: config.includeAudio)

        print("Building latent cache...")
        try await cache.build(from: dataset, pipeline: pipeline) { idx, total, filename in
            print("  Encoding [\(idx + 1)/\(total)] \(filename)")
        }

        // Load cached samples
        try cache.loadAll()
        guard !cache.isEmpty else {
            throw TrainingError.datasetError("No cached samples available after encoding")
        }
        print("Cached \(cache.count) samples")

        // Step 4: Unload Gemma and VAE encoder to free memory
        print("Unloading Gemma and VAE encoder...")
        await pipeline.clearGemma()
        await pipeline.unloadVAEEncoder()
        Memory.clearCache()

        // Step 5: Get transformer and inject LoRA
        let transformerRef = try await pipeline.getTransformerForTraining()
        let transformer = transformerRef.module
        print("Injecting LoRA (rank=\(config.rank), alpha=\(config.alpha))...")
        let injection = LoRAInjector.inject(
            into: transformer,
            rank: config.rank,
            alpha: config.alpha,
            includeAudio: config.includeAudio,
            includeFFN: config.includeFFN
        )
        print("Injected \(injection.injectedCount) LoRA layers")

        // Count trainable parameters
        let trainableParams = transformer.trainableParameters().flattenedValues()
        let totalTrainableElements = trainableParams.reduce(0) { $0 + $1.size }
        print("Trainable parameters: \(totalTrainableElements) elements (\(trainableParams.count) tensors)")

        // Step 6: Set up optimizer
        let optimizer = AdamW(learningRate: config.learningRate, weightDecay: config.weightDecay)

        // Step 7: Compute latent shape from config
        let latentShape = VideoLatentShape.fromPixelDimensions(
            frames: config.numFrames,
            height: config.height,
            width: config.width
        )

        // Step 8: Training loop
        print("\nStarting training...")
        print("Steps: \(config.maxSteps), LR: \(config.learningRate), Rank: \(config.rank)")
        print("Resolution: \(config.width)x\(config.height), Frames: \(config.numFrames)")
        print()

        let audioWeight = config.audioLossWeight
        let lFrames = latentShape.frames
        let lHeight = latentShape.height
        let lWidth = latentShape.width

        // Create loss+grad function using valueAndGrad(model:_:) with [MLXArray] signature
        // Pack all inputs into an [MLXArray] and unpack inside the closure
        let lossGradFn = valueAndGrad(model: transformer) {
            (model: Module, arrays: [MLXArray]) -> [MLXArray] in
            let videoLatent = arrays[0]
            let promptEmbeddings = arrays[1]
            let promptMask = arrays[2]
            let batchSize = videoLatent.dim(0)

            // Sample random sigma
            let sigma = MLXRandom.uniform(0..<1, [batchSize, 1]).asType(.float32)
            let videoNoise = MLXRandom.normal(videoLatent.shape).asType(videoLatent.dtype)
            let sigmaVideo = sigma.asType(videoLatent.dtype)
            let noisyVideo = (1 - sigmaVideo) * videoLatent + sigmaVideo * videoNoise
            let sigmaFlat = sigma.squeezed(axis: 1)

            if let ltx2 = model as? LTX2Transformer,
               arrays.count >= 6 {
                // Dual video/audio path
                let audioLatent = arrays[3]
                let audioEmbeddings = arrays[4]
                let audioMask = arrays[5]
                let audioNumFrames = audioLatent.dim(1)

                let audioNoise = MLXRandom.normal(audioLatent.shape).asType(audioLatent.dtype)
                let sigmaAudio = sigma.asType(audioLatent.dtype)
                let noisyAudio = (1 - sigmaAudio) * audioLatent + sigmaAudio * audioNoise

                let (predVideo, predAudio) = ltx2(
                    videoLatent: noisyVideo.asType(DType.bfloat16),
                    audioLatent: noisyAudio.asType(DType.bfloat16),
                    videoContext: promptEmbeddings,
                    audioContext: audioEmbeddings,
                    videoTimesteps: sigmaFlat,
                    audioTimesteps: sigmaFlat,
                    videoContextMask: promptMask,
                    audioContextMask: audioMask,
                    videoLatentShape: (lFrames, lHeight, lWidth),
                    audioNumFrames: audioNumFrames
                )

                let videoTarget = (videoNoise - videoLatent).asType(predVideo.dtype)
                let audioTarget = (audioNoise - audioLatent).asType(predAudio.dtype)
                let videoLoss = mseLoss(predictions: predVideo, targets: videoTarget, reduction: .mean)
                let audioLoss = mseLoss(predictions: predAudio, targets: audioTarget, reduction: .mean)
                return [videoLoss + audioWeight * audioLoss]
            } else if let ltx = model as? LTXTransformer {
                // Video-only path
                let predicted = ltx(
                    latent: noisyVideo.asType(DType.bfloat16),
                    context: promptEmbeddings,
                    timesteps: sigmaFlat,
                    contextMask: promptMask,
                    latentShape: (lFrames, lHeight, lWidth)
                )
                let target = (videoNoise - videoLatent).asType(predicted.dtype)
                return [mseLoss(predictions: predicted, targets: target, reduction: .mean)]
            } else {
                // Fallback — shouldn't happen
                return [MLXArray(Float(0))]
            }
        }

        var bestLoss: Float = .infinity
        let loopStart = Date()

        for step in 0..<config.maxSteps {
            // Get random sample
            guard let sample = cache.randomSample() else {
                throw TrainingError.datasetError("No cached samples available")
            }

            // Pack inputs
            var inputs: [MLXArray] = [
                sample.videoLatent,
                sample.promptEmbeddings,
                sample.promptMask,
            ]

            if config.includeAudio {
                inputs.append(sample.audioLatent ?? MLXArray.zeros([1, 1, 128]))
                inputs.append(sample.audioEmbeddings ?? sample.promptEmbeddings)
                inputs.append(sample.audioMask ?? sample.promptMask)
            }

            // Compute loss and gradients
            let (losses, grads) = lossGradFn(transformer, inputs)
            let loss = losses[0]

            // Update parameters
            optimizer.update(model: transformer, gradients: grads)

            // Materialize computation
            eval(loss, transformer.trainableParameters())
            Memory.clearCache()

            let lossValue = loss.item(Float.self)

            // Apply LR warmup
            let currentLR: Float
            if step < config.warmupSteps {
                currentLR = config.learningRate * Float(step + 1) / Float(config.warmupSteps)
                optimizer.learningRate = currentLR
            } else {
                currentLR = config.learningRate
            }

            // Progress callback
            let elapsed = Date().timeIntervalSince(loopStart)
            let progress = TrainingProgress(
                step: step + 1,
                totalSteps: config.maxSteps,
                loss: lossValue,
                learningRate: currentLR,
                elapsedSeconds: elapsed,
                samplesPerSecond: Double(step + 1) / elapsed
            )
            onProgress?(progress)

            // Track best loss
            if lossValue < bestLoss {
                bestLoss = lossValue
            }

            // Save checkpoint
            if (step + 1) % config.saveEvery == 0 || step == config.maxSteps - 1 {
                print("  Saving checkpoint at step \(step + 1)...")
                try LoRASaver.saveCheckpoint(
                    model: transformer,
                    step: step + 1,
                    loss: lossValue,
                    outputDir: outputDir,
                    rank: config.rank,
                    alpha: config.alpha
                )
            }
        }

        // Save final LoRA
        let finalPath = (outputDir as NSString).appendingPathComponent("lora-final.safetensors")
        let savedCount = try LoRASaver.save(
            model: transformer,
            to: finalPath,
            rank: config.rank,
            alpha: config.alpha
        )

        let totalTime = Date().timeIntervalSince(startTime)
        print("\nTraining complete!")
        print("  Total time: \(String(format: "%.1f", totalTime))s")
        print("  Best loss: \(String(format: "%.6f", bestLoss))")
        print("  Saved \(savedCount) LoRA layers to \(finalPath)")
    }
}
