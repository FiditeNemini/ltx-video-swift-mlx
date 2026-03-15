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

    /// Controller for pause/resume/stop (optional, nil = no control)
    public var controller: TrainingController?

    public init(config: LoRATrainingConfig, datasetPath: String, outputDir: String) {
        self.config = config
        self.datasetPath = datasetPath
        self.outputDir = outputDir
    }

    /// Run the full training pipeline.
    ///
    /// Supports pause/resume/stop via ``TrainingController``:
    /// - **Pause**: saves checkpoint, blocks until resumed or stopped
    /// - **Stop**: saves checkpoint, exits cleanly
    /// - **Resume**: auto-detects latest checkpoint in output directory
    ///
    /// - Parameters:
    ///   - resumeFromStep: If set, resume from this checkpoint step instead of starting fresh.
    ///     Pass `nil` to auto-detect the latest checkpoint, or `0` to force a fresh start.
    ///   - onProgress: Callback invoked at each training step with progress info.
    public func train(
        resumeFromStep: Int? = nil,
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
            extractAudio: false
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
            throw TrainingError.invalidConfig(
                "Audio LoRA training is not yet supported. " +
                "Audio latent caching requires AudioProcessor, which will be added in a future version."
            )
        }

        // Step 3: Build latent cache
        let cacheDir = (outputDir as NSString).appendingPathComponent("latent_cache")
        let cache = LatentCache(cacheDir: cacheDir, triggerWord: config.triggerWord)

        print("Building latent cache...")
        try await cache.build(from: dataset, pipeline: pipeline) { idx, total, filename in
            print("  Encoding [\(idx + 1)/\(total)] \(filename)")
        }

        // Load cached samples and validate shapes
        try cache.loadAll()
        guard !cache.isEmpty else {
            throw TrainingError.datasetError("No cached samples available after encoding")
        }
        try cache.validate(config: config)
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
            includeAudio: false,
            includeFFN: config.includeFFN
        )
        print("Injected \(injection.injectedCount) LoRA layers")

        // Count trainable parameters
        let trainableParams = transformer.trainableParameters().flattenedValues()
        let totalTrainableElements = trainableParams.reduce(0) { $0 + $1.size }
        print("Trainable parameters: \(totalTrainableElements) elements (\(trainableParams.count) tensors)")


        guard !trainableParams.isEmpty else {
            throw TrainingError.modelError(
                "No trainable parameters found after LoRA injection. " +
                "Injected \(injection.injectedCount) layers but none are trainable."
            )
        }

        // Step 6: Set up optimizer
        let optimizer = AdamW(learningRate: config.learningRate, weightDecay: config.weightDecay)

        // Step 6b: Handle resume from checkpoint
        var firstStep = 0
        var trainingState = TrainingState(
            totalSteps: config.maxSteps,
            modelType: config.model,
            loraRank: config.rank,
            loraAlpha: config.alpha,
            learningRate: config.learningRate,
            rngSeed: config.seed
        )

        // Determine resume step
        let requestedStep = resumeFromStep
        let checkpoint: (step: Int, path: String)?
        if let step = requestedStep, step > 0 {
            // Explicit resume from a specific step
            let jsonPath = (outputDir as NSString).appendingPathComponent("checkpoint-step\(step).json")
            checkpoint = (step, jsonPath)
        } else if requestedStep == nil {
            // Auto-detect latest checkpoint
            checkpoint = TrainingState.findLatestCheckpoint(in: outputDir)
        } else {
            // resumeFromStep == 0 → force fresh start
            checkpoint = nil
        }

        if let ckpt = checkpoint {
            let weightsPath = (outputDir as NSString).appendingPathComponent("checkpoint-step\(ckpt.step).safetensors")
            let fm = FileManager.default
            if fm.fileExists(atPath: weightsPath) && fm.fileExists(atPath: ckpt.path) {
                // Load saved state
                let savedState = try TrainingState.load(from: ckpt.path)
                guard savedState.isCompatible(
                    modelType: config.model, rank: config.rank,
                    alpha: config.alpha, lr: config.learningRate
                ) else {
                    throw TrainingError.checkpointError(
                        "Checkpoint at step \(ckpt.step) is incompatible with current config " +
                        "(model/rank/alpha/lr mismatch)"
                    )
                }

                // Load LoRA weights from checkpoint into the injected model
                let loraWeights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
                // The checkpoint is in PEFT format (lora_A/lora_B, transposed).
                // We need to load them back into the LoRALinear layers.
                try LoRAInjector.loadLoRAWeights(loraWeights, into: transformer, rank: config.rank)

                firstStep = savedState.currentStep
                trainingState = savedState
                trainingState.status = .running
                print("Resumed from checkpoint at step \(firstStep)")
            }
        }

        // Clean up control files from previous runs
        TrainingController.cleanupControlFiles(outputDir: outputDir)
        controller?.setStatus(.running)

        // Step 7: Training loop
        print("\nStarting training...")
        print("Steps: \(config.maxSteps), LR: \(config.learningRate), Rank: \(config.rank)")
        print("Resolution: \(config.width)x\(config.height), Frames: \(config.numFrames)")
        if firstStep > 0 {
            print("Resuming from step \(firstStep)")
        }
        if config.gradientAccumulationSteps > 1 {
            print("Gradient accumulation: \(config.gradientAccumulationSteps) steps")
        }
        if let trigger = config.triggerWord {
            print("Trigger word: \(trigger)")
        }
        print()
        fflush(stdout)


        // Create loss+grad function using valueAndGrad(model:_:) with [MLXArray] signature
        // Pack all inputs into an [MLXArray] and unpack inside the closure
        // latentShape is passed via the arrays to allow per-sample shapes from cache
        let lossGradFn = valueAndGrad(model: transformer) {
            (model: Module, arrays: [MLXArray]) -> [MLXArray] in
            // Latent from cache is 5D (B,C,T,H,W) — patchify to 3D (B, T*H*W, C)
            // for transformer input. All noise/target ops happen in patchified space.
            let videoLatent5D = arrays[0]
            let videoLatent = patchify(videoLatent5D)
            let promptEmbeddings = arrays[1]
            let promptMask = arrays[2]
            let lFrames = Int(arrays[3].item(Int32.self))
            let lHeight = Int(arrays[4].item(Int32.self))
            let lWidth = Int(arrays[5].item(Int32.self))
            let batchSize = videoLatent.dim(0)

            // Sample random sigma — flow-matching noise schedule
            let sigma = MLXRandom.uniform(0..<1, [batchSize, 1]).asType(.float32)
            let videoNoise = MLXRandom.normal(videoLatent.shape).asType(videoLatent.dtype)
            let sigmaVideo = sigma.asType(videoLatent.dtype)
            let noisyVideo = (1 - sigmaVideo) * videoLatent + sigmaVideo * videoNoise
            let sigmaFlat = sigma.squeezed(axis: 1)

            if let ltx2 = model as? LTX2Transformer {
                // LTX2 requires both video+audio inputs; pass zero audio for video-only training
                let dummyAudioLatent = MLXArray.zeros([batchSize, 1, 128]).asType(DType.bfloat16)
                let (predVideo, _) = ltx2(
                    videoLatent: noisyVideo.asType(DType.bfloat16),
                    audioLatent: dummyAudioLatent,
                    videoContext: promptEmbeddings,
                    audioContext: promptEmbeddings,
                    videoTimesteps: sigmaFlat,
                    audioTimesteps: sigmaFlat,
                    videoContextMask: nil,
                    audioContextMask: nil,
                    videoLatentShape: (lFrames, lHeight, lWidth),
                    audioNumFrames: 1
                )
                let target = (videoNoise - videoLatent).asType(predVideo.dtype)
                return [mseLoss(predictions: predVideo, targets: target, reduction: .mean)]
            } else if let ltx = model as? LTXTransformer {
                // Video-only transformer (legacy)
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
                fatalError("Unknown transformer type: \(type(of: model))")
            }
        }

        let accumSteps = config.gradientAccumulationSteps
        let accumScale = 1.0 / Float(accumSteps)
        var bestLoss: Float = trainingState.bestLoss.isFinite ? trainingState.bestLoss : .infinity
        var lossHistory: [(step: Int, loss: Float)] = trainingState.recentLosses.enumerated().map {
            (step: $0.offset + 1, loss: $0.element)
        }
        let loopStart = Date()
        var wasStopped = false

        for step in firstStep..<config.maxSteps {
            // --- Control: force stop ---
            if let ctrl = controller {
                if ctrl.shouldForceStop() {
                    wasStopped = true
                    break
                }
                // --- Control: graceful stop (save checkpoint first) ---
                if ctrl.shouldStop() {
                    print("  Stop requested, saving checkpoint...")
                    try saveCheckpoint(
                        model: transformer, step: step, loss: bestLoss,
                        state: &trainingState
                    )
                    ctrl.setStatus(.cancelled)
                    ctrl.notifyObservers { $0.trainingFinished(success: true, message: "Stopped at step \(step)") }
                    wasStopped = true
                    break
                }
                // --- Control: pause (save checkpoint, then block) ---
                if ctrl.shouldPause() {
                    print("  Pausing at step \(step)...")
                    try saveCheckpoint(
                        model: transformer, step: step, loss: lossHistory.last?.loss ?? 0,
                        state: &trainingState, isPause: true
                    )
                    ctrl.notifyObservers { $0.trainingPaused(atStep: step) }
                    // Block until resumed or stopped
                    if !ctrl.waitWhilePaused() {
                        wasStopped = true
                        break
                    }
                    print("  Resumed at step \(step)")
                    ctrl.notifyObservers { $0.trainingResumed(atStep: step) }
                }
            }
            // Apply LR warmup (before optimizer step)
            let currentLR: Float
            if step < config.warmupSteps {
                currentLR = config.learningRate * Float(step + 1) / Float(config.warmupSteps)
                optimizer.learningRate = currentLR
            } else {
                currentLR = config.learningRate
            }

            // Training step
            var accumGrads: ModuleParameters? = nil
            var accumLoss: Float = 0

            for _ in 0..<accumSteps {
                guard let sample = cache.randomSample() else {
                    throw TrainingError.datasetError("No cached samples available")
                }

                let inputs: [MLXArray] = [
                    sample.videoLatent,
                    sample.promptEmbeddings,
                    sample.promptMask,
                    MLXArray(Int32(sample.latentFrames)),
                    MLXArray(Int32(sample.latentHeight)),
                    MLXArray(Int32(sample.latentWidth)),
                ]

                let (losses, grads) = lossGradFn(transformer, inputs)
                let loss = losses[0]

                if accumSteps > 1 {
                    let scaledGrads = scaleParameters(grads, by: accumScale)
                    if var existing = accumGrads {
                        addParameters(scaledGrads, to: &existing)
                        accumGrads = existing
                    } else {
                        accumGrads = scaledGrads
                    }
                } else {
                    accumGrads = grads
                }

                eval(loss)
                accumLoss += loss.item(Float.self)
            }

            accumLoss /= Float(accumSteps)

            // Gradient clipping (max norm)
            if config.maxGradNorm > 0, var grads = accumGrads {
                let gradNorm = computeGradNorm(grads)
                if gradNorm > config.maxGradNorm {
                    let clipScale = config.maxGradNorm / gradNorm
                    grads = scaleParameters(grads, by: clipScale)
                }
                accumGrads = grads
            }

            // Update parameters and materialize
            optimizer.update(model: transformer, gradients: accumGrads!)
            eval(transformer, optimizer)

            // Progress callback
            let elapsed = Date().timeIntervalSince(loopStart)
            let progress = TrainingProgress(
                step: step + 1,
                totalSteps: config.maxSteps,
                loss: accumLoss,
                learningRate: currentLR,
                elapsedSeconds: elapsed,
                samplesPerSecond: Double(step + 1) / elapsed
            )
            onProgress?(progress)

            // Track best loss and record for learning curve
            if accumLoss < bestLoss {
                bestLoss = accumLoss
            }
            lossHistory.append((step: step + 1, loss: accumLoss))
            trainingState.recordLoss(accumLoss, atStep: step + 1)
            trainingState.totalTrainingTime = elapsed

            if lossHistory.count >= 2 {
                LearningCurveSVG.generate(
                    lossHistory: lossHistory,
                    outputDir: outputDir,
                    smoothingWindow: 20
                )
            }

            // Notify controller
            controller?.notifyStepCompleted(step: step + 1, totalSteps: config.maxSteps, loss: accumLoss)

            // Save checkpoint (scheduled, on-demand, or final step)
            let isManualCheckpoint = controller?.shouldCheckpoint() ?? false
            let isScheduledCheckpoint = (step + 1) % config.saveEvery == 0
            let isFinalStep = step == config.maxSteps - 1
            if isScheduledCheckpoint || isManualCheckpoint || isFinalStep {
                print("  Saving checkpoint at step \(step + 1)...")
                fflush(stdout)
                try saveCheckpoint(
                    model: transformer, step: step + 1, loss: accumLoss,
                    state: &trainingState
                )
            }
        }

        // Save final LoRA (unless force-stopped)
        if !wasStopped || !(controller?.shouldForceStop() ?? false) {
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
            controller?.setStatus(.completed)
            controller?.notifyObservers { $0.trainingFinished(success: true, message: "Completed at step \(config.maxSteps)") }
        } else {
            let totalTime = Date().timeIntervalSince(startTime)
            print("\nTraining stopped.")
            print("  Total time: \(String(format: "%.1f", totalTime))s")
            print("  Best loss: \(String(format: "%.6f", bestLoss))")
            controller?.setStatus(.cancelled)
        }
    }

    // MARK: - Private Helpers

    private func saveCheckpoint(
        model: Module,
        step: Int,
        loss: Float,
        state: inout TrainingState,
        isPause: Bool = false
    ) throws {
        state.lastCheckpointAt = Date()
        state.status = isPause ? .paused : .checkpointing
        state.isPauseCheckpoint = isPause

        try LoRASaver.saveCheckpoint(
            model: model,
            step: step,
            loss: loss,
            outputDir: outputDir,
            rank: config.rank,
            alpha: config.alpha
        )

        // Save training state alongside the checkpoint
        let statePath = (outputDir as NSString).appendingPathComponent("checkpoint-step\(step).json")
        try state.save(to: statePath)

        controller?.notifyObservers { $0.trainingCheckpointSaved(step: step, path: outputDir) }
    }
}

// MARK: - Gradient Helpers

/// Scale all leaf arrays in a ModuleParameters tree by a scalar factor
private func scaleParameters(_ params: ModuleParameters, by factor: Float) -> ModuleParameters {
    let factorArray = MLXArray(factor)
    return params.mapValues { (arr: MLXArray) -> MLXArray in
        arr * factorArray
    }
}

/// Add source parameters into destination (element-wise accumulation)
private func addParameters(_ source: ModuleParameters, to destination: inout ModuleParameters) {
    destination = destination.mapValues(source) { dst, src in
        dst + (src ?? MLXArray(Float(0)))
    }
}

/// Compute global L2 norm of all gradient arrays.
/// Uses a single MLX graph to avoid CPU-GPU round trips per tensor.
private func computeGradNorm(_ params: ModuleParameters) -> Float {
    let values = params.flattenedValues()
    guard !values.isEmpty else { return 0 }
    // Build a single sum-of-squares across all grad tensors in one graph
    let sumSq = values.reduce(MLXArray(Float(0))) { acc, arr in
        acc + (arr * arr).sum()
    }
    eval(sumSq)
    return sqrt(sumSq.item(Float.self))
}
