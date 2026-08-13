// DevSingleStage.swift - Full-quality dev-checkpoint generation (CFG + STG)
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXRandom

extension LTXPipeline {

    /// Single-stage generation on a dev checkpoint at full quality: 30 steps by
    /// default, classical CFG 3.0 against an empty negative prompt, STG on block
    /// 28, guidance rescale 0.7 — the parameters `retake` already uses for dev,
    /// which mirror upstream's `ti2vid_one_stage.py`.
    ///
    /// This exists because the two-stage path is distilled-only by design: its
    /// fixed 8-step schedule is a property of the distilled training, and a dev
    /// checkpoint run through it produces mush (that is what the distilled LoRA
    /// is *for*). Guidance is applied on the denoised x₀, not on velocities,
    /// matching the Lightricks pipelines and the existing retake loop.
    public func generateVideoDev(
        prompt: String,
        config: LTXVideoGenerationConfig,
        onProgress: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> VideoGenerationResult {
        let startTime = Date()
        try config.validate()
        guard model.isForTraining else {
            throw LTXError.invalidConfiguration(
                "generateVideoDev is for dev checkpoints; \(model.displayName) is distilled — "
                + "use generateVideo, or fuse the distilled LoRA and use generateVideo on dev.")
        }
        if !isLoaded { try await loadModels(progressCallback: nil) }
        let beacon = RuntimeBeacon.begin(task: "generate-dev", model: model.rawValue)
        defer { beacon?.end() }

        // Text conditioning, positive and (for CFG) the empty negative.
        let encoded = try await encodeText(prompt, enhance: config.enhancePrompt)
        let negative = try await encodeText("")
        unloadGemmaIfConfigured()

        // i2v: the conditioning image rides as an appended guide token, exactly as
        // in the two-stage path.
        let shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames, height: config.height, width: config.width)
        guard let transformerConfig = (transformer?.config ?? ltx2Transformer?.config) else {
            throw LTXError.modelNotLoaded("Transformer not loaded")
        }
        var appendCtx: AppendKeyframeContext? = nil
        let keyframes: [KeyframeInput] = !config.keyframes.isEmpty
            ? config.keyframes
            : config.imagePath.map { [KeyframeInput(path: $0, pixelFrameIndex: 0)] } ?? []
        if !keyframes.isEmpty {
            let encodedKF = try await encodeKeyframes(
                keyframes, width: config.width, height: config.height)
            appendCtx = prepareKeyframeAppend(
                encoded: encodedKF, shape: shape, hasAudio: false,
                refConfig: transformerConfig, stageLabel: "dev single-stage")
            unloadVAEEncoder()
        }

        // Dev guidance parameters — identical to the retake dev path.
        let cfgScale: Float = 3.0
        let stgScale: Float = 1.0
        let stgBlocks = [28]
        let guidanceRescale: Float = 0.7

        // Token-shifted schedule: unlike the distilled model's fixed sigmas, the
        // dev schedule shifts with the token count.
        let scheduler = LTXScheduler(isDistilled: false)
        scheduler.setTimesteps(
            numSteps: config.numSteps, distilled: false, latentTokenCount: shape.tokenCount)
        let sigmas = scheduler.sigmas
        let numSteps = sigmas.count - 1

        var latent = generateNoise(shape: shape, seed: config.seed)
        MLX.eval(latent)
        LTXDebug.log("[dev] \(numSteps) steps, cfg=\(cfgScale), stg=\(stgScale)@\(stgBlocks), "
            + "rescale=\(guidanceRescale), tokens=\(shape.tokenCount)")

        for step in 0 ..< numSteps {
            let sigma = sigmas[step]
            onProgress?(GenerationProgress(
                currentStep: step, totalSteps: numSteps, sigma: sigma, phase: .denoising))

            // Each pass reuses runDenoiseStep so the appended guide tokens, RoPE
            // extension and velocity cropping stay identical to every other path.
            func denoisedX0(context: MLXArray, mask: MLXArray?) -> MLXArray {
                let velocity = runDenoiseStep(
                    sigma: sigma, videoLatent: latent, audioLatentPacked: nil,
                    shape: shape, videoAppendCtx: appendCtx, audioRefCtx: nil,
                    audioNumFrames: 0,
                    videoTextEmbeddings: context, audioTextEmbeddings: context,
                    textMask: mask)
                return latent - MLXArray(sigma) * velocity.video
            }

            let condX0 = denoisedX0(context: encoded.embeddings, mask: encoded.mask)
            var combined = condX0

            // CFG on x0: pred = cond + (scale − 1)(cond − uncond)
            let negX0 = denoisedX0(context: negative.embeddings, mask: negative.mask)
            combined = combined + MLXArray(cfgScale - 1.0) * (condX0 - negX0)

            // STG: perturbed pass with self-attention skipped on the STG blocks.
            transformer?.setSTGBlocks(stgBlocks)
            ltx2Transformer?.setSTGBlocks(stgBlocks)
            let stgX0 = denoisedX0(context: encoded.embeddings, mask: encoded.mask)
            transformer?.clearSTG()
            ltx2Transformer?.clearSTG()
            combined = combined + MLXArray(stgScale) * (condX0 - stgX0)

            // Rescale toward the conditioned prediction's variance.
            if guidanceRescale > 0 {
                let condStd = condX0.asType(.float32).variance().sqrt()
                let predStd = combined.asType(.float32).variance().sqrt()
                let factor = MLXArray(guidanceRescale) * (condStd / predStd)
                    + MLXArray(1.0 - guidanceRescale)
                combined = combined * factor
            }

            // Euler on the recomposed velocity.
            let velocity = (latent - combined) / MLXArray(sigma)
            latent = (latent.asType(.float32)
                + velocity.asType(.float32) * MLXArray(sigmas[step + 1] - sigma))
            MLX.eval(latent)
            if (step + 1) % 5 == 0 { Memory.clearCache() }
        }

        onProgress?(GenerationProgress(
            currentStep: numSteps, totalSteps: numSteps, sigma: 0, phase: .decoding))
        guard let decoder = vaeDecoder else {
            throw LTXError.modelNotLoaded("VAE decoder not loaded")
        }
        let frames = decodeVideo(
            latent: latent, decoder: decoder, timestep: nil,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap)
        MLX.eval(frames)

        return VideoGenerationResult(
            frames: frames,
            seed: config.seed ?? 0,
            generationTime: Date().timeIntervalSince(startTime),
            audioWaveform: nil, audioSampleRate: nil,
            effectivePrompt: prompt)
    }
}
