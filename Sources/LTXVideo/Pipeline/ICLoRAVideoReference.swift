// ICLoRAVideoReference.swift - Video-to-video through a reference-conditioned IC-LoRA
// Copyright 2025

import Foundation
@preconcurrency import MLX

extension LTXPipeline {

    /// Re-render a clip with an IC-LoRA that consumes a reference video in context.
    ///
    /// This is the shape LTX's pixel spatial upscaler takes: the reference is the
    /// low-resolution clip, the output is a re-render at `referenceDownscaleFactor`
    /// times its linear resolution, with fine detail **synthesised** rather than
    /// interpolated. Composition, motion and identity carry over from the
    /// reference; texture and micro-contrast are invented.
    ///
    /// The scale factor is not a parameter: it is read from the adapter's
    /// safetensors metadata (`reference_downscale_factor`), which is how the
    /// reference implementation derives it too. Passing a mismatched output size
    /// would silently ask the model for a mapping it was never trained on.
    ///
    /// Unlike `generateLipDub`, this path is video-only — no audio reference, no
    /// dialogue prompt — so it runs on the single-stream transformer.
    ///
    /// - Parameters:
    ///   - prompt: describes the scene. The upscaler still conditions on text.
    ///   - referenceVideoPath: the low-resolution clip.
    ///   - loraPath: the IC-LoRA. Fused for the run and unfused afterwards.
    ///   - config: target geometry. `width`/`height` are the **output** size.
    ///   - loraScale: adapter strength; these weights ship pre-scaled for 1.0.
    public func generateWithVideoReference(
        prompt: String,
        referenceVideoPath: String,
        loraPath: String,
        config: LTXVideoGenerationConfig,
        loraScale: Float = 1.0,
        onProgress: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> VideoGenerationResult {
        let startTime = Date()
        try config.validate()
        guard FileManager.default.fileExists(atPath: referenceVideoPath) else {
            throw LTXError.fileNotFound("Reference video not found: \(referenceVideoPath)")
        }

        let downscaleFactor = LoRALoader.referenceDownscaleFactor(from: loraPath)
        guard downscaleFactor >= 1 else {
            throw LTXError.invalidLoRA("reference_downscale_factor must be >= 1")
        }
        guard config.width % downscaleFactor == 0, config.height % downscaleFactor == 0 else {
            throw LTXError.invalidConfiguration(
                "Output \(config.width)x\(config.height) must divide by the adapter's "
                + "reference_downscale_factor (\(downscaleFactor))")
        }
        let referenceWidth = config.width / downscaleFactor
        let referenceHeight = config.height / downscaleFactor

        if !isLoaded { try await loadModels(progressCallback: nil) }
        let beacon = RuntimeBeacon.begin(task: "ic-lora-video", model: model.rawValue)
        defer { beacon?.end() }

        // 1. Encode the reference at its own, smaller geometry.
        LTXDebug.log("[ic-lora] reference \(referenceWidth)x\(referenceHeight) → output "
            + "\(config.width)x\(config.height) (x\(downscaleFactor)), \(config.numFrames) frames")
        let referenceLatent = try await encodeVideo(
            path: referenceVideoPath,
            width: referenceWidth, height: referenceHeight, numFrames: config.numFrames)

        // 2. Text conditioning.
        let encoded = try await encodeText(prompt)
        unloadGemmaIfConfigured()

        // 3. Fuse the adapter for this run only. Fusion is destructive, so the
        //    unfuse is deferred rather than left to the caller.
        let fusedLayers = try fuseLoRA(from: loraPath, scale: loraScale)
        defer { unfuseLoRA() }
        LTXDebug.log("[ic-lora] fused \(fusedLayers) layer-pairs at scale \(loraScale)")

        // 4. Denoise at the output geometry with the reference appended in context.
        let targetShape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames, height: config.height, width: config.width)
        guard let transformerConfig = (transformer?.config ?? ltx2Transformer?.config) else {
            throw LTXError.modelNotLoaded("Transformer not loaded")
        }
        let referenceContext = buildVideoReference(
            referenceLatent: referenceLatent,
            targetShape: targetShape,
            downscaleFactor: downscaleFactor,
            hasAudio: false,
            refConfig: transformerConfig)
        LTXDebug.log("[ic-lora] reference tokens=\(referenceContext.guideCount) "
            + "target tokens=\(referenceContext.originalCount)")

        let sigmas = DISTILLED_SIGMA_VALUES
        let steps = sigmas.count - 1
        var latent = generateNoise(shape: targetShape, seed: config.seed)
        MLX.eval(latent)

        for step in 0 ..< steps {
            let sigma = sigmas[step]
            onProgress?(GenerationProgress(
                currentStep: step, totalSteps: steps, sigma: sigma, phase: .denoising))
            let velocity = runDenoiseStep(
                sigma: sigma,
                videoLatent: latent,
                audioLatentPacked: nil,
                shape: targetShape,
                videoAppendCtx: referenceContext,
                audioRefCtx: nil,
                audioNumFrames: 0,
                videoTextEmbeddings: encoded.embeddings,
                audioTextEmbeddings: encoded.embeddings,
                textMask: encoded.mask)
            latent = latent + MLXArray(sigmas[step + 1] - sigma) * velocity.video
            MLX.eval(latent)
        }

        // 5. Decode.
        onProgress?(GenerationProgress(
            currentStep: steps, totalSteps: steps, sigma: 0, phase: .decoding))
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
            audioWaveform: nil,
            audioSampleRate: nil,
            effectivePrompt: prompt)
    }
}
