// TemporalInterpolation.swift - Frame-rate doubling through the temporal upsampler
// Copyright 2026

import Foundation
@preconcurrency import MLX

extension LTXPipeline {

    /// Double a clip's frame rate: `n` frames become `2n - 1`, at the same
    /// resolution and duration.
    ///
    /// Two steps, and the second is what makes it worth doing. The latent
    /// temporal upsampler lays the existing motion onto a denser frame grid —
    /// on its own that is an interpolation, and it looks like one. A short
    /// refinement pass then re-denoises the whole clip with an **ancestral**
    /// sampler, whose re-injected noise lets the model invent plausible
    /// in-between motion rather than average the frames it already has.
    ///
    /// ## What this is not
    ///
    /// Upstream reaches the same place inside `DFRPipeline`, which additionally
    /// tiles the canvas, seams the tiles on shared keyframes and invents
    /// mid-segment keyframe slots per tile. That machinery exists to keep long
    /// clips coherent and within memory; this single-window version does not
    /// have it, so it is capped at ``maximumInterpolationFrames`` rather than
    /// silently degrading on long inputs.
    ///
    /// - Parameters:
    ///   - videoPath: the clip to densify. Its frame count must be `8n + 1`.
    ///   - prompt: still conditions the refinement — pass what the clip depicts.
    ///   - upscalerPath: the *temporal* upscaler; a spatial one is refused.
    ///   - strength: how far to renoise before refining. Higher invents more
    ///     motion and drifts further from the source; the default matches
    ///     upstream's temporal rounds.
    public func interpolateTemporally(
        videoPath: String,
        prompt: String,
        upscalerPath: String,
        width: Int,
        height: Int,
        numFrames: Int,
        seed: UInt64? = nil,
        eta: Float = 0.5,
        onProgress: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> VideoGenerationResult {
        let startTime = Date()
        guard FileManager.default.fileExists(atPath: videoPath) else {
            throw LTXError.fileNotFound("Video not found: \(videoPath)")
        }
        guard numFrames <= Self.maximumInterpolationFrames else {
            throw LTXError.invalidConfiguration(
                "Temporal interpolation is capped at \(Self.maximumInterpolationFrames) frames "
                + "in this single-window form (\(numFrames) requested); upstream tiles the canvas "
                + "for longer clips, which is not ported yet.")
        }
        if !isLoaded { try await loadModels(progressCallback: nil) }
        let beacon = RuntimeBeacon.begin(task: "temporal-interpolate", model: model.rawValue)
        defer { beacon?.end() }

        let upscaler = try loadTemporalUpscaler(from: upscalerPath)

        // Encode the source clip, then densify its latent.
        let sourceLatent = try await encodeVideo(
            path: videoPath, width: width, height: height, numFrames: numFrames)
        onProgress?(GenerationProgress(
            currentStep: 0, totalSteps: temporalSigmas.count, sigma: 1, phase: .upscaling))

        let decoderStats = vaeDecoder
        let mean = (decoderStats?.meanOfMeans ?? MLXArray.zeros([128])).reshaped([1, -1, 1, 1, 1])
        let std = (decoderStats?.stdOfMeans ?? MLXArray.ones([128])).reshaped([1, -1, 1, 1, 1])
        // The upsampler works on un-normalised latents, like the spatial one.
        var latent = (upscaler(sourceLatent * std + mean) - mean) / std
        MLX.eval(latent)
        let densifiedFrames = (latent.dim(2) - 1) * 8 + 1
        LTXDebug.log("[temporal] \(numFrames) → \(densifiedFrames) frames, latent \(latent.shape)")

        // Refine: renoise to the first sigma, then walk the schedule ancestrally.
        let encoded = try await encodeText(prompt)
        unloadGemmaIfConfigured()
        let shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: densifiedFrames, height: height, width: width)

        if let seed { MLXRandom.seed(seed) }
        let sigmas = temporalSigmas
        latent = MLXArray(sigmas[0]) * MLXRandom.normal(latent.shape).asType(latent.dtype)
            + MLXArray(1.0 - sigmas[0]) * latent
        MLX.eval(latent)

        let stepper = AncestralEulerStep(eta: eta)
        for step in 0 ..< (sigmas.count - 1) {
            let sigma = sigmas[step]
            onProgress?(GenerationProgress(
                currentStep: step, totalSteps: sigmas.count - 1, sigma: sigma, phase: .refinement))
            let velocity = runDenoiseStep(
                sigma: sigma, videoLatent: latent, audioLatentPacked: nil,
                shape: shape, videoAppendCtx: nil, audioRefCtx: nil, audioNumFrames: 0,
                videoTextEmbeddings: encoded.embeddings,
                audioTextEmbeddings: encoded.embeddings,
                textMask: encoded.mask)
            // The transformer predicts velocity; the ancestral step wants x₀.
            let denoised = latent - MLXArray(sigma) * velocity.video
            latent = stepper(
                sample: latent, denoised: denoised,
                sigma: sigma, sigmaNext: sigmas[step + 1],
                noise: MLXRandom.normal(latent.shape).asType(latent.dtype))
            MLX.eval(latent)
        }

        onProgress?(GenerationProgress(
            currentStep: sigmas.count - 1, totalSteps: sigmas.count - 1, sigma: 0, phase: .decoding))
        let frames = decodeFrames(latent: latent)
        MLX.eval(frames)

        return VideoGenerationResult(
            frames: frames,
            seed: seed ?? 0,
            generationTime: Date().timeIntervalSince(startTime),
            audioWaveform: nil, audioSampleRate: nil,
            effectivePrompt: prompt)
    }

    /// The tail of the distilled schedule, which upstream's temporal rounds use:
    /// four steps starting below the high-noise region, since the input already
    /// carries the composition.
    var temporalSigmas: [Float] { Array(DISTILLED_SIGMA_VALUES.dropFirst(4)) }

    /// Single-window interpolation holds the whole densified clip in memory at
    /// once; beyond this, upstream's tiling is required.
    static let maximumInterpolationFrames = 121
}
