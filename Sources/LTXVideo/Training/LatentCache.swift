// LatentCache.swift - Pre-computed latent cache for efficient training
// Copyright 2026

import Foundation
@preconcurrency import MLX
import MLXNN

/// Pre-computed latent cache for training.
///
/// Encodes all training videos/audio through VAE and text through Gemma+connector once,
/// then stores on disk as safetensors for fast loading during training.
/// This allows VAE and Gemma to be unloaded, saving ~30GB+ RAM.
struct CachedSample: Sendable {
    /// Video latent (1, T_video, C) — patchified and packed
    let videoLatent: MLXArray
    /// Video latent shape info
    let latentFrames: Int
    let latentHeight: Int
    let latentWidth: Int
    /// Text embeddings for video (1, S, D)
    let promptEmbeddings: MLXArray
    /// Attention mask (1, S)
    let promptMask: MLXArray
    /// Audio latent (1, T_audio, C_audio) — optional
    let audioLatent: MLXArray?
    /// Audio text embeddings (1, S, D) — optional
    let audioEmbeddings: MLXArray?
    /// Audio attention mask (1, S) — optional
    let audioMask: MLXArray?
    /// Audio latent frames count — optional
    let audioNumFrames: Int?
    /// Source filename
    let filename: String
}

/// Manages the latent cache directory
class LatentCache {
    let cacheDir: String
    let includeAudio: Bool
    private var cachedSamples: [CachedSample] = []

    init(cacheDir: String, includeAudio: Bool = false) {
        self.cacheDir = cacheDir
        self.includeAudio = includeAudio
    }

    /// Build cache from dataset using the pipeline's encoder components.
    ///
    /// - Parameters:
    ///   - dataset: Video dataset to encode
    ///   - pipeline: Pipeline with loaded models (VAE encoder, Gemma, text encoder)
    ///   - progressCallback: Called with (currentIndex, total, filename)
    func build(
        from dataset: VideoDataset,
        pipeline: LTXPipeline,
        progressCallback: ((Int, Int, String) -> Void)? = nil
    ) async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: cacheDir) {
            try fm.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
        }

        for i in 0..<dataset.count {
            let sample = try await dataset.loadSample(at: i)
            progressCallback?(i, dataset.count, sample.filename)

            let cachePath = samplePath(for: sample.filename)

            // Skip if already cached
            if fm.fileExists(atPath: cachePath) {
                LTXDebug.log("Cache hit: \(sample.filename)")
                continue
            }

            // Encode video frames through VAE encoder
            let videoLatent = try await pipeline.encodeVideoLatents(frames: sample.frames)
            eval(videoLatent)

            // Encode text through Gemma + connector
            let textResult = try await pipeline.encodeText(sample.caption)
            eval(textResult.embeddings, textResult.mask)

            // Build save dictionary
            var saveDict: [String: MLXArray] = [
                "video_latent": videoLatent,
                "prompt_embeddings": textResult.embeddings,
                "prompt_mask": textResult.mask,
            ]

            // Encode audio if needed
            if includeAudio, let audioWaveform = sample.audioWaveform {
                let audioLatent = try await pipeline.encodeAudioLatents(waveform: audioWaveform)
                eval(audioLatent)

                // Get audio text embeddings (uses same Gemma + audio connector)
                let audioTextResult = try await pipeline.encodeAudioText(prompt: sample.caption)
                eval(audioTextResult.embeddings, audioTextResult.mask)

                saveDict["audio_latent"] = audioLatent
                saveDict["audio_embeddings"] = audioTextResult.embeddings
                saveDict["audio_mask"] = audioTextResult.mask
            }

            // Save to disk
            let url = URL(fileURLWithPath: cachePath)
            try MLX.save(arrays: saveDict, url: url)

            LTXDebug.log("Cached: \(sample.filename) → \(cachePath)")

            // Clear GPU cache after each encoding
            Memory.clearCache()
        }
    }

    /// Load all cached samples into memory
    func loadAll() throws {
        cachedSamples = []

        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: cacheDir)
        let safetensorsFiles = contents.filter { $0.hasSuffix(".safetensors") }
            .sorted()

        for file in safetensorsFiles {
            let path = (cacheDir as NSString).appendingPathComponent(file)
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))

            guard let videoLatent = arrays["video_latent"],
                  let promptEmbeddings = arrays["prompt_embeddings"],
                  let promptMask = arrays["prompt_mask"]
            else {
                print("Warning: Incomplete cache file \(file), skipping")
                continue
            }

            // Infer latent shape from video latent
            // Video latent is (1, T*H*W, C) — we need the original spatial dims
            // Store them as metadata in a separate approach, or infer from config
            let sample = CachedSample(
                videoLatent: videoLatent,
                latentFrames: 0,  // Will be set from config
                latentHeight: 0,
                latentWidth: 0,
                promptEmbeddings: promptEmbeddings,
                promptMask: promptMask,
                audioLatent: arrays["audio_latent"],
                audioEmbeddings: arrays["audio_embeddings"],
                audioMask: arrays["audio_mask"],
                audioNumFrames: arrays["audio_latent"].map { $0.dim(1) },
                filename: (file as NSString).deletingPathExtension
            )
            cachedSamples.append(sample)
        }

        LTXDebug.log("Loaded \(cachedSamples.count) cached samples")
    }

    /// Get a random cached sample
    func randomSample() -> CachedSample? {
        guard !cachedSamples.isEmpty else { return nil }
        let idx = Int.random(in: 0..<cachedSamples.count)
        return cachedSamples[idx]
    }

    /// Get sample at index (wraps around)
    func sample(at index: Int) -> CachedSample? {
        guard !cachedSamples.isEmpty else { return nil }
        return cachedSamples[index % cachedSamples.count]
    }

    /// Number of cached samples
    var count: Int { cachedSamples.count }

    /// Whether the cache is populated
    var isEmpty: Bool { cachedSamples.isEmpty }

    // MARK: - Private

    private func samplePath(for filename: String) -> String {
        let baseName = (filename as NSString).deletingPathExtension
        return (cacheDir as NSString).appendingPathComponent(baseName + ".safetensors")
    }
}
