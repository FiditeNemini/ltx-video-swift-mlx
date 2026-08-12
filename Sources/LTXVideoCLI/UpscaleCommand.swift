// UpscaleCommand.swift - Generative spatial upscaling through the pixel IC-LoRA
// Copyright 2025

import ArgumentParser
import Foundation
import LTXVideo

/// Re-render a clip at higher resolution with LTX's pixel spatial upscaler.
///
/// Distinct from the latent upscaler the two-stage `generate` already uses
/// between its stages: that one refines inside the diffusion loop, this one
/// takes a finished low-resolution clip and re-renders it, inventing detail that
/// was never in the source. The scale factor comes from the adapter, not a flag.
struct Upscale: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Upscale a video with the pixel spatial upscaler IC-LoRA (generative, x2)"
    )

    @Argument(help: "Prompt describing the scene — the upscaler still conditions on text")
    var prompt: String

    @Option(name: .shortAndLong, help: "Low-resolution source video")
    var input: String

    @Option(name: .shortAndLong, help: "Output file path")
    var output: String = "upscaled.mp4"

    @Option(name: .shortAndLong, help: "Output width (must be divisible by 64 and by the adapter's factor)")
    var width: Int = 1536

    @Option(name: .shortAndLong, help: "Output height (must be divisible by 64 and by the adapter's factor)")
    var height: Int = 1024

    @Option(name: .shortAndLong, help: "Number of frames (8n+1); must match the source clip")
    var frames: Int = 121

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .long, help: "Path to the upscaler IC-LoRA (downloaded from the catalog if omitted)")
    var lora: String?

    @Option(name: .long, help: "LoRA scale — the weights ship pre-scaled for 1.0")
    var loraScale: Float = 1.0

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4")
    var transformerQuant: String = "bf16"

    @Option(name: .long, help: "Model variant (the x2 pixel upscaler is published for LTX-2.5)")
    var model: String = "2.5-distilled"

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Custom directory for model storage")
    var modelsDir: String?

    @Option(name: .long, help: "Video bitrate in kbps")
    var bitrate: Int?

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    mutating func run() async throws {
        if let dir = modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }
        if debug { LTXDebug.enableDebugMode() }

        let variant = try parseModelVariant(model)
        guard let quantization = TransformerQuantization(rawValue: transformerQuant) else {
            throw ValidationError("Invalid quantization: \(transformerQuant)")
        }
        guard FileManager.default.fileExists(atPath: input) else {
            throw ValidationError("Source video not found: \(input)")
        }

        print("\(variant.displayName) — Pixel Spatial Upscaler")
        print("=========================================")
        print("Source: \(input)")
        print("Output: \(width)x\(height), \(frames) frames")
        print()

        let pipeline = LTXPipeline(
            model: variant,
            quantization: LTXQuantizationConfig(
                transformer: quantization, textEncoder: quantization),
            hfToken: hfToken)

        print("Loading models (this may take a while)...")
        try await pipeline.loadModels { progress in
            print("  \(progress.message) (\(Int(progress.progress * 100))%)")
        }

        let adapterPath: String
        if let lora {
            adapterPath = lora
        } else {
            print("Fetching the upscaler adapter...")
            let aux: LTXAuxiliaryModel = variant.family == .ltx25
                ? .pixelSpatialUpscalerX2_25 : .spatialUpscalerX2_23
            let downloader = ModelDownloader(hfToken: hfToken)
            adapterPath = try await downloader.downloadAuxiliaryModel(aux) { progress in
                print("  \(progress.message)")
            }.path
        }
        print("Adapter: \((adapterPath as NSString).lastPathComponent)")

        let config = LTXVideoGenerationConfig(
            width: width, height: height, numFrames: frames, numSteps: 8, seed: seed)

        let start = Date()
        let result = try await pipeline.generateWithVideoReference(
            prompt: prompt,
            referenceVideoPath: input,
            loraPath: adapterPath,
            config: config,
            loraScale: loraScale
        ) { progress in
            print("  Step \(progress.currentStep + 1)/\(progress.totalSteps) [\(progress.phase)]")
            fflush(stdout)
        }
        print("Upscaled in \(String(format: "%.1f", Date().timeIntervalSince(start)))s")

        var exportConfig = VideoExportConfig.default
        if let bitrate { exportConfig.averageBitRate = bitrate * 1000 }
        _ = try await VideoExporter.exportVideo(
            frames: result.frames, width: width, height: height, fps: 24,
            config: exportConfig, to: URL(fileURLWithPath: output))
        print("Saved to: \(output)")
    }
}
