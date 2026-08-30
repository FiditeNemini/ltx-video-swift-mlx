// ConvVAEEncoderParityTests.swift — element-wise parity against the reference encoder
// Copyright 2026
//
// Every retake, every i2v conditioning image, and every LipDub video
// reference goes through this encoder — until now, nothing verified it
// element-wise against Lightricks' own implementation. Sub-task 2 of #57's
// breakdown (sub-task 1 was the decoder, PR #76).
//
// Ground truth comes from Lightricks' own VideoEncoder, run on CPU float32
// over a fixed pixel input — deterministic, since encoding only extracts the
// mean of the latent distribution (no sampling) and this checkpoint has no
// attention — dumped by scripts/conv_video_encoder_reference.py.
//
// Compared at the same module boundary as the Swift port: the encoder's own
// raw (pre-normalize) means. The Swift port normalizes outside the encoder,
// reusing the VAE decoder's already-loaded per-channel statistics
// (LTXPipeline.encodeVideo) — that normalize step is exercised by
// ConvVAEDecoderParityTests, which loads and checks meanOfMeans/stdOfMeans.
//
//   TEST_RUNNER_LTX25_CONVVAE=<video-vae-conv-bf16.safetensors> \
//   TEST_RUNNER_LTX25_CONVVAE_ENCODER_REF=<conv_video_encoder_reference.safetensors> \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/ConvVAEEncoderParityTests

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

@Suite("Conv VAE encoder parity vs the reference implementation",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_CONVVAE_ENCODER_REF"] != nil))
struct ConvVAEEncoderParityTests {

    func env(_ key: String) -> String { ProcessInfo.processInfo.environment[key]! }

    /// Relative error, the scale-free measure that stays meaningful whether the
    /// tensor lives at 0.05 or 5.
    func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
        let scale = MLX.abs(b.asType(.float32)).mean().item(Float.self)
        return scale > 0 ? diff / scale : diff
    }

    func loadEncoder() throws -> VideoEncoder {
        let encoder = VideoEncoder()
        let weights = try LTXWeightLoader.loadVAEEncoderWeights(from: env("LTX25_CONVVAE"))
        try LTXWeightLoader.applyVAEEncoderWeights(weights, to: encoder)
        eval(encoder.parameters())
        return encoder
    }

    /// Pixel tensor: (B, 3, T, H, W) in [-1, 1], layout matches what
    /// scripts/conv_video_encoder_reference.py dumped as "pixels".
    @Test func outputMatchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_CONVVAE_ENCODER_REF")))
        let encoder = try loadEncoder()

        let pixels = reference["pixels"]!.asType(DType.float32)
        let refRawMeans = reference["raw_means"]!.asType(DType.float32)

        let output = encoder(pixels)
        MLX.eval(output)
        #expect(output.shape == refRawMeans.shape,
                "output \(output.shape) vs reference \(refRawMeans.shape)")
        let err = relativeError(output, refRawMeans)
        print("PARITY output (raw means) relative error: \(err)")
        #expect(err < 0.02, "conv VAE encoder diverges from the reference: \(err)")
    }

    /// Full pipeline check: normalize the encoder's raw output the way
    /// LTXPipeline.encodeVideo does — using the *decoder's* per-channel
    /// statistics — and compare against the reference's fully normalized
    /// `forward()` output.
    @Test func normalizedOutputMatchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_CONVVAE_ENCODER_REF")))
        let encoder = try loadEncoder()
        let decoder = VideoDecoder()
        let decoderWeights = try LTXWeightLoader.loadVAEWeights(from: env("LTX25_CONVVAE"))
        _ = decoder.update(parameters: ModuleParameters.unflattened(decoderWeights))
        eval(decoder.parameters())

        let pixels = reference["pixels"]!.asType(DType.float32)
        let refNormalized = reference["output"]!.asType(DType.float32)

        let raw = encoder(pixels)
        let mean = decoder.meanOfMeans.asType(DType.float32).reshaped([1, -1, 1, 1, 1])
        let std = decoder.stdOfMeans.asType(DType.float32).reshaped([1, -1, 1, 1, 1])
        let normalized = (raw.asType(DType.float32) - mean) / std
        MLX.eval(normalized)

        #expect(normalized.shape == refNormalized.shape)
        let err = relativeError(normalized, refNormalized)
        print("PARITY normalized output relative error: \(err)")
        #expect(err < 0.02, "normalized encoder output diverges from the reference: \(err)")
    }
}
