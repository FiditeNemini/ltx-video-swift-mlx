// VAERoundTripE2ETests.swift — Gated VAE encode→decode round trip
// Copyright 2026
//
// The reference an IC-LoRA consumes is a VAE *encoding* of a source clip. If that
// encoding is geometrically wrong — shifted, scaled, transposed — every
// downstream conclusion about the adapter is worthless, and the symptom (a
// re-render whose framing drifts) looks exactly like an adapter problem.
//
// So: encode a clip and decode it straight back. Any geometry error shows up
// here, with no diffusion in the way.
//
// Gated behind LTX25_MODELS_DIR.
//
// Run:
//   TEST_RUNNER_LTX25_MODELS_DIR=/Volumes/Lexar/models/ltx-2.5 \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/VAERoundTripE2ETests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("VAE round trip (gated: LTX25_MODELS_DIR)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] != nil),
       .serialized)
struct VAERoundTripE2ETests {

    static var vaeURL: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] ?? "")
            .appendingPathComponent("ltx-2.5-video-vae-conv-bf16.safetensors")
    }

    /// A deterministic 384x256 pattern: a bright square off-centre, so a shift or
    /// a flip is visible as a moved centroid rather than as a diffuse difference.
    static func testPattern(frames: Int = 9, height: Int = 256, width: Int = 384) -> MLXArray {
        var image = MLXArray.zeros([1, 3, frames, height, width], dtype: .float32)
        // Square spanning rows 40..<104 and columns 60..<124 — clearly off-centre.
        image[0..., 0..., 0..., 40 ..< 104, 60 ..< 124] = MLXArray(1.0)
        return image * MLXArray(2.0) - MLXArray(1.0)   // to [-1, 1]
    }

    /// Centre of mass of the bright region, as (row, column).
    static func centroid(_ frame: MLXArray) -> (row: Float, column: Float) {
        let gray = frame.mean(axis: 0)                    // (H, W)
        let mass = MLX.maximum(gray, MLXArray(Float(0)))
        let total = mass.sum().item(Float.self) + 1e-6
        let rows = MLXArray(0 ..< gray.dim(0)).asType(.float32).reshaped([gray.dim(0), 1])
        let cols = MLXArray(0 ..< gray.dim(1)).asType(.float32).reshaped([1, gray.dim(1)])
        return ((mass * rows).sum().item(Float.self) / total,
                (mass * cols).sum().item(Float.self) / total)
    }

    @Test func encodeThenDecodePreservesGeometry() throws {
        let decoderWeights = try LTXWeightLoader.loadVAEWeights(from: Self.vaeURL.path)
        let decoder = VideoDecoder()
        try LTXWeightLoader.applyVAEWeights(decoderWeights, to: decoder)

        let encoderWeights = try LTXWeightLoader.loadVAEEncoderWeights(from: Self.vaeURL.path)
        let encoder = VideoEncoder()
        try LTXWeightLoader.applyVAEEncoderWeights(encoderWeights, to: encoder)
        eval(encoder.parameters(), decoder.parameters())

        let source = Self.testPattern()
        let latent = encoder(source)
        MLX.eval(latent)

        // Latent geometry: 32x spatial, 8x temporal (+1 for the causal first frame).
        #expect(latent.dim(3) == 256 / 32)
        #expect(latent.dim(4) == 384 / 32)
        #expect(latent.dim(2) == (9 - 1) / 8 + 1)

        // Normalise exactly as the pipeline does before handing latents to the model,
        // then invert, so the round trip exercises the statistics too.
        let mean = decoder.meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let std = decoder.stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let normalised = (latent.asType(.float32) - mean) / std
        let restored = normalised * std + mean

        // decodeVideo returns (F, H, W, C) in [0, 1] — a different layout from the
        // (B, C, F, H, W) it consumes.
        let decoded = decodeVideo(latent: restored, decoder: decoder, timestep: nil)
        MLX.eval(decoded)

        #expect(decoded.dim(1) == 256, "height")
        #expect(decoded.dim(2) == 384, "width")
        #expect(decoded.dim(3) == 3, "channels last")

        // The square must come back where it went in. A half-latent-cell error is
        // 16 px here — the tolerance is deliberately tighter than that.
        // Source is (B, C, F, H, W); decoded is (F, H, W, C). Both reduced to (C, H, W).
        let sourceCentroid = Self.centroid(source[0, 0..., 4, 0..., 0...])
        let decodedCentroid = Self.centroid(
            decoded[4, 0..., 0..., 0...].transposed(2, 0, 1) * MLXArray(2.0) - MLXArray(1.0))
        #expect(abs(sourceCentroid.row - decodedCentroid.row) < 6,
                "row \(sourceCentroid.row) -> \(decodedCentroid.row)")
        #expect(abs(sourceCentroid.column - decodedCentroid.column) < 6,
                "column \(sourceCentroid.column) -> \(decodedCentroid.column)")
    }
}
