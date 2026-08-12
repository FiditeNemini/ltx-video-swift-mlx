// DurationHeadE2ETests.swift — Gated numerical check of the LTX-2.5 duration head
// Copyright 2026
//
// The head is a regression model: its output is a number nobody can eyeball. A
// mis-ported attention pooler still returns a plausible-looking duration, so the
// port is pinned against an independent NumPy implementation of the same
// forward, reading the same weights, on a deterministic synthetic input.
//
// Reference computed with (float64, weights read straight from the safetensors):
//     x    = vid @ video_input_proj.W.T + b + video_modality_emb
//     q/k/v from the packed in_proj, 4 heads of 64, softmax(qk/sqrt(64)) @ v
//     pooled -> out_proj -> mlp_hidden -> gelu(tanh) -> mlp_out -> exp
//   => log_duration = 2.433859, seconds = 11.402804
//
// Gated behind LTX25_MODELS_DIR.
//
// Run:
//   TEST_RUNNER_LTX25_MODELS_DIR=/Volumes/Lexar/models/ltx-2.5 \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/DurationHeadE2ETests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("LTX-2.5 duration head (gated: LTX25_MODELS_DIR)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] != nil),
       .serialized)
struct DurationHeadE2ETests {

    static var headURL: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] ?? "")
            .appendingPathComponent("ltx-2.5-duration-head-bf16.safetensors")
    }

    /// The same deterministic input the NumPy reference used: `sin(i/10 + j/100)`.
    static func syntheticTokens(tokens: Int = 8, dim: Int = 4096) -> MLXArray {
        let rows = MLXArray(0 ..< tokens).asType(.float32).reshaped([tokens, 1]) * 0.1
        let cols = MLXArray(0 ..< dim).asType(.float32).reshaped([1, dim]) * 0.01
        return MLX.sin(rows + cols).expandedDimensions(axis: 0)
    }

    @Test func matchesTheNumPyReference() throws {
        let head = try LTXDurationHead.load(from: Self.headURL)
        let seconds = try head.predictSeconds(
            videoTokens: Self.syntheticTokens(), audioTokens: nil)

        // bf16 weights through a 4096-wide projection: ~1% is the precision floor,
        // and far tighter than any plausible mis-port (a transposed packed
        // projection or a wrong head split moves this by tens of percent).
        #expect(abs(seconds - 11.402804) / 11.402804 < 0.01,
                "predicted \(seconds)s against the NumPy reference 11.402804s")
    }

    @Test func snapsFramesToTheGridAndReportsClamping() throws {
        let head = try LTXDurationHead.load(from: Self.headURL)
        let tokens = Self.syntheticTokens()

        let normal = try head.predictFrameCount(
            videoTokens: tokens, audioTokens: nil, frameRate: 24.0)
        #expect((normal.frames - 1) % 8 == 0)
        #expect(normal.wasClamped == false)
        #expect(normal.frames == 273)   // 11.4028 s x 24 = 273.7 -> 273 on the grid

        // A ceiling below the prediction must clamp, and say so.
        let clamped = try head.predictFrameCount(
            videoTokens: tokens, audioTokens: nil, frameRate: 24.0,
            minSeconds: 1.0, maxSeconds: 5.0)
        #expect(clamped.wasClamped)
        #expect(clamped.frames <= 121)
        #expect((clamped.frames - 1) % 8 == 0)
    }
}
