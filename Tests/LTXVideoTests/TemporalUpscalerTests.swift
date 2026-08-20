// TemporalUpscalerTests.swift — the temporal upsampler against the reference
// Copyright 2026
//
//   TEST_RUNNER_LTX25_TEMPORAL=<latent-temporal-upscaler...safetensors> \
//   TEST_RUNNER_LTX25_TEMPORAL_REF=<temporal_ref.safetensors> \
//   xcodebuild ... -only-testing:LTXVideoTests/TemporalUpscalerTests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Latent temporal upscaler",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_TEMPORAL_REF"] != nil))
struct TemporalUpscalerTests {

    func env(_ key: String) -> String { ProcessInfo.processInfo.environment[key]! }

    @Test func matchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_TEMPORAL_REF")))
        let upscaler = try loadTemporalUpscaler(from: env("LTX25_TEMPORAL"))

        let latent = reference["latent"]!.asType(DType.float32)
        let expected = reference["output"]!.asType(DType.float32)

        let output = upscaler(latent)
        MLX.eval(output)

        // 3 latent frames → 5: the shuffle doubles to 6 and the duplicate
        // leading frame is dropped, because latent frame 0 stands for one pixel
        // frame rather than a group of eight.
        #expect(output.shape == expected.shape, "\(output.shape) vs \(expected.shape)")
        #expect(output.dim(2) == latent.dim(2) * 2 - 1)

        let diff = MLX.abs(output - expected).mean().item(Float.self)
        let scale = MLX.abs(expected).mean().item(Float.self)
        let relative = diff / scale
        print("TEMPORAL relative error: \(relative)")
        #expect(relative < 0.002, "diverges from the reference: \(relative)")
    }
}
