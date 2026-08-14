// Gemma4EnhancerProbeTests.swift — Gated load check for the LTX-2.5 prompt
// enhancer (Gemma 4 E2B-it through Gemma4Swift).
//
// Regression for gemma-4-swift-mlx#37: the weight sanitizer's KV-shared drop
// used to also strip the vision tower's layer-15 K/V, so the multimodal load
// failed with keyNotFound. Loading is the whole test.
//
// Run:
//   TEST_RUNNER_LTX25_ENHANCER_DIR=/Volumes/Lexar/models/ltx-2.5-cache/enhancer-gemma4-e2b \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/Gemma4EnhancerProbeTests

import Foundation
import Testing
@preconcurrency import MLX
import Gemma4Swift
@testable import LTXVideo

@Suite("Gemma4 E2B enhancer load (gated: LTX25_ENHANCER_DIR)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_ENHANCER_DIR"] != nil))
struct Gemma4EnhancerProbeTests {

    @Test func loadMultimodalE2B() async throws {
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_ENHANCER_DIR"]!)
        let pipeline = await Gemma4Pipeline()
        try await pipeline.load(from: dir, multimodal: true)
        await pipeline.unload()
    }
}
