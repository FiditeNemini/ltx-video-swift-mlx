//
//  LTXConfigTests.swift
//  ltx-video-swift-mlx
//

import Testing
import Foundation
@testable import LTXVideo

// MARK: - LTXModel Tests

@Suite("LTXModel")
struct LTXModelTests {
    @Test func testDistilledProperties() {
        let model = LTXModel.distilled
        #expect(model.rawValue == "distilled")
        #expect(model.displayName == "LTX-2.3 Distilled (~46GB)")
        #expect(model.defaultSteps == 8)
        #expect(model.estimatedVRAM == 46)
        #expect(model.huggingFaceRepo == "Lightricks/LTX-2.3")
        #expect(model.unifiedWeightsFilename == "ltx-2.3-22b-distilled.safetensors")
    }

    @Test func testTransformerConfig() {
        let config = LTXModel.distilled.transformerConfig
        #expect(config.numLayers == 48)
        #expect(config.numAttentionHeads == 32)
        #expect(config.attentionHeadDim == 128)
        #expect(config.innerDim == 4096)
        #expect(config.gatedAttention == true)
        #expect(config.crossAttentionAdaLN == true)
        #expect(config.captionProjBeforeConnector == true)
    }

    @Test func testCaseIterable() {
        let allCases = LTXModel.allCases
        #expect(allCases.count == 1)
        #expect(allCases.contains(.distilled))
    }
}

// MARK: - LTXTransformerConfig Tests

@Suite("LTXTransformerConfig")
struct LTXTransformerConfigTests {
    @Test func testDefaultConfig() {
        let config = LTXTransformerConfig.default
        #expect(config.numLayers == 48)
        #expect(config.gatedAttention == false)
        #expect(config.crossAttentionAdaLN == false)
        #expect(config.captionProjBeforeConnector == false)
        #expect(config.captionChannels == 3840)
    }

    @Test func testLTX23Config() {
        let config = LTXTransformerConfig.ltx23
        #expect(config.numLayers == 48)
        #expect(config.gatedAttention == true)
        #expect(config.crossAttentionAdaLN == true)
        #expect(config.captionProjBeforeConnector == true)
        #expect(config.captionChannels == 4096)
    }

    @Test func testInnerDim() {
        let config = LTXTransformerConfig(numAttentionHeads: 16, attentionHeadDim: 64)
        #expect(config.innerDim == 1024)
    }

    @Test func testAudioDimensions() {
        let config = LTXTransformerConfig.ltx23
        #expect(config.audioNumAttentionHeads == 32)
        #expect(config.audioAttentionHeadDim == 64)
        #expect(config.audioInnerDim == 2048)
        #expect(config.audioCrossAttentionDim == 2048)
    }

    @Test func testDescription() {
        let config = LTXTransformerConfig.ltx23
        let desc = config.description
        #expect(desc.contains("layers: 48"))
        #expect(desc.contains("heads: 32"))
    }

    @Test func testRoPEConfig() {
        let config = LTXTransformerConfig.ltx23
        #expect(config.ropeTheta == 10000.0)
        #expect(config.maxPos == [20, 2048, 2048])
        #expect(config.timestepScaleMultiplier == 1000)
    }
}

// MARK: - LTXVideoGenerationConfig Tests

@Suite("LTXVideoGenerationConfig")
struct LTXVideoGenerationConfigTests {
    @Test func testDefaultValues() {
        let config = LTXVideoGenerationConfig()
        #expect(config.width == 704)
        #expect(config.height == 480)
        #expect(config.numFrames == 121)
        #expect(config.numSteps == 8)
        #expect(config.seed == nil)
        #expect(config.enhancePrompt == false)
        #expect(config.imagePath == nil)
        #expect(config.imageCondNoiseScale == 0.0)
        #expect(config.videoPath == nil)
        #expect(config.retakeStrength == 0.8)
        #expect(config.retakeStartTime == nil)
        #expect(config.retakeEndTime == nil)
    }

    @Test func testModelConvenienceInit() {
        let config = LTXVideoGenerationConfig(model: .distilled)
        #expect(config.numSteps == 8)
    }

    @Test func testModelConvenienceInitOverride() {
        let config = LTXVideoGenerationConfig(model: .distilled, numSteps: 4)
        #expect(config.numSteps == 4)
    }

    @Test func testLatentDimensions() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 121)
        #expect(config.latentWidth == 24)   // 768 / 32
        #expect(config.latentHeight == 16)  // 512 / 32
        #expect(config.latentFrames == 16)  // (121 - 1) / 8 + 1
        #expect(config.numLatentTokens == 24 * 16 * 16)
    }

    @Test func testLatentDimensions1024x576() {
        let config = LTXVideoGenerationConfig(width: 1024, height: 576, numFrames: 241)
        #expect(config.latentWidth == 32)   // 1024 / 32
        #expect(config.latentHeight == 18)  // 576 / 32
        #expect(config.latentFrames == 31)  // (241 - 1) / 8 + 1
    }

    // MARK: - Validation Tests

    @Test func testValidConfig() throws {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 121)
        try config.validate()
    }

    @Test func testInvalidWidthNotDivisibleBy64() {
        let config = LTXVideoGenerationConfig(width: 700, height: 512)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testInvalidHeightNotDivisibleBy64() {
        let config = LTXVideoGenerationConfig(width: 768, height: 500)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testInvalidFrameCount() {
        let config = LTXVideoGenerationConfig(numFrames: 10)  // not 8n+1
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testValidFrameCounts() throws {
        for n in [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 241] {
            let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: n)
            try config.validate()
        }
    }

    @Test func testWidthTooSmall() {
        let config = LTXVideoGenerationConfig(width: 32, height: 512)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testWidthTooLarge() {
        let config = LTXVideoGenerationConfig(width: 4096, height: 512)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testFramesTooFew() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 1)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testStepsTooFew() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numSteps: 0)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testStepsTooMany() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numSteps: 200)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testMissingImagePath() {
        let config = LTXVideoGenerationConfig(
            width: 768, height: 512, imagePath: "/nonexistent/image.png"
        )
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testMissingVideoPath() {
        let config = LTXVideoGenerationConfig(
            width: 768, height: 512, videoPath: "/nonexistent/video.mp4"
        )
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testInvalidRetakeStrengthZero() {
        // Create a temporary file for valid videoPath
        let tmpPath = "/tmp/ltx_test_video.mp4"
        FileManager.default.createFile(atPath: tmpPath, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let config = LTXVideoGenerationConfig(
            width: 768, height: 512,
            videoPath: tmpPath,
            retakeStrength: 0.0
        )
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testRetakeStrengthBoundaries() throws {
        let tmpPath = "/tmp/ltx_test_video2.mp4"
        FileManager.default.createFile(atPath: tmpPath, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        // strength = 1.0 should be valid
        let config1 = LTXVideoGenerationConfig(
            width: 768, height: 512,
            videoPath: tmpPath,
            retakeStrength: 1.0
        )
        try config1.validate()

        // strength = 0.5 should be valid
        let config2 = LTXVideoGenerationConfig(
            width: 768, height: 512,
            videoPath: tmpPath,
            retakeStrength: 0.5
        )
        try config2.validate()
    }

    @Test func testRetakeFieldsSet() {
        let config = LTXVideoGenerationConfig(
            videoPath: "/tmp/vid.mp4",
            retakeStrength: 0.7,
            retakeStartTime: 2.0,
            retakeEndTime: 5.0
        )
        #expect(config.videoPath == "/tmp/vid.mp4")
        #expect(config.retakeStrength == 0.7)
        #expect(config.retakeStartTime == 2.0)
        #expect(config.retakeEndTime == 5.0)
    }
}
