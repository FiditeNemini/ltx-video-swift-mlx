//
//  VideoLatentShapeTests.swift
//  ltx-video-swift-mlx
//

import Testing
@testable import LTXVideo

// MARK: - SpatioTemporalScaleFactors Tests

@Suite("SpatioTemporalScaleFactors")
struct SpatioTemporalScaleFactorsTests {
    @Test func testDefaultFactors() {
        let sf = SpatioTemporalScaleFactors.default
        #expect(sf.time == 8)
        #expect(sf.height == 32)
        #expect(sf.width == 32)
    }

    @Test func testPixelToLatent() {
        let sf = SpatioTemporalScaleFactors.default
        // 121 frames → (121 - 1) / 8 + 1 = 16 latent frames
        let result = sf.pixelToLatent(frames: 121, height: 512, width: 768)
        #expect(result.frames == 16)
        #expect(result.height == 16)   // 512 / 32
        #expect(result.width == 24)    // 768 / 32
    }

    @Test func testPixelToLatent241Frames() {
        let sf = SpatioTemporalScaleFactors.default
        // 241 frames → (241 - 1) / 8 + 1 = 31 latent frames
        let result = sf.pixelToLatent(frames: 241, height: 576, width: 1024)
        #expect(result.frames == 31)
        #expect(result.height == 18)   // 576 / 32
        #expect(result.width == 32)    // 1024 / 32
    }

    @Test func testLatentToPixel() {
        let sf = SpatioTemporalScaleFactors.default
        // Reverse: 16 latent frames → (16 - 1) * 8 + 1 = 121 pixel frames
        let result = sf.latentToPixel(frames: 16, height: 16, width: 24)
        #expect(result.frames == 121)
        #expect(result.height == 512)
        #expect(result.width == 768)
    }

    @Test func testRoundTrip() {
        let sf = SpatioTemporalScaleFactors.default
        let pixelFrames = 121
        let pixelH = 512
        let pixelW = 768

        let latent = sf.pixelToLatent(frames: pixelFrames, height: pixelH, width: pixelW)
        let pixel = sf.latentToPixel(frames: latent.frames, height: latent.height, width: latent.width)

        #expect(pixel.frames == pixelFrames)
        #expect(pixel.height == pixelH)
        #expect(pixel.width == pixelW)
    }

    @Test func testMinimalFrames() {
        let sf = SpatioTemporalScaleFactors.default
        // 9 frames → (9 - 1) / 8 + 1 = 2 latent frames
        let result = sf.pixelToLatent(frames: 9, height: 64, width: 64)
        #expect(result.frames == 2)
    }
}

// MARK: - VideoLatentShape Tests

@Suite("VideoLatentShape")
struct VideoLatentShapeTests {
    @Test func testBasicShape() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 16, height: 16, width: 24)
        #expect(shape.batch == 1)
        #expect(shape.channels == 128)
        #expect(shape.frames == 16)
        #expect(shape.height == 16)
        #expect(shape.width == 24)
    }

    @Test func testTokenCount() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 16, height: 16, width: 24)
        #expect(shape.tokenCount == 16 * 16 * 24)  // 6144
    }

    @Test func testShapeArray() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 16, height: 16, width: 24)
        #expect(shape.shape == [1, 128, 16, 16, 24])
    }

    @Test func testPatchifiedShape() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 16, height: 16, width: 24)
        #expect(shape.patchifiedShape == [1, 6144, 128])
    }

    @Test func testFromPixelDimensions() {
        let shape = VideoLatentShape.fromPixelDimensions(
            frames: 121, height: 512, width: 768
        )
        #expect(shape.batch == 1)
        #expect(shape.channels == 128)
        #expect(shape.frames == 16)
        #expect(shape.height == 16)
        #expect(shape.width == 24)
    }

    @Test func testPixelDimensionsRoundTrip() {
        let shape = VideoLatentShape.fromPixelDimensions(
            frames: 241, height: 576, width: 1024
        )
        let pixel = shape.pixelDimensions()
        #expect(pixel.frames == 241)
        #expect(pixel.height == 576)
        #expect(pixel.width == 1024)
    }

    @Test func testValidShape() throws {
        let shape = VideoLatentShape.fromPixelDimensions(
            frames: 121, height: 512, width: 768
        )
        try shape.validate()
    }

    @Test func testInvalidChannels() {
        let shape = VideoLatentShape(batch: 1, channels: 64, frames: 16, height: 16, width: 24)
        #expect(throws: LTXError.self) { try shape.validate() }
    }

    // MARK: - Convenience Presets

    @Test func testStandard480p() {
        let shape = VideoLatentShape.standard480p
        let pixel = shape.pixelDimensions()
        #expect(pixel.frames == 121)
        #expect(pixel.height == 480)
        #expect(pixel.width == 704)
    }

    @Test func testStandard512() {
        let shape = VideoLatentShape.standard512
        let pixel = shape.pixelDimensions()
        #expect(pixel.frames == 25)
        #expect(pixel.height == 512)
        #expect(pixel.width == 512)
    }

    @Test func testLandscape768x512() {
        let shape = VideoLatentShape.landscape768x512
        let pixel = shape.pixelDimensions()
        #expect(pixel.frames == 49)
        #expect(pixel.height == 512)
        #expect(pixel.width == 768)
    }
}
