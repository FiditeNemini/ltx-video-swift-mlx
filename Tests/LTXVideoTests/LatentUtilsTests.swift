//
//  LatentUtilsTests.swift
//  ltx-video-swift-mlx
//
//  Tests for latent utility functions.
//  NOTE: Most tests require Metal (MLXArray operations).

import Testing
import MLX
import MLXRandom
@testable import LTXVideo

// MARK: - Pure Logic Tests (no Metal needed)

@Suite("LatentUtils Logic")
struct LatentUtilsLogicTests {
    @Test func testAdjustDimensionsValid() {
        // Already valid: 121 is 8n+1, 512 and 768 are multiples of 32
        let (f, h, w) = adjustDimensions(frames: 121, height: 512, width: 768)
        #expect(f == 121)
        #expect(h == 512)
        #expect(w == 768)
    }

    @Test func testAdjustDimensionsFrames() {
        // 120 → nearest valid: 121 (120-1=119, 119%8=7 ≥ 4, so round up)
        let (f, _, _) = adjustDimensions(frames: 120, height: 512, width: 768)
        #expect((f - 1) % 8 == 0)  // Must be 8n+1
    }

    @Test func testAdjustDimensionsFrames10() {
        let (f, _, _) = adjustDimensions(frames: 10, height: 512, width: 768)
        // 10 → 10-1=9, 9%8=1 < 4, so round down → 10-1=9
        #expect(f == 9)
        #expect((f - 1) % 8 == 0)
    }

    @Test func testAdjustDimensionsFrames14() {
        let (f, _, _) = adjustDimensions(frames: 14, height: 512, width: 768)
        // 14-1=13, 13%8=5 ≥ 4, round up → 14+(8-5)=17
        #expect(f == 17)
        #expect((f - 1) % 8 == 0)
    }

    @Test func testAdjustDimensionsHeight() {
        // 500 → rounds to nearest multiple of 32
        let (_, h, _) = adjustDimensions(frames: 121, height: 500, width: 768)
        #expect(h % 32 == 0)
        #expect(h >= 500) // rounds up
    }

    @Test func testAdjustDimensionsWidth() {
        let (_, _, w) = adjustDimensions(frames: 121, height: 512, width: 700)
        #expect(w % 32 == 0)
    }

    @Test func testFormatBytesGB() {
        #expect(formatBytes(2_147_483_648) == "2.0 GB")    // 2 GB
        #expect(formatBytes(53_687_091_200) == "50.0 GB")   // 50 GB
    }

    @Test func testFormatBytesMB() {
        #expect(formatBytes(524_288_000) == "500.0 MB")  // 500 MB
        #expect(formatBytes(104_857_600) == "100.0 MB")  // 100 MB
    }

    @Test func testTokenCount() {
        // 121 frames at 512x768
        let count = tokenCount(frames: 121, height: 512, width: 768)
        // Latent: 16 * 16 * 24 = 6144
        #expect(count == 6144)
    }

    @Test func testTokenCount241() {
        // 241 frames at 576x1024
        let count = tokenCount(frames: 241, height: 576, width: 1024)
        // Latent: 31 * 18 * 32 = 17856
        #expect(count == 17856)
    }
}

// MARK: - MLX Tests (require Metal)

@Suite("LatentUtils MLX Operations")
struct LatentUtilsMLXTests {
    @Test func testPatchifyUnpatchifyRoundTrip() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 2, height: 4, width: 4)
        let latent = MLXRandom.normal(shape.shape)
        eval(latent)

        let patchified = patchify(latent)
        #expect(patchified.shape == [1, 2 * 4 * 4, 128])  // [1, 32, 128]

        let unpatchified = unpatchify(patchified, shape: shape)
        #expect(unpatchified.shape == [1, 128, 2, 4, 4])

        // Values should match after round trip
        eval(unpatchified)
        let diff = (latent - unpatchified).abs().max().item(Float.self)
        #expect(diff < 1e-6)
    }

    @Test func testPatchifyShape() {
        let latent = MLXArray.zeros([1, 128, 4, 8, 6])  // B, C, F, H, W
        let patchified = patchify(latent)
        // T = 4 * 8 * 6 = 192
        #expect(patchified.shape == [1, 192, 128])
    }

    @Test func testUnpatchifyShape() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 4, height: 8, width: 6)
        let x = MLXArray.zeros([1, 192, 128])
        let unpatchified = unpatchify(x, shape: shape)
        #expect(unpatchified.shape == [1, 128, 4, 8, 6])
    }

    @Test func testGenerateNoise() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 2, height: 4, width: 4)
        let noise = generateNoise(shape: shape, seed: 42)
        eval(noise)
        #expect(noise.shape == [1, 128, 2, 4, 4])
        #expect(noise.dtype == .float32)
        // Check it's not all zeros (it's random)
        let maxVal = noise.abs().max().item(Float.self)
        #expect(maxVal > 0.1)
    }

    @Test func testGenerateNoiseStatistics() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 2, height: 4, width: 4)
        let noise = generateNoise(shape: shape, seed: 42)
        eval(noise)
        // Standard normal: mean ≈ 0, std ≈ 1
        let mean = noise.mean().item(Float.self)
        let std = noise.variance().sqrt().item(Float.self)
        #expect(abs(mean) < 0.1)
        #expect(std > 0.8 && std < 1.2)
    }

    @Test func testGenerateScaledNoise() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 2, height: 4, width: 4)
        let scaled = generateScaledNoise(shape: shape, sigma: 0.5, seed: 42)
        eval(scaled)
        #expect(scaled.shape == [1, 128, 2, 4, 4])
        #expect(scaled.dtype == .float32)
        // Standard normal * 0.5 → std ≈ 0.5
        let std = scaled.variance().sqrt().item(Float.self)
        #expect(std > 0.3 && std < 0.7)
    }

    @Test func testNormalizeLatent() {
        let latent = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0)]).reshaped([1, 1, 1, 1, 4])
        // shape: (1, 1, 1, 1, 4)
        let normed = normalizeLatent(latent)
        eval(normed)
        // After normalization, should have ~zero mean, ~unit variance
        let mean = normed.mean().item(Float.self)
        #expect(abs(mean) < 0.01)
    }

    @Test func testDenormalizeLatent() {
        let latent = MLXArray.zeros([1, 2, 1, 1, 1])
        let mean = MLXArray([Float(1.0), Float(2.0)])
        let std = MLXArray([Float(0.5), Float(1.0)])
        let result = denormalizeLatent(latent, mean: mean, std: std)
        eval(result)
        // result = 0 * std + mean = mean
        let val0 = result[0, 0, 0, 0, 0].item(Float.self)
        let val1 = result[0, 1, 0, 0, 0].item(Float.self)
        #expect(abs(val0 - 1.0) < 0.01)
        #expect(abs(val1 - 2.0) < 0.01)
    }

    @Test func testEstimateMemoryUsage() {
        let shape = VideoLatentShape(batch: 1, channels: 128, frames: 16, height: 16, width: 24)
        let mem = estimateMemoryUsage(shape: shape, numSteps: 8)
        #expect(mem > 0)
        // Basic sanity: latent = 1*128*16*16*24 * 4 bytes = ~25 MB
        #expect(mem > 10_000_000)  // > 10 MB
    }
}
