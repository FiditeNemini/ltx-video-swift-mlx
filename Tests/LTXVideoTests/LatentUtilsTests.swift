//
//  LatentUtilsTests.swift
//  ltx-video-swift-mlx
//
//  Tests for latent utility functions.
//  NOTE: Most tests require Metal (MLXArray operations).

import Testing
import CoreGraphics
import Foundation
import ImageIO
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

// MARK: - Pixel Conversion (loadImage / loadVideo)
//
// Locks the vectorized RGBA→float conversion (LatentUtils.swift) to the exact
// values and channel order the scalar loop it replaced produced: 255 → 1.0,
// 0 → -1.0, 128 → 128/127.5 - 1.0. A silent R/B channel swap would pass
// unnoticed on a photographic image but not on these known quadrant colors.

@Suite("Pixel conversion (loadImage / loadVideo)")
struct PixelConversionTests {

    /// Writes a 4x4 device-RGB PNG with four solid-color quadrants: red,
    /// black, white, gray128. Device RGB (no ICC profile) on both the write
    /// and the read side (`loadImage` also uses `CGColorSpaceCreateDeviceRGB`)
    /// avoids any color-management pass that could shift byte values.
    private func makeQuadrantPNG() throws -> URL {
        let width = 4, height = 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw LTXError.videoProcessingFailed("Failed to create test PNG context")
        }

        context.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)     // red
        context.fill(CGRect(x: 0, y: 2, width: 2, height: 2))
        context.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)     // black
        context.fill(CGRect(x: 2, y: 2, width: 2, height: 2))
        context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)     // white
        context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        context.setFillColor(red: 128.0 / 255.0, green: 128.0 / 255.0, blue: 128.0 / 255.0, alpha: 1)  // gray128
        context.fill(CGRect(x: 2, y: 0, width: 2, height: 2))

        guard let cgImage = context.makeImage() else {
            throw LTXError.videoProcessingFailed("Failed to snapshot test PNG context")
        }

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ltx-quadrant-\(UUID().uuidString).png")
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil
        ) else {
            throw LTXError.videoProcessingFailed("Failed to create PNG destination")
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw LTXError.videoProcessingFailed("Failed to write test PNG")
        }
        return url
    }

    @Test func testLoadImageExactNormalizedValues() throws {
        let url = try makeQuadrantPNG()
        defer { try? FileManager.default.removeItem(at: url) }

        let tensor = try loadImage(from: url.path, width: 4, height: 4)
        eval(tensor)

        #expect(tensor.shape == [1, 3, 1, 4, 4])

        let byte255: Float = 255.0 / 127.5 - 1.0   // 1.0
        let byte0: Float = 0.0 / 127.5 - 1.0       // -1.0
        let byte128: Float = 128.0 / 127.5 - 1.0   // ≈ 0.003922

        // Exact channel order: a red pixel must read (1.0, -1.0, -1.0) on
        // (R, G, B) — an R/B swap would still look plausible on a gray pixel
        // but not here. Quadrant → tensor (y, x) verified empirically (a
        // CGContext's y-up drawing space maps to a top-down pixel buffer).
        #expect(tensor[0, 0, 0, 0, 0].item(Float.self) == byte255)  // red quadrant, R
        #expect(tensor[0, 1, 0, 0, 0].item(Float.self) == byte0)    // red quadrant, G
        #expect(tensor[0, 2, 0, 0, 0].item(Float.self) == byte0)    // red quadrant, B

        #expect(tensor[0, 0, 0, 0, 3].item(Float.self) == byte0)    // black quadrant, R
        #expect(tensor[0, 1, 0, 0, 3].item(Float.self) == byte0)    // black quadrant, G
        #expect(tensor[0, 2, 0, 0, 3].item(Float.self) == byte0)    // black quadrant, B

        #expect(tensor[0, 0, 0, 3, 0].item(Float.self) == byte255)  // white quadrant, R
        #expect(tensor[0, 1, 0, 3, 0].item(Float.self) == byte255)  // white quadrant, G
        #expect(tensor[0, 2, 0, 3, 0].item(Float.self) == byte255)  // white quadrant, B

        #expect(tensor[0, 0, 0, 3, 3].item(Float.self) == byte128)  // gray quadrant, R
        #expect(tensor[0, 1, 0, 3, 3].item(Float.self) == byte128)  // gray quadrant, G
        #expect(tensor[0, 2, 0, 3, 3].item(Float.self) == byte128)  // gray quadrant, B

        // Every value stays in range regardless of which quadrant it lands in.
        #expect(tensor.min().item(Float.self) >= -1.0)
        #expect(tensor.max().item(Float.self) <= 1.0)
    }

    @Test func testLoadVideoShapeAndRange() async throws {
        let path = "\(#filePath)"
            .replacingOccurrences(of: "Tests/LTXVideoTests/LatentUtilsTests.swift", with: "")
            + "docs/examples/lipdub/lipdub-teaser-french-ours-768x512-121f.mp4"
        guard FileManager.default.fileExists(atPath: path) else {
            Issue.record("Example video missing at \(path); skipping shape/range check")
            return
        }

        let tensor = try await loadVideo(from: path, width: 128, height: 128, numFrames: 9)
        eval(tensor)

        #expect(tensor.shape == [1, 3, 9, 128, 128])
        #expect(tensor.min().item(Float.self) >= -1.0)
        #expect(tensor.max().item(Float.self) <= 1.0)
    }
}
