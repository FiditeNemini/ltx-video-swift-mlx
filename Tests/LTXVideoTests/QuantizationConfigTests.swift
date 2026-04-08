//
//  QuantizationConfigTests.swift
//  ltx-video-swift-mlx
//

import Testing
@testable import LTXVideo

// MARK: - TransformerQuantization Tests

@Suite("TransformerQuantization")
struct TransformerQuantizationTests {
    @Test func testBf16() {
        let q = TransformerQuantization.bf16
        #expect(q.rawValue == "bf16")
        #expect(q.displayName == "BFloat16")
        #expect(q.bits == 16)
        #expect(q.groupSize == 64)
        #expect(!q.needsQuantization)
        #expect(q.memoryReduction == 1.0)
    }

    @Test func testQint8() {
        let q = TransformerQuantization.qint8
        #expect(q.rawValue == "qint8")
        #expect(q.displayName == "8-bit (qint8)")
        #expect(q.bits == 8)
        #expect(q.needsQuantization)
        #expect(q.memoryReduction == 0.5)
    }

    @Test func testInt4() {
        let q = TransformerQuantization.int4
        #expect(q.rawValue == "int4")
        #expect(q.displayName == "4-bit (int4)")
        #expect(q.bits == 4)
        #expect(q.needsQuantization)
        #expect(q.memoryReduction == 0.25)
    }

    @Test func testAllCases() {
        #expect(TransformerQuantization.allCases.count == 3)
    }

    @Test func testRawValueInit() {
        #expect(TransformerQuantization(rawValue: "bf16") == .bf16)
        #expect(TransformerQuantization(rawValue: "qint8") == .qint8)
        #expect(TransformerQuantization(rawValue: "int4") == .int4)
        #expect(TransformerQuantization(rawValue: "fp32") == nil)
    }
}

// MARK: - LTXQuantizationConfig Tests

@Suite("LTXQuantizationConfig")
struct LTXQuantizationConfigTests {
    @Test func testDefaultPreset() {
        let config = LTXQuantizationConfig.default
        #expect(config.transformer == .bf16)
    }

    @Test func testMemoryEfficientPreset() {
        let config = LTXQuantizationConfig.memoryEfficient
        #expect(config.transformer == .qint8)
    }

    @Test func testMinimalPreset() {
        let config = LTXQuantizationConfig.minimal
        #expect(config.transformer == .int4)
    }

    @Test func testDescription() {
        let config = LTXQuantizationConfig(transformer: .qint8)
        #expect(config.description.contains("8-bit"))
    }

    @Test func testCustomInit() {
        let config = LTXQuantizationConfig(transformer: .int4)
        #expect(config.transformer == .int4)
        #expect(config.mixedPrecision == nil)
    }

    @Test func testMixedDefaultPreset() {
        let config = LTXQuantizationConfig.mixedDefault
        #expect(config.mixedPrecision != nil)
        let mp = config.mixedPrecision!
        #expect(mp.highPrecisionBlocks.contains(0))
        #expect(mp.highPrecisionBlocks.contains(5))
        #expect(mp.highPrecisionBlocks.contains(42))
        #expect(mp.highPrecisionBlocks.contains(47))
        #expect(!mp.highPrecisionBlocks.contains(20))
        #expect(mp.highPrecisionBits == 8)
        #expect(mp.lowPrecisionBits == 4)
    }
}

// MARK: - MixedPrecisionConfig Tests

@Suite("MixedPrecisionConfig")
struct MixedPrecisionConfigTests {
    @Test func testDefaultPreset() {
        let mp = MixedPrecisionConfig.default
        // First 6 + last 6 = 12 blocks at high precision
        #expect(mp.highPrecisionBlocks.count == 12)
        #expect(mp.highPrecisionBits == 8)
        #expect(mp.lowPrecisionBits == 4)
        #expect(mp.groupSize == 64)
    }

    @Test func testConservativePreset() {
        let mp = MixedPrecisionConfig.conservative
        // First 8 + last 8 = 16 blocks
        #expect(mp.highPrecisionBlocks.count == 16)
        #expect(mp.highPrecisionBlocks.contains(0))
        #expect(mp.highPrecisionBlocks.contains(7))
        #expect(mp.highPrecisionBlocks.contains(40))
        #expect(mp.highPrecisionBlocks.contains(47))
    }

    @Test func testAggressivePreset() {
        let mp = MixedPrecisionConfig.aggressive
        // First 4 + last 4 = 8 blocks
        #expect(mp.highPrecisionBlocks.count == 8)
        #expect(!mp.highPrecisionBlocks.contains(4))
        #expect(!mp.highPrecisionBlocks.contains(43))
    }

    @Test func testCustomConfig() {
        let mp = MixedPrecisionConfig(
            highPrecisionBlocks: Set([0, 1, 46, 47]),
            highPrecisionBits: 16,
            lowPrecisionBits: 8,
            groupSize: 32
        )
        #expect(mp.highPrecisionBlocks.count == 4)
        #expect(mp.highPrecisionBits == 16)
        #expect(mp.lowPrecisionBits == 8)
        #expect(mp.groupSize == 32)
    }

    @Test func testEstimatedMemory() {
        let mp = MixedPrecisionConfig.default
        let estimated = mp.estimatedTransformerGB
        // Should be between int4 (~7GB) and qint8 (~13GB)
        #expect(estimated > 6.0)
        #expect(estimated < 14.0)
    }
}
