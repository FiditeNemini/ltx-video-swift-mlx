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
    }
}
