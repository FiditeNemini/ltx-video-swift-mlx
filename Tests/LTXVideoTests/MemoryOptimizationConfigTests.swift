//
//  MemoryOptimizationConfigTests.swift
//  ltx-video-swift-mlx
//

import Testing
@testable import LTXVideo

// MARK: - MemoryOptimizationConfig Tests

@Suite("MemoryOptimizationConfig")
struct MemoryOptimizationConfigTests {
    @Test func testDisabledPreset() {
        let config = MemoryOptimizationConfig.disabled
        #expect(config.evalFrequency == 8)
        #expect(!config.clearCacheOnEval)
        #expect(!config.unloadAfterUse)
        #expect(config.unloadSleepSeconds == 0)
        #expect(config.vaeTemporalTileSize == 0)
    }

    @Test func testLightPreset() {
        let config = MemoryOptimizationConfig.light
        #expect(config.evalFrequency == 4)
        #expect(!config.clearCacheOnEval)
        #expect(config.unloadAfterUse)
        #expect(config.unloadSleepSeconds == 0.3)
        #expect(config.vaeTemporalTileSize == 0)
    }

    @Test func testModeratePreset() {
        let config = MemoryOptimizationConfig.moderate
        #expect(config.evalFrequency == 2)
        #expect(config.clearCacheOnEval)
        #expect(config.unloadAfterUse)
        #expect(config.vaeTemporalTileSize == 8)
        #expect(config.vaeTemporalTileOverlap == 1)
    }

    @Test func testAggressivePreset() {
        let config = MemoryOptimizationConfig.aggressive
        #expect(config.evalFrequency == 1)
        #expect(config.clearCacheOnEval)
        #expect(config.unloadAfterUse)
        #expect(config.vaeTemporalTileSize == 6)
    }

    @Test func testDefaultIsLight() {
        let def = MemoryOptimizationConfig.default
        let light = MemoryOptimizationConfig.light
        #expect(def.evalFrequency == light.evalFrequency)
        #expect(def.clearCacheOnEval == light.clearCacheOnEval)
        #expect(def.unloadAfterUse == light.unloadAfterUse)
    }

    @Test func testRecommendedForRAM() {
        let r16 = MemoryOptimizationConfig.recommended(forRAMGB: 16)
        #expect(r16.evalFrequency == 1)  // aggressive

        let r32 = MemoryOptimizationConfig.recommended(forRAMGB: 32)
        #expect(r32.evalFrequency == 1)  // aggressive (<=32)

        let r48 = MemoryOptimizationConfig.recommended(forRAMGB: 48)
        #expect(r48.evalFrequency == 2)  // moderate

        let r64 = MemoryOptimizationConfig.recommended(forRAMGB: 64)
        #expect(r64.evalFrequency == 2)  // moderate (33-64)

        let r96 = MemoryOptimizationConfig.recommended(forRAMGB: 96)
        #expect(r96.evalFrequency == 8)  // disabled (96+)

        let r128 = MemoryOptimizationConfig.recommended(forRAMGB: 128)
        #expect(r128.evalFrequency == 8) // disabled (96+)
    }

    @Test func testCustomInit() {
        let config = MemoryOptimizationConfig(
            evalFrequency: 3,
            clearCacheOnEval: true,
            unloadAfterUse: false,
            unloadSleepSeconds: 2.0,
            vaeTemporalTileSize: 10,
            vaeTemporalTileOverlap: 2
        )
        #expect(config.evalFrequency == 3)
        #expect(config.clearCacheOnEval)
        #expect(!config.unloadAfterUse)
        #expect(config.unloadSleepSeconds == 2.0)
        #expect(config.vaeTemporalTileSize == 10)
        #expect(config.vaeTemporalTileOverlap == 2)
    }
}
