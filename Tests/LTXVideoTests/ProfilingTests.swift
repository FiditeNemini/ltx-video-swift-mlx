//  ProfilingTests.swift — Integration tests for MLXProfiler package usage
//  Copyright 2026 Vincent Gourbin

import Testing
import Foundation
@preconcurrency import MLX
@testable import LTXVideo
import MLXProfiler

// MARK: - LTXVideoProfiler Typealias

@Suite("LTXVideoProfiler Integration", .serialized)
struct LTXVideoProfilerIntegrationTests {
    @Test func testTypealiasWorks() {
        // LTXVideoProfiler is a typealias for MLXProfiler
        let profiler = LTXVideoProfiler.shared
        #expect(profiler === MLXProfiler.shared)
    }

    @Test func testEnableDisable() {
        let profiler = LTXVideoProfiler.shared
        profiler.disable(); profiler.reset()
        #expect(profiler.isEnabled == false)
        profiler.enable()
        #expect(profiler.isEnabled == true)
        profiler.disable()
    }

    @Test func testSessionBridge() {
        let profiler = LTXVideoProfiler.shared
        let session = ProfilingSession()
        session.title = "LTX-2.3 PROFILING REPORT"
        session.metadata = ["model": "distilled", "quant": "qint8"]

        profiler.enable()
        profiler.activeSession = session

        profiler.start("Text Encoding")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Text Encoding")

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].name == "Text Encoding")
        #expect(events[0].phase == .begin)

        let report = session.generateReport()
        #expect(report.contains("LTX-2.3 PROFILING REPORT"))
        #expect(report.contains("Text Encoding"))
        #expect(report.contains("GPU%"))

        profiler.activeSession = nil
        profiler.disable(); profiler.reset()
    }

    @Test func testChromeTraceExport() {
        let session = ProfilingSession()
        session.metadata = ["model": "distilled"]
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)

        let data = ChromeTraceExporter.export(session: session)
        #expect(data.count > 0)

        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
    }

    @Test func testSubPhaseProfiling() {
        let profiler = LTXVideoProfiler.shared
        let session = ProfilingSession()
        profiler.enable()
        profiler.activeSession = session

        // Simulate sub-phases like the pipeline does
        profiler.start("Text Encoding")
        profiler.start("Tokenization")
        profiler.end("Tokenization")
        profiler.start("Gemma Forward")
        profiler.end("Gemma Forward")
        profiler.end("Text Encoding")

        let events = session.getEvents()
        let names = events.map(\.name)
        #expect(names.contains("Tokenization"))
        #expect(names.contains("Gemma Forward"))
        #expect(names.contains("Text Encoding"))

        profiler.activeSession = nil
        profiler.disable(); profiler.reset()
    }
}

// MARK: - Frame Conversion Tests

@Suite("Frame Conversion")
struct FrameConversionTests {
    @Test func testTensorToImagesARGB() {
        // Create a small test tensor (2 frames, 4x4, RGB)
        let tensor = MLX.zeros([2, 4, 4, 3]).asType(.float32)
        let images = VideoExporter.tensorToImages(tensor)
        #expect(images.count == 2)
        #expect(images[0].width == 4)
        #expect(images[0].height == 4)
        // 32 bits per pixel (ARGB format)
        #expect(images[0].bitsPerPixel == 32)
    }

    @Test func testTensorToImagesValues() {
        // Create a red frame: RGB = (1, 0, 0)
        var data = [Float](repeating: 0, count: 2 * 2 * 3)
        // Set R channel to 1.0
        for i in stride(from: 0, to: data.count, by: 3) {
            data[i] = 1.0     // R
            data[i+1] = 0.0   // G
            data[i+2] = 0.0   // B
        }
        let tensor = MLXArray(data).reshaped([1, 2, 2, 3])
        let images = VideoExporter.tensorToImages(tensor)
        #expect(images.count == 1)
    }

    @Test func testTensorToImagesBatchDim() {
        // 5D tensor with batch dim
        let tensor = MLX.zeros([1, 3, 4, 4, 3]).asType(.float32)
        let images = VideoExporter.tensorToImages(tensor)
        #expect(images.count == 3)
    }

    @Test func testTensorToImagesInvalidShape() {
        // 3D tensor should return empty
        let tensor = MLX.zeros([4, 4, 3]).asType(.float32)
        let images = VideoExporter.tensorToImages(tensor)
        #expect(images.isEmpty)
    }
}

// MARK: - Mixed Precision Integration

@Suite("Mixed Precision Integration")
struct MixedPrecisionIntegrationTests {
    @Test func testMixedDefaultPreset() {
        let config = LTXQuantizationConfig.mixedDefault
        #expect(config.mixedPrecision != nil)
        let mp = config.mixedPrecision!
        #expect(mp.highPrecisionBlocks.count == 12)
        #expect(mp.highPrecisionBits == 8)
        #expect(mp.lowPrecisionBits == 4)
    }

    @Test func testEstimatedMemoryBetweenInt4AndQint8() {
        let mp = MixedPrecisionConfig.default
        let estimated = mp.estimatedTransformerGB
        // Should be between int4 (~7GB) and qint8 (~13GB)
        #expect(estimated > 6.0)
        #expect(estimated < 14.0)
    }
}
