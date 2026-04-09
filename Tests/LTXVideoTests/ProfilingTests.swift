//  ProfilingTests.swift
//  Copyright 2026 Vincent Gourbin

import Testing
import Foundation
@testable import LTXVideo

// MARK: - ProfilingEvent Tests

@Suite("ProfilingEvent")
struct ProfilingEventTests {
    @Test func testCategoryThreadIds() {
        let categories: [ProfilingCategory] = [
            .textEncoderLoad, .textEncoding, .textEncoderUnload, .vlmInterpretation,
            .transformerLoad, .denoisingLoop, .denoisingStep, .transformerUnload,
            .upscaler, .vaeLoad, .vaeDecode, .audioLoad, .audioDenoise,
            .postProcess, .memoryOp, .evalSync, .custom,
        ]
        for category in categories {
            #expect(category.threadId > 0)
            #expect(!category.threadName.isEmpty)
        }
    }

    @Test func testTextEncodingCategoriesShareThread() {
        #expect(ProfilingCategory.textEncoderLoad.threadId == ProfilingCategory.textEncoding.threadId)
        #expect(ProfilingCategory.textEncoding.threadId == ProfilingCategory.textEncoderUnload.threadId)
    }

    @Test func testTransformerCategoriesShareThread() {
        #expect(ProfilingCategory.transformerLoad.threadId == ProfilingCategory.denoisingLoop.threadId)
        #expect(ProfilingCategory.denoisingLoop.threadId == ProfilingCategory.denoisingStep.threadId)
    }

    @Test func testSortOrder() {
        #expect(ProfilingCategory.textEncoding.sortOrder < ProfilingCategory.denoisingLoop.sortOrder)
        #expect(ProfilingCategory.denoisingLoop.sortOrder < ProfilingCategory.vaeDecode.sortOrder)
        #expect(ProfilingCategory.vaeDecode.sortOrder < ProfilingCategory.postProcess.sortOrder)
    }

    @Test func testEventInit() {
        let event = ProfilingEvent(name: "Test", category: .textEncoding, phase: .begin, timestampUs: 1000)
        #expect(event.name == "Test")
        #expect(event.category == .textEncoding)
        #expect(event.phase == .begin)
        #expect(event.durationUs == nil)
        #expect(event.threadId == ProfilingCategory.textEncoding.threadId)
    }

    @Test func testCompleteEventWithDuration() {
        let event = ProfilingEvent(
            name: "Step 1/8", category: .denoisingStep, phase: .complete,
            timestampUs: 5000, durationUs: 2000, stepIndex: 1, totalSteps: 8
        )
        #expect(event.durationUs == 2000)
        #expect(event.stepIndex == 1)
        #expect(event.totalSteps == 8)
    }

    @Test func testPhaseRawValues() {
        #expect(ProfilingPhase.begin.rawValue == "B")
        #expect(ProfilingPhase.end.rawValue == "E")
        #expect(ProfilingPhase.complete.rawValue == "X")
        #expect(ProfilingPhase.counter.rawValue == "C")
        #expect(ProfilingPhase.metadata.rawValue == "M")
    }
}

// MARK: - ProfilingConfig Tests

@Suite("ProfilingConfig")
struct ProfilingConfigTests {
    @Test func testDefaultConfig() {
        let config = ProfilingConfig()
        #expect(config.trackMemory == true)
        #expect(config.trackPerStepMemory == false)
        #expect(config.exportChromeTrace == true)
        #expect(config.printSummary == true)
    }

    @Test func testSingleRunPreset() {
        let config = ProfilingConfig.singleRun
        #expect(config.trackMemory == true)
        #expect(config.exportChromeTrace == true)
    }

    @Test func testBenchmarkPreset() {
        let config = ProfilingConfig.benchmark(runs: 5, warmup: 2)
        #expect(config.exportChromeTrace == false)
        #expect(config.trackMemory == true)
    }

    @Test func testDetailedPreset() {
        let config = ProfilingConfig.detailed
        #expect(config.trackPerStepMemory == true)
        #expect(config.exportChromeTrace == true)
    }
}

// MARK: - ProfilingSession Tests

@Suite("ProfilingSession")
struct ProfilingSessionTests {
    @Test func testSessionInit() {
        let session = ProfilingSession(config: .singleRun)
        #expect(!session.sessionId.isEmpty)
        #expect(session.systemRAMGB > 0)
        #expect(session.modelVariant == "")
    }

    @Test func testBeginEndPhase() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("Text Encoding", category: .textEncoding)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Text Encoding", category: .textEncoding)

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].phase == .begin)
        #expect(events[1].phase == .end)
        #expect(events[1].timestampUs > events[0].timestampUs)
    }

    @Test func testRecordComplete() {
        let session = ProfilingSession(config: .singleRun)
        session.recordComplete("Quick op", category: .evalSync, durationUs: 5000)
        let events = session.getEvents()
        #expect(events.count == 1)
        #expect(events[0].phase == .complete)
        #expect(events[0].durationUs == 5000)
    }

    @Test func testRecordDenoisingStep() {
        let session = ProfilingSession(config: ProfilingConfig(trackPerStepMemory: true))
        session.recordDenoisingStep(index: 1, total: 8, durationUs: 100_000)
        session.recordDenoisingStep(index: 2, total: 8, durationUs: 95_000)
        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].stepIndex == 1)
        #expect(events[1].stepIndex == 2)
    }

    @Test func testMemoryTimeline() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)
        let timeline = session.getMemoryTimeline()
        #expect(timeline.count == 2)
        #expect(timeline[0].context == "begin:Test")
        #expect(timeline[1].context == "end:Test")
    }

    @Test func testMemoryTimelineDisabled() {
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)
        #expect(session.getMemoryTimeline().isEmpty)
    }

    @Test func testRecordMemorySnapshot() {
        let session = ProfilingSession(config: .singleRun)
        session.recordMemorySnapshot(context: "after_load")
        #expect(session.getMemoryTimeline().count == 1)
        #expect(session.getMemoryTimeline()[0].context == "after_load")
    }

    @Test func testGenerateReport() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.quantization = "qint8"
        session.resolution = "256x256"
        session.frames = 9
        session.steps = 8

        session.beginPhase("Text Encoding", category: .textEncoding)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Text Encoding", category: .textEncoding)
        session.recordDenoisingStep(index: 1, total: 2, durationUs: 50_000)
        session.recordDenoisingStep(index: 2, total: 2, durationUs: 55_000)

        let report = session.generateReport()
        #expect(report.contains("LTX-2.3 PROFILING REPORT"))
        #expect(report.contains("distilled"))
        #expect(report.contains("Text Encoding"))
        #expect(report.contains("DENOISING STEP STATISTICS"))
        #expect(report.contains("MEMORY"))
    }

    @Test func testElapsedSeconds() {
        let session = ProfilingSession(config: .singleRun)
        Thread.sleep(forTimeInterval: 0.01)
        #expect(session.elapsedSeconds > 0.005)
    }

    @Test func testMetadataFields() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "dev"
        session.quantization = "bf16"
        session.resolution = "768x512"
        session.frames = 121
        session.steps = 40
        #expect(session.modelVariant == "dev")
        #expect(session.frames == 121)
    }
}

// MARK: - Category Inference Tests

@Suite("Category Inference")
struct CategoryInferenceTests {
    @Test func testInferTextPhases() {
        #expect(ProfilingSession.inferCategory("Load Text Encoder") == .textEncoderLoad)
        #expect(ProfilingSession.inferCategory("Load Gemma") == .textEncoderLoad)
        #expect(ProfilingSession.inferCategory("Text Encoding") == .textEncoding)
        #expect(ProfilingSession.inferCategory("Unload Gemma") == .textEncoderUnload)
    }

    @Test func testInferTransformerPhases() {
        #expect(ProfilingSession.inferCategory("Load Transformer") == .transformerLoad)
        #expect(ProfilingSession.inferCategory("Unload Transformer") == .transformerUnload)
        #expect(ProfilingSession.inferCategory("Denoising loop") == .denoisingLoop)
    }

    @Test func testInferVAEPhases() {
        #expect(ProfilingSession.inferCategory("Load VAE") == .vaeLoad)
        #expect(ProfilingSession.inferCategory("VAE Decode") == .vaeDecode)
    }

    @Test func testInferOtherPhases() {
        #expect(ProfilingSession.inferCategory("Upscaler 2x") == .upscaler)
        #expect(ProfilingSession.inferCategory("VLM prompt") == .vlmInterpretation)
        #expect(ProfilingSession.inferCategory("Load Audio") == .audioLoad)
        #expect(ProfilingSession.inferCategory("unknown") == .custom)
    }
}

// MARK: - ChromeTraceExporter Tests

@Suite("ChromeTraceExporter")
struct ChromeTraceExporterTests {
    @Test func testExportSingleSession() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.beginPhase("Text Encoding", category: .textEncoding)
        session.endPhase("Text Encoding", category: .textEncoding)

        let data = ChromeTraceExporter.export(session: session)
        #expect(data.count > 0)
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
        let traceEvents = json?["traceEvents"] as? [[String: Any]]
        #expect((traceEvents?.count ?? 0) > 0)
        let sessionInfo = traceEvents?.first { ($0["name"] as? String) == "Session Info" }
        #expect(sessionInfo != nil)
    }

    @Test func testExportComparison() {
        let s1 = ProfilingSession(config: .singleRun)
        s1.beginPhase("Test", category: .textEncoding)
        s1.endPhase("Test", category: .textEncoding)
        let s2 = ProfilingSession(config: .singleRun)
        s2.beginPhase("Test", category: .textEncoding)
        s2.endPhase("Test", category: .textEncoding)

        let data = ChromeTraceExporter.exportComparison(sessions: [
            (label: "A", session: s1), (label: "B", session: s2),
        ])
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let pids = Set((json?["traceEvents"] as? [[String: Any]])?.compactMap { $0["pid"] as? Int } ?? [])
        #expect(pids.contains(1) && pids.contains(2))
    }

    @Test func testExportIncludesMemoryCounters() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)
        let data = ChromeTraceExporter.export(session: session)
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let counters = (json?["traceEvents"] as? [[String: Any]])?.filter { ($0["ph"] as? String) == "C" } ?? []
        #expect(!counters.isEmpty)
    }
}

// MARK: - BenchmarkRunner Tests

@Suite("BenchmarkRunner")
struct BenchmarkRunnerTests {
    @Test func testAggregateEmpty() {
        let result = BenchmarkAggregator.aggregate(sessions: [], warmupCount: 0)
        #expect(result.measuredRuns == 0)
        #expect(result.phaseStats.isEmpty)
    }

    @Test func testAggregateSingleSession() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.quantization = "qint8"
        session.resolution = "256x256"
        session.frames = 9
        session.steps = 8
        session.beginPhase("Text Encoding", category: .textEncoding)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Text Encoding", category: .textEncoding)

        let result = BenchmarkAggregator.aggregate(sessions: [session], warmupCount: 0)
        #expect(result.measuredRuns == 1)
        #expect(!result.phaseStats.isEmpty)
        #expect(result.totalStats.meanMs > 0)
    }

    @Test func testPhasesSortedByCategory() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        // Record phases out of order
        session.beginPhase("VAE Decode", category: .vaeDecode)
        session.endPhase("VAE Decode", category: .vaeDecode)
        session.beginPhase("Text Encoding", category: .textEncoding)
        session.endPhase("Text Encoding", category: .textEncoding)

        let result = BenchmarkAggregator.aggregate(sessions: [session], warmupCount: 0)
        #expect(result.phaseStats.count == 2)
        #expect(result.phaseStats[0].name == "Text Encoding")
        #expect(result.phaseStats[1].name == "VAE Decode")
    }

    @Test func testBenchmarkReport() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.quantization = "bf16"
        session.resolution = "512x512"
        session.frames = 9; session.steps = 8
        session.beginPhase("Denoising", category: .denoisingLoop)
        session.recordDenoisingStep(index: 1, total: 2, durationUs: 100_000)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Denoising", category: .denoisingLoop)

        let result = BenchmarkAggregator.aggregate(sessions: [session], warmupCount: 1)
        let report = result.generateReport()
        #expect(report.contains("LTX-2.3 BENCHMARK REPORT"))
        #expect(report.contains("PHASE TIMINGS"))
        #expect(report.contains("MEMORY"))
    }
}

// MARK: - LTXVideoProfiler Tests

@Suite("LTXVideoProfiler", .serialized)
struct LTXVideoProfilerTests {
    @Test func testEnableDisable() {
        let profiler = LTXVideoProfiler.shared
        profiler.disable(); profiler.reset()
        #expect(profiler.isEnabled == false)
        profiler.enable()
        #expect(profiler.isEnabled == true)
        profiler.disable()
        #expect(profiler.isEnabled == false)
    }

    @Test func testRecordTimings() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        profiler.start("Phase A")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Phase A")

        let timings = profiler.getTimings()
        #expect(timings.count >= 1)
        #expect(timings.last?.name == "Phase A")
        #expect((timings.last?.duration ?? 0) > 0.005)
        profiler.disable(); profiler.reset()
    }

    @Test func testRecordDirect() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        profiler.record("Direct Phase", duration: 1.234)
        let timings = profiler.getTimings()
        #expect(timings.contains(where: { $0.name == "Direct Phase" && $0.duration == 1.234 }))
        profiler.disable(); profiler.reset()
    }

    @Test func testRecordSteps() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        profiler.setTotalSteps(4)
        profiler.recordStep(duration: 1.5)
        profiler.recordStep(duration: 1.4)
        let steps = profiler.getStepTimes()
        #expect(steps.count >= 2)
        profiler.disable(); profiler.reset()
    }

    @Test func testSessionBridge() {
        let profiler = LTXVideoProfiler.shared
        let session = ProfilingSession(config: .singleRun)
        profiler.enable()
        profiler.activeSession = session
        profiler.start("Bridged Phase")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Bridged Phase")

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].name == "Bridged Phase")
        #expect(events[0].phase == .begin)
        #expect(events[1].phase == .end)
        profiler.activeSession = nil; profiler.disable(); profiler.reset()
    }

    @Test func testDisabledProfilerNoOps() {
        let profiler = LTXVideoProfiler.shared
        profiler.disable(); profiler.reset()
        profiler.start("Should not record")
        profiler.end("Should not record")
        profiler.recordStep(duration: 1.0)
        #expect(profiler.getTimings().isEmpty)
    }

    @Test func testEndWithoutStartIsNoOp() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        profiler.end("Never started")
        #expect(profiler.getTimings().isEmpty)
        profiler.disable(); profiler.reset()
    }

    @Test func testGenerateReport() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        profiler.start("Text Encoding")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Text Encoding")
        let report = profiler.generateReport()
        #expect(report.contains("Text Encoding"))
        #expect(report.contains("PHASE TIMINGS"))
        profiler.disable(); profiler.reset()
    }

    @Test func testMeasureClosure() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        let result = profiler.measure("Computation") { 42 }
        #expect(result == 42)
        #expect(profiler.getTimings().contains(where: { $0.name == "Computation" }))
        profiler.disable(); profiler.reset()
    }

    @Test func testMeasureThrowingClosure() throws {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        struct TestError: Error {}
        #expect(throws: TestError.self) {
            try profiler.measure("Throwing") { throw TestError() as Error }
        }
        profiler.disable(); profiler.reset()
    }
}

// MARK: - TimingEntry Tests

@Suite("TimingEntry")
struct TimingEntryTests {
    @Test func testDurationFormatMs() {
        let entry = TimingEntry(name: "fast", duration: 0.05, startTime: Date(), endTime: Date())
        #expect(entry.durationMs == 50.0)
        #expect(entry.durationFormatted.contains("ms"))
    }

    @Test func testDurationFormatSeconds() {
        let entry = TimingEntry(name: "medium", duration: 5.5, startTime: Date(), endTime: Date())
        #expect(entry.durationFormatted.contains("s"))
        #expect(!entry.durationFormatted.contains("m "))
    }

    @Test func testDurationFormatMinutes() {
        let entry = TimingEntry(name: "long", duration: 125.3, startTime: Date(), endTime: Date())
        #expect(entry.durationFormatted.contains("m"))
    }
}

// MARK: - MemoryTimelineEntry Tests

@Suite("MemoryTimelineEntry")
struct MemoryTimelineEntryTests {
    @Test func testInit() {
        let entry = MemoryTimelineEntry(
            timestampUs: 1000, context: "test",
            mlxActiveMB: 1024.5, mlxCacheMB: 256.0,
            mlxPeakMB: 2048.0, processFootprintMB: 4096.0,
            cpuTimeSeconds: 12.5, gpuUtilization: 85
        )
        #expect(entry.mlxActiveMB == 1024.5)
        #expect(entry.processFootprintMB == 4096.0)
        #expect(entry.cpuTimeSeconds == 12.5)
        #expect(entry.gpuUtilization == 85)
    }
}
