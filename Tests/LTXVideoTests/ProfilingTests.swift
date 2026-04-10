//  ProfilingTests.swift — Integration tests for MLXProfiler package usage
//  Copyright 2026 Vincent Gourbin

import Testing
import Foundation
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
}
