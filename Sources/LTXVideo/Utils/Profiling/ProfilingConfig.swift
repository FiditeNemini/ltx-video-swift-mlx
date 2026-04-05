// ProfilingConfig.swift - Configuration for profiling sessions
// Copyright 2026 Vincent Gourbin

import Foundation

/// Configuration for what to profile and how
public struct ProfilingConfig: Sendable {
    public var trackMemory: Bool
    public var trackPerStepMemory: Bool
    public var outputDirectory: URL?
    public var exportChromeTrace: Bool
    public var printSummary: Bool

    public init(
        trackMemory: Bool = true,
        trackPerStepMemory: Bool = false,
        outputDirectory: URL? = nil,
        exportChromeTrace: Bool = true,
        printSummary: Bool = true
    ) {
        self.trackMemory = trackMemory
        self.trackPerStepMemory = trackPerStepMemory
        self.outputDirectory = outputDirectory
        self.exportChromeTrace = exportChromeTrace
        self.printSummary = printSummary
    }

    public static let singleRun = ProfilingConfig()

    public static func benchmark(runs: Int = 3, warmup: Int = 1) -> ProfilingConfig {
        ProfilingConfig(trackMemory: true, trackPerStepMemory: false, exportChromeTrace: false, printSummary: true)
    }

    public static let detailed = ProfilingConfig(
        trackMemory: true, trackPerStepMemory: true, exportChromeTrace: true, printSummary: true
    )
}
