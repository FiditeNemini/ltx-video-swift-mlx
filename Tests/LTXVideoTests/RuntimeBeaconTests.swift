// RuntimeBeaconTests.swift — Unit tests for the opt-in activity beacon
// Copyright 2026

import Foundation
import Testing
@testable import LTXVideo

/// Serialized: RuntimeBeacon.isEnabled and directoryOverride are global state.
@Suite("RuntimeBeacon", .serialized)
struct RuntimeBeaconTests {

    /// Runs `body` with the beacon enabled and sandboxed into a fresh temp
    /// directory, restoring global state afterwards.
    private func withSandbox<T>(
        enabled: Bool = true,
        _ body: (URL) throws -> T
    ) throws -> T {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("beacon-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        RuntimeBeacon.directoryOverride = dir
        RuntimeBeacon.isEnabled = enabled
        // The env var force-enables the beacon, so a developer (or CI) with
        // LTX_RUNTIME_BEACON=1 exported would fail the disabled-by-default
        // assertion. Neutralize it for the duration and put it back after.
        let exportedBeaconEnv = getenv("LTX_RUNTIME_BEACON").map { String(cString: $0) }
        unsetenv("LTX_RUNTIME_BEACON")
        defer {
            RuntimeBeacon.isEnabled = false
            RuntimeBeacon.directoryOverride = nil
            if let exportedBeaconEnv { setenv("LTX_RUNTIME_BEACON", exportedBeaconEnv, 1) }
            try? FileManager.default.removeItem(at: dir)
        }
        return try body(dir)
    }

    private func manifestFiles(in dir: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "json" } ?? []
    }

    @Test("disabled by default: begin returns nil and writes nothing")
    func disabledByDefault() throws {
        try withSandbox(enabled: false) { dir in
            #expect(RuntimeBeacon.begin(task: "generate") == nil)
            #expect(manifestFiles(in: dir).isEmpty)
        }
    }

    @Test("session lifecycle: manifest created, updated, then deleted on end")
    func sessionLifecycle() throws {
        try withSandbox { dir in
            let session = try #require(RuntimeBeacon.begin(task: "generate", model: "distilled"))

            let files = manifestFiles(in: dir)
            #expect(files.count == 1)
            let url = try #require(files.first)

            var json = try #require(
                try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            #expect(json["runtime"] as? String == "ltx-video-swift-mlx")
            #expect(json["task"] as? String == "generate")
            #expect(json["model"] as? String == "distilled")
            #expect(json["pid"] as? Int32 == ProcessInfo.processInfo.processIdentifier)
            #expect(json["version"] as? Int == RuntimeBeacon.schemaVersion)
            #expect(json["phase"] == nil)

            session.update(phase: "denoising", step: 3, totalSteps: 11)
            json = try #require(
                try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            #expect(json["phase"] as? String == "denoising")
            #expect(json["step"] as? Int == 3)
            #expect(json["totalSteps"] as? Int == 11)

            session.end()
            #expect(manifestFiles(in: dir).isEmpty)

            // end() is idempotent and update() after end() writes nothing back.
            session.end()
            session.update(phase: "decoding")
            #expect(manifestFiles(in: dir).isEmpty)
        }
    }

    @Test("deinit removes the manifest even without an explicit end()")
    func deinitCleansUp() throws {
        try withSandbox { dir in
            do {
                let session = try #require(RuntimeBeacon.begin(task: "train"))
                #expect(manifestFiles(in: dir).count == 1)
                _ = session  // silence unused warning; deallocated at scope exit
            }
            #expect(manifestFiles(in: dir).isEmpty)
        }
    }

    @Test("begin removes stale manifests left by dead processes, keeps live ones")
    func staleManifestCleanup() throws {
        try withSandbox { dir in
            // Fake leftover from a crashed process. PID 99999997 is outside
            // macOS's PID range, so kill(pid, 0) fails with ESRCH.
            let stale = dir.appendingPathComponent("99999997-deadbeef.json")
            try Data("{}".utf8).write(to: stale)
            // Manifest of a live foreign process (launchd, pid 1) must survive.
            let live = dir.appendingPathComponent("1-cafebabe.json")
            try Data("{}".utf8).write(to: live)

            let session = try #require(RuntimeBeacon.begin(task: "generate"))
            defer { session.end() }

            let names = manifestFiles(in: dir).map(\.lastPathComponent)
            #expect(!names.contains("99999997-deadbeef.json"))
            #expect(names.contains("1-cafebabe.json"))
            #expect(names.count == 2)  // live foreign + our session
        }
    }

    /// `update()` racing `end()` must never leave a manifest behind. Before the
    /// I/O moved inside the state lock, an update that had passed the `!ended`
    /// check could be preempted by `end()` and then re-create the file it had
    /// just deleted — a phantom nothing could remove afterwards (`end()` is
    /// idempotent, and stale-GC skips own-pid files), so a monitor showed
    /// activity for the rest of the process.
    @Test("update() racing end() never resurrects the manifest")
    func concurrentUpdateAndEndLeaveNoManifest() throws {
        for attempt in 0 ..< 40 {
            try withSandbox { dir in
                let session = try #require(RuntimeBeacon.begin(task: "generate", model: "distilled"))

                // Hammer update() from several threads while end() lands in the
                // middle; the interleaving that used to resurrect the file is
                // timing-dependent, hence the repeats.
                DispatchQueue.concurrentPerform(iterations: 8) { i in
                    if i == 4 {
                        session.end()
                    } else {
                        session.update(phase: "denoising", step: i, totalSteps: 8)
                    }
                }
                // Late updates after end() must also stay no-ops.
                session.update(phase: "denoising", step: 99, totalSteps: 8)
                session.end()

                #expect(manifestFiles(in: dir).isEmpty,
                        "manifest survived end() on attempt \(attempt)")
            }
        }
    }

    /// Sequential updates keep the last write on disk (the lock also orders the
    /// atomic writes, which racing writers outside it could commit in reverse).
    @Test("the last update is what remains on disk")
    func lastUpdateWins() throws {
        try withSandbox { dir in
            let session = try #require(RuntimeBeacon.begin(task: "generate", model: "distilled"))
            defer { session.end() }

            for step in 1 ... 5 { session.update(phase: "denoising", step: step, totalSteps: 5) }

            let file = try #require(manifestFiles(in: dir).first)
            let json = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
            #expect(json?["step"] as? Int == 5)
            #expect(json?["phase"] as? String == "denoising")
        }
    }
}
