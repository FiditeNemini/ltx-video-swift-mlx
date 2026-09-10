// ModelDownloaderCacheSizeTests.swift
// Copyright 2026
//
// Regression test for the Fluxforge Studio model-storage-locations ask:
// ModelDownloader.cacheSize() must follow a file symlink to its target's real
// size, not report the symlink's own (near-zero) size. See
// docs/FRAMEWORK_ASKS_STORAGE.md (Fluxforge Studio repo) ask #4.

import Foundation
import Testing
@testable import LTXVideo

@Suite("ModelDownloader.cacheSize() follows symlinked weights")
struct ModelDownloaderCacheSizeTests {

    @Test func followsSymlinkedWeightToItsRealSize() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("ltx-cachesize-\(UUID().uuidString)")
        let cacheDir = root.appendingPathComponent("cache")
        let externalDir = root.appendingPathComponent("external")
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: externalDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        // A regular, non-relocated file.
        let regularData = Data(repeating: 0x41, count: 1_000)
        try regularData.write(to: cacheDir.appendingPathComponent("config.json"))

        // A "relocated" weight: real bytes live on the external target,
        // the cache directory only holds an absolute file symlink to it.
        let targetData = Data(repeating: 0x42, count: 50_000)
        let targetURL = externalDir.appendingPathComponent("model.safetensors")
        try targetData.write(to: targetURL)
        let symlinkURL = cacheDir.appendingPathComponent("model.safetensors")
        try fm.createSymbolicLink(at: symlinkURL, withDestinationURL: targetURL)

        let downloader = ModelDownloader(cacheDir: cacheDir)
        let size = try await downloader.cacheSize()

        #expect(size == Int64(regularData.count + targetData.count))
    }

    @Test func brokenSymlinkContributesZeroInsteadOfThrowing() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("ltx-cachesize-\(UUID().uuidString)")
        let cacheDir = root.appendingPathComponent("cache")
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let regularData = Data(repeating: 0x41, count: 1_234)
        try regularData.write(to: cacheDir.appendingPathComponent("config.json"))

        // Simulates an unmounted external disk: the symlink target doesn't exist.
        let missingTarget = root.appendingPathComponent("not-mounted/model.safetensors")
        let symlinkURL = cacheDir.appendingPathComponent("model.safetensors")
        try fm.createSymbolicLink(at: symlinkURL, withDestinationURL: missingTarget)

        let downloader = ModelDownloader(cacheDir: cacheDir)
        let size = try await downloader.cacheSize()

        #expect(size == Int64(regularData.count))
    }
}
