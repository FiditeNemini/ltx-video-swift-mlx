// TemporalTilingTests.swift — the tiler's geometry contract
// Copyright 2026

import Foundation
import Testing
@testable import LTXVideo

@Suite("Temporal tiling")
struct TemporalTilingTests {

    @Test func shortCanvasIsOneTileWithNoLeadIn() {
        let tiles = TemporalTiling.tiles(latentFrames: 16, maxTileFrames: 32)
        #expect(tiles.count == 1)
        #expect(tiles[0] == TemporalTile(start: 0, endExclusive: 16, dropPrefix: 0))
    }

    @Test(arguments: [(31, 12, 2), (61, 16, 2), (121, 24, 3), (100, 10, 1), (17, 16, 4)])
    func ownedRangesTileTheCanvasExactly(_ frames: Int, _ maxTile: Int, _ lead: Int) {
        let tiles = TemporalTiling.tiles(
            latentFrames: frames, maxTileFrames: maxTile, leadFrames: lead)
        // No gap, no double ownership, nothing past the end.
        #expect(TemporalTiling.ownedRangesCover(tiles, latentFrames: frames),
                "frames \(frames) tile \(maxTile) lead \(lead): \(tiles)")
        for tile in tiles {
            #expect(tile.length <= maxTile, "tile \(tile) exceeds the budget")
            #expect(tile.endExclusive <= frames)
            #expect(tile.dropPrefix < tile.length)
        }
        #expect(tiles[0].dropPrefix == 0, "the first tile owns its start")
    }

    @Test func everyLaterTileOverlapsItsPredecessor() {
        let tiles = TemporalTiling.tiles(latentFrames: 61, maxTileFrames: 16, leadFrames: 2)
        #expect(tiles.count > 1)
        for (previous, tile) in zip(tiles, tiles.dropFirst()) {
            // The lead-in must fall inside what the previous tile already denoised,
            // otherwise the seam is a hard cut rather than a shared boundary.
            #expect(tile.start < previous.endExclusive)
            #expect(tile.dropPrefix > 0)
            #expect(tile.start + tile.dropPrefix == previous.endExclusive)
        }
    }

    @Test func leadInIsClampedNotAssumed() {
        // A lead longer than what precedes it must clamp rather than index
        // before the start of the canvas.
        let tiles = TemporalTiling.tiles(latentFrames: 20, maxTileFrames: 6, leadFrames: 10)
        #expect(TemporalTiling.ownedRangesCover(tiles, latentFrames: 20))
        for tile in tiles { #expect(tile.start >= 0) }
    }
}
