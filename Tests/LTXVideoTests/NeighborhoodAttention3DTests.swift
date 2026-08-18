// NeighborhoodAttention3DTests.swift — the tiled NA3D must equal a brute-force one
// Copyright 2026
//
// The tiling in NeighborhoodAttention3D exists only to bound memory; it must not
// change a single output value. These tests pin that against an independent
// brute-force implementation (build the full [N, N] mask, one attention call),
// on volumes small enough for the brute force to be tractable.

import Foundation
import Testing
@preconcurrency import MLX
import MLXFast
@testable import LTXVideo

@Suite("Neighborhood attention 3D")
struct NeighborhoodAttention3DTests {

    /// Full-volume reference: one mask over every (query, key) pair.
    static func bruteForce(
        q: MLXArray, k: MLXArray, v: MLXArray,
        dims: (Int, Int, Int), kernels: (Int, Int, Int),
        causalTime: Bool, scale: Float
    ) -> MLXArray {
        let (T, H, W) = dims
        let bounds = [
            NeighborhoodAttention3D.windowBounds(
                length: T, kernel: min(kernels.0, T), causal: causalTime),
            NeighborhoodAttention3D.windowBounds(length: H, kernel: min(kernels.1, H), causal: false),
            NeighborhoodAttention3D.windowBounds(length: W, kernel: min(kernels.2, W), causal: false),
        ]
        let mask = NeighborhoodAttention3D.groupMask(
            bounds: bounds, keyLengths: [T, H, W], dtype: q.dtype)
        let B = q.dim(0), heads = q.dim(1), headDim = q.dim(5)
        let n = T * H * W
        let out = MLXFast.scaledDotProductAttention(
            queries: q.reshaped([B, heads, n, headDim]),
            keys: k.reshaped([B, heads, n, headDim]),
            values: v.reshaped([B, heads, n, headDim]),
            scale: scale, mask: mask)
        return out.reshaped([B, heads, T, H, W, headDim])
    }

    static func randomQKV(_ dims: (Int, Int, Int), heads: Int = 2, headDim: Int = 8)
        -> (MLXArray, MLXArray, MLXArray)
    {
        MLXRandom.seed(7)
        let shape = [1, heads, dims.0, dims.1, dims.2, headDim]
        return (MLXRandom.normal(shape), MLXRandom.normal(shape), MLXRandom.normal(shape))
    }

    @Test(arguments: [
        // (dims, kernels, causalTime) — interior, borders, kernel wider than axis
        ((4, 6, 6), (3, 3, 3), false),
        ((3, 5, 7), (3, 5, 5), false),
        ((5, 4, 4), (3, 3, 3), true),
        ((1, 5, 5), (3, 3, 3), false),   // single frame: kernel floor on T
        ((4, 3, 3), (3, 7, 7), false),   // kernel wider than H and W
    ])
    func tiledMatchesBruteForce(
        _ dims: (Int, Int, Int), _ kernels: (Int, Int, Int), _ causal: Bool
    ) {
        let (q, k, v) = Self.randomQKV(dims)
        let scale: Float = 0.35
        let tiled = NeighborhoodAttention3D.callAsFunction(
            q: q, k: k, v: v, dims: dims, kernels: kernels, causalTime: causal, scale: scale)
        let reference = Self.bruteForce(
            q: q, k: k, v: v, dims: dims, kernels: kernels, causalTime: causal, scale: scale)
        MLX.eval(tiled, reference)
        let maxDiff = MLX.abs(tiled - reference).max().item(Float.self)
        #expect(maxDiff < 2e-5, "dims \(dims) kernels \(kernels) causal \(causal): Δ=\(maxDiff)")
    }

    @Test func tilingActuallySplitsWhenVolumeIsLarge() {
        // A volume whose full score matrix would blow the budget must be tiled;
        // if this ever returns the whole volume, the memory guarantee is gone.
        let tiles = NeighborhoodAttention3D.pickTiles(
            dims: (32, 256, 256), kernels: (3, 7, 7))
        let nq = tiles.0 * tiles.1 * tiles.2
        #expect(nq < 32 * 256 * 256)
        #expect(nq > 0)
    }

    @Test func windowsAreClampedNotTruncated() {
        // Every non-causal query must see exactly `kernel` keys, including at
        // the borders — that clamping is what the weights were trained with.
        let (starts, ends) = NeighborhoodAttention3D.windowBounds(
            length: 8, kernel: 5, causal: false)
        for i in 0 ..< 8 {
            #expect(ends[i] - starts[i] == 5, "index \(i)")
            #expect(starts[i] >= 0 && ends[i] <= 8)
        }
        // Causal: trailing window, shorter only at the very start.
        let (cs, ce) = NeighborhoodAttention3D.windowBounds(length: 6, kernel: 3, causal: true)
        #expect(ce[0] - cs[0] == 1)
        #expect(ce[5] - cs[5] == 3)
        #expect(ce[5] == 6)
    }
}
