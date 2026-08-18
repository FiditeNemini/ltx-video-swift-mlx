// DiffVAEDecoderTests.swift — diffusion decoder: shapes, patch round trip, real load
// Copyright 2026
//
// The gated part needs the LTX-2.5 video VAE bundle:
//   TEST_RUNNER_LTX25_DIFFVAE=/Volumes/Lexar/models/ltx-2.5-cache/ltx-2.5-distilled/ltx-2.5-video-vae-bf16.safetensors \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/DiffVAEDecoderTests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Diffusion VAE decoder")
struct DiffVAEDecoderTests {

    static var checkpointPath: String? {
        ProcessInfo.processInfo.environment["LTX25_DIFFVAE"]
    }

    @Test func patchifyRoundTrips() {
        // patchify/unpatchify must be exact inverses, including the channel
        // interleave order — a transposed pair here would scramble pixels in a
        // way that still "looks like video", which is the worst kind of bug.
        let x = MLXArray(0 ..< (1 * 3 * 2 * 8 * 12)).reshaped([1, 3, 2, 8, 12]).asType(.float32)
        let packed = DiffusionVideoDecoder.patchify(x, patch: 4)
        #expect(packed.shape == [1, 48, 2, 2, 3])
        let back = DiffusionVideoDecoder.unpatchify(packed, patch: 4)
        MLX.eval(back)
        #expect(back.shape == x.shape)
        #expect(MLX.abs(back - x).max().item(Float.self) == 0)
    }

    @Test func upsampleGeometryComposesTo32x8() {
        // The four upsamples must compose to the VAE's compression: ×8 spatial
        // here plus the ×4 patch at the end = ×32, and ×8 temporal.
        var spatial = 1, temporal = 1
        for entry in DiffVAEConfig.upsampleStrides {
            temporal *= entry.stride.0
            spatial *= entry.stride.1
        }
        #expect(temporal == 8)
        #expect(spatial * 4 == 32)
    }

    @Test(.enabled(if: checkpointPath != nil))
    func loadsRealCheckpointAndDecodes() throws {
        let path = Self.checkpointPath!
        #expect(DiffVAEWeightLoader.isDiffusionVAE(path: path))

        let decoder = try DiffVAEWeightLoader.load(from: path)
        // Config as declared by the shipped 2.5 bundle.
        #expect(decoder.config.stageChannels == [2048, 1024, 512, 512, 256])
        #expect(decoder.config.stageDepths == [4, 6, 4, 2, 8])
        #expect(decoder.config.patchSize == 4)
        #expect(decoder.config.numInferenceSteps == 1)
        #expect(decoder.config.predictsX0)

        // Smallest legal volume: one latent frame at 64×64 px worth of latent.
        // Kernels are floored to the volume, so this exercises the border paths.
        let latent = MLXRandom.normal([1, 128, 1, 2, 2]).asType(.float32)
        let frames = decoder.decode(latent: latent, seed: 42)
        MLX.eval(frames)

        // [F, H, W, C]: one latent frame → one pixel frame, 2 latent cells → 64 px.
        #expect(frames.dim(1) == 64)
        #expect(frames.dim(2) == 64)
        #expect(frames.dim(3) == 3)
        let minV = frames.min().item(Float.self), maxV = frames.max().item(Float.self)
        #expect(minV >= 0 && maxV <= 1, "pixels must land in [0, 1], got [\(minV), \(maxV)]")
        #expect(maxV > minV, "a constant frame means the decoder produced nothing")
    }
}
