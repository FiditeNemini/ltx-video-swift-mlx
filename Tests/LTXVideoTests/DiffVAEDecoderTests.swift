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
    func padsSmallVolumesToTheKernelFloor() throws {
        let decoder = try DiffVAEWeightLoader.load(from: Self.checkpointPath!)
        let minimum = decoder.minimumLatentShape
        // The shipped config's floor: stage 0 needs 7 spatially, the diffusion
        // stage's kernel of 11 needs 3 latent frames after the temporal chain.
        #expect(minimum.frames == 3)
        #expect(minimum.height == 7 && minimum.width == 7)

        // A latent below the floor must still decode, and to its *true* size —
        // the padding is cropped back off in pixel space.
        let latent = MLXRandom.normal([1, 128, 1, 2, 2]).asType(DType.float32)
        let frames = decoder.decode(latent: latent, seed: 1)
        MLX.eval(frames)
        #expect(frames.dim(0) == 1, "one latent frame stays one pixel frame")
        #expect(frames.dim(1) == 64 && frames.dim(2) == 64)
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

@Suite("DiffVAE activation probe (gated)", .enabled(if: ProcessInfo.processInfo.environment["LTX25_DIFFVAE"] != nil))
struct DiffVAEProbeTests {
    @Test func stageMagnitudes() throws {
        let decoder = try DiffVAEWeightLoader.load(
            from: ProcessInfo.processInfo.environment["LTX25_DIFFVAE"]!)
        let latent = MLXRandom.normal([1, 128, 2, 4, 4]).asType(.float32)

        func stats(_ label: String, _ x: MLXArray) {
            MLX.eval(x)
            let a = MLX.abs(x)
            print(String(format: "PROBE %@ shape %@ mean|x| %.4f max|x| %.4f std %.4f",
                         label, "\(x.shape)",
                         a.mean().item(Float.self), a.max().item(Float.self),
                         MLX.variance(x.asType(.float32)).sqrt().item(Float.self)))
        }

        MLX.eval(decoder.meanOfMeans, decoder.stdOfMeans)
        print(String(format: "PROBE stats mean[0..3] %.3f %.3f %.3f  std[0..3] %.3f %.3f %.3f",
            decoder.meanOfMeans[0].item(Float.self), decoder.meanOfMeans[1].item(Float.self),
            decoder.meanOfMeans[2].item(Float.self),
            decoder.stdOfMeans[0].item(Float.self), decoder.stdOfMeans[1].item(Float.self),
            decoder.stdOfMeans[2].item(Float.self)))
        stats("latent", latent)
        let ctx = decoder.context(from: latent)
        stats("context", ctx)

        let pixelShape = [1, 3, ctx.dim(1), ctx.dim(2) * 4, ctx.dim(3) * 4]
        MLXRandom.seed(42)
        let xT = MLXRandom.normal(pixelShape).asType(ctx.dtype)
        let out = decoder.diffusionStep(context: ctx, xT: xT, t: MLXArray([Float(1.0)]))
        stats("prediction", out)
    }
}
