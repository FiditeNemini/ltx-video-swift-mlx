// ConvVAEDecoderParityTests.swift — element-wise parity against the reference decoder
// Copyright 2026
//
// The convolutional decoder is the *default* decode path — every clip this
// repo has ever produced went through it (the diffusion decoder, already
// covered by DiffVAEParityTests, is opt-in via --diffvae), and until now
// nothing verified it element-wise against Lightricks' own implementation.
//
// Ground truth comes from Lightricks' own ConvVideoDecoder, run on CPU
// float32 over a fixed latent — deterministic, since the real checkpoint has
// timestep_conditioning=false and this decoder has no attention, so there is
// no RNG to match across frameworks — dumped by
// scripts/conv_video_decoder_reference.py.
//
//   TEST_RUNNER_LTX25_CONVVAE=<video-vae-conv-bf16.safetensors> \
//   TEST_RUNNER_LTX25_CONVVAE_REF=<conv_video_decoder_reference.safetensors> \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/ConvVAEDecoderParityTests

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

@Suite("Conv VAE decoder parity vs the reference implementation",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_CONVVAE_REF"] != nil))
struct ConvVAEDecoderParityTests {

    func env(_ key: String) -> String { ProcessInfo.processInfo.environment[key]! }

    /// Relative error, the scale-free measure that stays meaningful whether the
    /// tensor lives at 0.05 or 5.
    func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
        let scale = MLX.abs(b.asType(.float32)).mean().item(Float.self)
        return scale > 0 ? diff / scale : diff
    }

    func loadDecoder() throws -> VideoDecoder {
        let decoder = VideoDecoder()
        let mapped = try LTXWeightLoader.loadVAEWeights(from: env("LTX25_CONVVAE"))
        _ = decoder.update(parameters: ModuleParameters.unflattened(mapped))
        eval(decoder.parameters())
        return decoder
    }

    /// Walk every up-block and report where the port first departs from the
    /// reference, so a failure names the layer instead of the whole decoder.
    @Test func bisectFirstDivergence() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_CONVVAE_REF")))
        let decoder = try loadDecoder()
        let latent = reference["latent"]!.asType(DType.float32)

        let mean = decoder.meanOfMeans.reshaped([1, -1, 1, 1, 1]).asType(DType.float32)
        let std = decoder.stdOfMeans.reshaped([1, -1, 1, 1, 1]).asType(DType.float32)
        var x = latent * std + mean

        var taps: [(String, MLXArray)] = []
        x = decoder.convIn(x, causal: decoder.causal)
        taps.append(("conv_in", x))
        x = decoder.upBlocks0(x, causal: decoder.causal)
        taps.append(("up_blocks_0", x))
        x = decoder.upBlocks1(x, causal: decoder.causal)
        taps.append(("up_blocks_1", x))
        x = decoder.upBlocks2(x, causal: decoder.causal)
        taps.append(("up_blocks_2", x))
        x = decoder.upBlocks3(x, causal: decoder.causal)
        taps.append(("up_blocks_3", x))
        x = decoder.upBlocks4(x, causal: decoder.causal)
        taps.append(("up_blocks_4", x))
        x = decoder.upBlocks5(x, causal: decoder.causal)
        taps.append(("up_blocks_5", x))
        x = decoder.upBlocks6(x, causal: decoder.causal)
        taps.append(("up_blocks_6", x))
        x = decoder.upBlocks7(x, causal: decoder.causal)
        taps.append(("up_blocks_7", x))
        x = decoder.upBlocks8(x, causal: decoder.causal)
        taps.append(("up_blocks_8", x))
        x = vaePixelNorm(x)
        x = MLXNN.silu(x)
        x = decoder.convOut(x, causal: decoder.causal)
        taps.append(("conv_out", x))

        for (name, ours) in taps {
            guard let ref = reference[name]?.asType(DType.float32) else { continue }
            MLX.eval(ours)
            guard ours.shape == ref.shape else {
                print("PARITY \(name): SHAPE ours \(ours.shape) ref \(ref.shape)")
                #expect(Bool(false), "\(name) shape mismatch")
                continue
            }
            let err = relativeError(ours, ref)
            print("PARITY \(name): relative error \(err)")
        }
    }

    @Test func outputMatchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_CONVVAE_REF")))
        let decoder = try loadDecoder()

        let latent = reference["latent"]!.asType(DType.float32)
        let refOutput = reference["output"]!.asType(DType.float32)

        let output = decoder(latent)
        MLX.eval(output)
        #expect(output.shape == refOutput.shape,
                "output \(output.shape) vs reference \(refOutput.shape)")
        let err = relativeError(output, refOutput)
        print("PARITY output relative error: \(err)")
        #expect(err < 0.02, "conv VAE decoder diverges from the reference: \(err)")
    }
}
