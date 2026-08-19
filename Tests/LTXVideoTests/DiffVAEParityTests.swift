// DiffVAEParityTests.swift — element-wise parity against the reference decoder
// Copyright 2026
//
// Ground truth comes from Lightricks' own DiffusionVideoDecoder, run on CPU
// float32 over a fixed latent with its eager NA fallback (the same algorithm
// this port implements), dumped by scripts/diffvae_reference.py.
//
//   TEST_RUNNER_LTX25_DIFFVAE=<video-vae-bf16.safetensors> \
//   TEST_RUNNER_LTX25_DIFFVAE_REF=<diffvae_reference.safetensors> \
//   xcodebuild ... -only-testing:LTXVideoTests/DiffVAEParityTests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("DiffVAE parity vs the reference implementation",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_DIFFVAE_REF"] != nil))
struct DiffVAEParityTests {

    func env(_ key: String) -> String { ProcessInfo.processInfo.environment[key]! }

    /// Relative error, the scale-free measure that stays meaningful whether the
    /// tensor lives at 0.05 or 5.
    func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
        let scale = MLX.abs(b.asType(.float32)).mean().item(Float.self)
        return scale > 0 ? diff / scale : diff
    }

    /// Walk the first blocks and report where the port first departs from the
    /// reference, so a failure names the layer instead of the whole decoder.
    @Test func bisectFirstDivergence() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_DIFFVAE_REF")))
        let decoder = try DiffVAEWeightLoader.load(from: env("LTX25_DIFFVAE"))
        let latent = reference["latent"]!.asType(DType.float32)

        let mean = decoder.meanOfMeans.reshaped([1, -1, 1, 1, 1]).asType(DType.float32)
        let std = decoder.stdOfMeans.reshaped([1, -1, 1, 1, 1]).asType(DType.float32)
        let z = latent * std + mean

        var taps: [(String, MLXArray)] = []
        let convIn = decoder.convIn(z.transposed(0, 2, 3, 4, 1))
        taps.append(("conv_in", convIn))

        let block0 = decoder.detStages[0][0]
        let n1 = block0.norm1(convIn)
        taps.append(("b0_norm1", n1))
        let a0 = block0.attn(n1)
        taps.append(("b0_attn", a0))
        let out0 = block0(convIn)
        taps.append(("b0_out", out0))
        var h = decoder.detStages[0][1](out0)
        taps.append(("b1_out", h))
        for stage in 0 ..< 4 {
            let from = stage == 0 ? 2 : 0
            for i in from ..< decoder.detStages[stage].count { h = decoder.detStages[stage][i](h) }
            taps.append(("stage\(stage)_last", h))
            h = decoder.upsamples[stage](h, dropLeadingFrame: true)
            taps.append(("up\(stage)_out", h))
        }

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

    @Test func contextAndPredictionMatchReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_DIFFVAE_REF")))
        let decoder = try DiffVAEWeightLoader.load(from: env("LTX25_DIFFVAE"))

        let latent = reference["latent"]!.asType(DType.float32)
        let refContext = reference["context"]!.asType(DType.float32)
        let refPrediction = reference["prediction"]!.asType(DType.float32)

        // Stage 1-4: the deterministic context volume.
        let context = decoder.context(from: latent)
        MLX.eval(context)
        #expect(context.shape == refContext.shape,
                "context \(context.shape) vs reference \(refContext.shape)")
        let contextError = relativeError(context, refContext)
        print("PARITY context relative error: \(contextError)")
        #expect(contextError < 0.02, "deterministic stages diverge: \(contextError)")

        // Stage 5: one diffusion step from a zero x_t, as the reference dumped.
        let xT = MLXArray.zeros(
            [1, 3, context.dim(1), context.dim(2) * 4, context.dim(3) * 4], dtype: DType.float32)
        let prediction = decoder.diffusionStep(
            context: context, xT: xT, t: MLXArray([Float(1.0)]))
        MLX.eval(prediction)
        #expect(prediction.shape == refPrediction.shape)
        let predictionError = relativeError(prediction, refPrediction)
        print("PARITY prediction relative error: \(predictionError)")
        #expect(predictionError < 0.02, "diffusion stage diverges: \(predictionError)")
    }
}
