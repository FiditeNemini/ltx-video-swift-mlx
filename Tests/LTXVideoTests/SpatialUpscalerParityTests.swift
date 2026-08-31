// SpatialUpscalerParityTests.swift — element-wise parity against the reference latent upsampler
// Copyright 2026
//
// Sub-task 6 of issue #57's breakdown (sub-tasks 1-5: video VAE decoder/encoder,
// text connector, audio VAE + vocoder, dual-stream transformer — PRs #76, #77,
// #78, #80, #82). `SpatialUpscaler` feeds the second stage of every two-stage
// generation, LipDub, and the IC-LoRA v2v path — every generation this repo
// ships that isn't single-stage dev.
//
// Ground truth comes from Lightricks' own `LatentUpsampler` (config-driven:
// spatial_upsample/temporal_upsample/rational_resampler read from the
// checkpoint), run on CPU float32 — dumped by scripts/spatial_upscaler_reference.py.
// The resampler is exactly where a dimension-order bug (which axis pairs with
// which pixel-shuffle factor) can be wrong without the output *shape* changing
// at all — sub-task 1's decoder padding bug (PR #76) was the same class of
// silent-until-measured mistake one component over.
//
//   PYTHONPATH=<ltx-core>/src python3 scripts/spatial_upscaler_reference.py \
//     <ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors> ref.safetensors
//   TEST_RUNNER_LTX25_SPATIAL_UPSCALER=<path-to-checkpoint> \
//   TEST_RUNNER_LTX25_SPATIAL_UPSCALER_REF=ref.safetensors \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/SpatialUpscalerParityTests

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

@Suite("Spatial upscaler parity vs the reference implementation",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_SPATIAL_UPSCALER_REF"] != nil))
struct SpatialUpscalerParityTests {

    func env(_ key: String) -> String { ProcessInfo.processInfo.environment[key]! }

    func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
        let scale = MLX.abs(b.asType(.float32)).mean().item(Float.self)
        return scale > 0 ? diff / scale : diff
    }

    /// `loadSpatialUpscaler` only *logs* missing/unmatched keys (behind
    /// `LTXDebug.isEnabled`), never throws — a broken key mapping would leave
    /// a parameter at random init and every parity number below would
    /// compare against nothing, with no failure to say so. Redone here with
    /// an explicit coverage check instead of delegating to it.
    func loadUpscaler() throws -> SpatialUpscaler {
        let raw = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_SPATIAL_UPSCALER")))
        var midChannels = 1024
        if let sample = raw["res_blocks.0.conv1.weight"] {
            midChannels = sample.dim(0)
        }
        let model = SpatialUpscaler(inChannels: 128, midChannels: midChannels, numBlocksPerStage: 4)

        var sanitized: [String: MLXArray] = [:]
        for (key, value) in raw {
            var newKey = key
            var newValue = value
            if newKey.hasPrefix("upsampler.0.") {
                newKey = newKey.replacingOccurrences(of: "upsampler.0.", with: "upsampler.conv.")
            }
            if newKey.contains("conv") && newKey.hasSuffix(".weight") && value.ndim == 5 {
                newValue = value.transposed(0, 2, 3, 4, 1)
            } else if newKey.contains("conv") && newKey.hasSuffix(".weight") && value.ndim == 4 {
                newValue = value.transposed(0, 2, 3, 1)
            }
            if newKey.contains("blur_down") { continue }
            sanitized[newKey] = newValue.asType(.float32)
        }

        let declared = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let missing = declared.keys.filter { sanitized[$0] == nil }.sorted()
        guard missing.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "SpatialUpscalerParityTests: \(missing.count) parameters got no checkpoint value, "
                + "e.g. \(missing.prefix(5).joined(separator: ", "))")
        }
        model.update(parameters: ModuleParameters.unflattened(sanitized))
        MLX.eval(model.parameters())
        return model
    }

    /// Walks initial_conv, each pre-upsample res block, the resampler, each
    /// post-upsample res block — reporting where the port first departs from
    /// the reference. Intermediates stay in the port's native NDHWC layout;
    /// the reference's NCDHW/NCHW taps are transposed to match before
    /// comparing, never the other way, so a wrong transpose in this test
    /// couldn't accidentally cancel a wrong transpose in the port.
    @Test func bisectFirstDivergence() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_SPATIAL_UPSCALER_REF")))
        let model = try loadUpscaler()
        let latent = reference["latent"]!.asType(.float32)  // (B, C, F, H, W), NCDHW

        func checkTap5D(_ name: String, _ oursNDHWC: MLXArray) {
            guard let refNCDHW = reference[name]?.asType(.float32) else { return }
            let ref = refNCDHW.transposed(0, 2, 3, 4, 1)  // -> NDHWC
            MLX.eval(oursNDHWC)
            #expect(oursNDHWC.shape == ref.shape, "\(name) shape \(oursNDHWC.shape) vs ref \(ref.shape)")
            guard oursNDHWC.shape == ref.shape else { return }
            let err = relativeError(oursNDHWC, ref)
            print("PARITY \(name): relative error \(err)")
            #expect(err < 0.02, "\(name) diverges from the reference: \(err)")
        }

        var h = latent.transposed(0, 2, 3, 4, 1)  // NCDHW -> NDHWC
        h = model.initialConv(h)
        checkTap5D("initial_conv", h)  // reference hooks the bare Conv3d, before norm+SiLU
        h = model.initialNorm(h)
        h = MLXNN.silu(h)

        for (i, block) in model.resBlocks.enumerated() {
            h = block(h)
            checkTap5D("res_block\(i)", h)
        }

        h = model.upsampler(h)  // (N, D, H*2, W*2, C), NDHWC
        // Reference taps the resampler per-frame, (B*F, C, H*2, W*2) NCHW —
        // fold N and D together and transpose to NHWC to match.
        let n = h.dim(0), d = h.dim(1), hh = h.dim(2), ww = h.dim(3), c = h.dim(4)
        let upsamplerPerFrame = h.reshaped([n * d, hh, ww, c])
        if let refUpsampler = reference["upsampler"]?.asType(.float32) {
            let ref = refUpsampler.transposed(0, 2, 3, 1)  // NCHW -> NHWC
            MLX.eval(upsamplerPerFrame)
            #expect(upsamplerPerFrame.shape == ref.shape,
                    "upsampler shape \(upsamplerPerFrame.shape) vs ref \(ref.shape)")
            if upsamplerPerFrame.shape == ref.shape {
                let err = relativeError(upsamplerPerFrame, ref)
                print("PARITY upsampler: relative error \(err)")
                #expect(err < 0.02, "upsampler diverges from the reference: \(err)")
            }
        }

        for (i, block) in model.postResBlocks.enumerated() {
            h = block(h)
            checkTap5D("post_res_block\(i)", h)
        }
    }

    @Test func outputMatchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_SPATIAL_UPSCALER_REF")))
        let model = try loadUpscaler()
        let latent = reference["latent"]!.asType(.float32)

        let output = model(latent)
        MLX.eval(output)
        let refOutput = reference["output"]!.asType(.float32)
        #expect(output.shape == refOutput.shape, "output \(output.shape) vs ref \(refOutput.shape)")
        let err = relativeError(output, refOutput)
        print("PARITY output relative error: \(err)")
        #expect(err < 0.02, "spatial upscaler diverges from the reference: \(err)")
    }
}
