// AudioVAEVocoderParityTests.swift — element-wise parity against the reference audio decode chain
// Copyright 2026
//
// The real production audio decode path (ltx_core's own `decode_audio`):
// audio latent -> AudioVAE.decode -> mel spectrogram -> LTXVocoderWithBWE ->
// 48 kHz waveform. Sub-task 4 of issue #57's breakdown (sub-tasks 1-3: video
// VAE decoder/encoder, text connector — PRs #76, #77, #78).
//
// This is the path this repo's own worst historically-shipped bugs lived on
// (docs/knowledge/pitfalls/wrong-vocoder-lost-the-top-octave.md,
// audio-must-stay-stereo.md) — both found by ear, not by a harness. This is
// the first element-wise reference for it.
//
// Ground truth comes from Lightricks' own AudioDecoder + VocoderWithBWE, run
// on CPU float32 (the vocoder's own docstring: bf16 accumulation across ~108
// sequential convolutions degrades spectral metrics by 40-90%) over a fixed
// latent — dumped by scripts/audio_vae_reference.py.
//
//   TEST_RUNNER_LTX25_AUDIOVAE=<ltx-2.5-audio-vae-bf16.safetensors> \
//   TEST_RUNNER_LTX25_AUDIOVAE_REF=<audio_vae_reference.safetensors> \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/AudioVAEVocoderParityTests

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

@Suite("Audio VAE + vocoder parity vs the reference implementation",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_AUDIOVAE_REF"] != nil),
       .serialized)
struct AudioVAEVocoderParityTests {

    func env(_ key: String) -> String { ProcessInfo.processInfo.environment[key]! }

    func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
        let scale = MLX.abs(b.asType(.float32)).mean().item(Float.self)
        return scale > 0 ? diff / scale : diff
    }

    /// MLXNN's no-verify: update() silently no-ops on keys the model doesn't
    /// declare a parameter for — a broken key mapping would leave a
    /// parameter at random init and every parity number below would compare
    /// against nothing, with no failure to say so.
    func loadAudioVAE() throws -> AudioVAE {
        let vae = AudioVAE()
        let weights = try LTXWeightLoader.loadAudioVAEWeights(from: env("LTX25_AUDIOVAE"))
        let declared = Dictionary(uniqueKeysWithValues: vae.parameters().flattened())
        let missing = declared.keys.filter { weights[$0] == nil }.sorted()
        guard missing.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "AudioVAEVocoderParityTests: \(missing.count) decoder parameters got no checkpoint "
                + "value, e.g. \(missing.prefix(5).joined(separator: ", "))")
        }
        try LTXWeightLoader.applyAudioVAEWeights(weights, to: vae)
        return vae
    }

    /// BigVGANWeightLoader.load already throws on any unfed parameter or
    /// shape mismatch (see BigVGANVocoderE2ETests) — no separate coverage
    /// check needed here.
    func loadVocoder() throws -> LTXVocoderWithBWE {
        try BigVGANWeightLoader.load(from: URL(fileURLWithPath: env("LTX25_AUDIOVAE")))
    }

    /// Replicates AudioVAE.decode()'s denormalize + patchify prelude, so the
    /// bisection test can tap the decoder's internal stages directly instead
    /// of only the black-box decode() result.
    func denormalizedLatent(_ vae: AudioVAE, _ latent: MLXArray) -> MLXArray {
        let b = latent.dim(0), latentT = latent.dim(2), latentMel = latent.dim(3)
        var sample = latent.transposed(0, 2, 1, 3).reshaped([b, latentT, -1]).asType(DType.float32)
        let mean = vae.latentsMean.asType(DType.float32).reshaped([1, 1, 128])
        let std = vae.latentsStd.asType(DType.float32).reshaped([1, 1, 128])
        sample = sample * std + mean
        return sample.reshaped([b, latentT, 8, latentMel]).transposed(0, 2, 1, 3)
    }

    /// Walks the decoder's stages (conv_in, mid, each up-level, final) and the
    /// vocoder's two generators (base + BWE residual), reporting where the
    /// port first departs from the reference. The skip connection (a
    /// resampler derived in Swift, not loaded — see
    /// docs/knowledge/pitfalls/wrong-vocoder-lost-the-top-octave.md) is
    /// `private` and not independently tapped, but its correctness is implied
    /// by the base + residual + final-waveform checks together: final =
    /// residual + skip, and both final and residual are verified.
    @Test func bisectFirstDivergence() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_AUDIOVAE_REF")))
        let vae = try loadAudioVAE()
        let vocoder = try loadVocoder()
        let latent = reference["latent"]!.asType(DType.float32)

        func checkTap(_ name: String, _ ours: MLXArray) {
            guard let ref = reference[name]?.asType(DType.float32) else { return }
            MLX.eval(ours)
            #expect(ours.shape == ref.shape, "\(name) shape \(ours.shape) vs ref \(ref.shape)")
            guard ours.shape == ref.shape else { return }
            let err = relativeError(ours, ref)
            print("PARITY \(name): relative error \(err)")
            #expect(err < 0.02, "\(name) diverges from the reference: \(err)")
        }

        var h = denormalizedLatent(vae, latent)
        let decoder = vae.decoder
        h = decoder.convIn(h)
        checkTap("decoder_conv_in", h)
        h = decoder.mid(h)
        checkTap("decoder_mid", h)
        h = decoder.upLevels[2](h)
        checkTap("decoder_up2", h)
        h = decoder.upLevels[1](h)
        checkTap("decoder_up1", h)
        h = decoder.upLevels[0](h)
        checkTap("decoder_up0", h)
        h = decoder.normOut(h)
        h = MLXNN.silu(h)
        h = decoder.convOut(h)
        checkTap("decoder_out", h)

        let mel = h.asType(DType.float32)
        // Base generator: BigVGANGenerator returns (B, T, 2) — transpose to
        // the reference's (B, 2, T) before comparing.
        let low = vocoder.vocoder(mel).transposed(0, 2, 1)
        checkTap("vocoder_base", low)

        // Replicate LTXVocoderWithBWE.callAsFunction's pad + mel-recompute
        // prelude to reach the BWE generator's own input.
        var lowNLC = low.transposed(0, 2, 1)  // back to (B, T, 2) for the pad/reshape below
        let lowLength = lowNLC.dim(1)
        let remainder = lowLength % 80
        if remainder != 0 {
            lowNLC = MLX.padded(lowNLC, widths: [.init((0, 0)), .init((0, 80 - remainder)), .init((0, 0))])
        }
        let batch = lowNLC.dim(0)
        let channels = lowNLC.dim(2)
        let flat = lowNLC.transposed(0, 2, 1).reshaped([batch * channels, -1])
        let melBWE = vocoder.melSTFT.logMel(flat)
        let melForBWE = melBWE.reshaped([batch, channels, melBWE.dim(1), melBWE.dim(2)])
            .transposed(0, 1, 3, 2)
        checkTap("vocoder_mel_for_bwe", melForBWE)
        let residual = vocoder.bweGenerator(melForBWE).transposed(0, 2, 1)
        checkTap("vocoder_bwe_residual", residual)
    }

    @Test func melOutputMatchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_AUDIOVAE_REF")))
        let vae = try loadAudioVAE()
        let latent = reference["latent"]!.asType(DType.float32)

        let mel = vae.decode(latent)
        MLX.eval(mel)
        let refMel = reference["mel"]!.asType(DType.float32)
        #expect(mel.shape == refMel.shape, "mel \(mel.shape) vs ref \(refMel.shape)")
        let err = relativeError(mel, refMel)
        print("PARITY mel relative error: \(err)")
        #expect(err < 0.02, "AudioVAE.decode diverges from the reference: \(err)")
    }

    @Test func waveformOutputMatchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_AUDIOVAE_REF")))
        let vae = try loadAudioVAE()
        let vocoder = try loadVocoder()
        let latent = reference["latent"]!.asType(DType.float32)

        let mel = vae.decode(latent)
        let waveform = vocoder(mel)  // LTXVocoderWithBWE already returns (B, 2, T)
        MLX.eval(waveform)
        let refWaveform = reference["waveform"]!.asType(DType.float32)
        #expect(waveform.shape == refWaveform.shape, "waveform \(waveform.shape) vs ref \(refWaveform.shape)")
        let err = relativeError(waveform, refWaveform)
        print("PARITY waveform relative error: \(err)")
        #expect(err < 0.02, "vocoder chain diverges from the reference: \(err)")
    }
}
