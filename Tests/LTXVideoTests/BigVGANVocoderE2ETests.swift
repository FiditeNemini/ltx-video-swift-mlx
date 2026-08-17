// BigVGANVocoderE2ETests.swift — Gated checks for the checkpoint's real vocoder
// Copyright 2026
//
// The vocoder LTX-2.3 and LTX-2.5 ship (BigVGAN v2 + bandwidth extension, 667 +
// 557 tensors) shares no key with the LTX-2-era one this package used to load,
// so "it produced audio" proves nothing. These tests pin the structural
// contract: every declared parameter is fed, shapes agree, and the chain lifts
// 16 kHz to 48 kHz with the right sample count.
//
// Gated behind LTX25_MODELS_DIR.
//
// Run:
//   TEST_RUNNER_LTX25_MODELS_DIR=/Volumes/Lexar/models/ltx-2.5 \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/BigVGANVocoderE2ETests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("LTX BigVGAN vocoder (gated: LTX25_MODELS_DIR)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] != nil),
       .serialized)
struct BigVGANVocoderE2ETests {

    static var audioVAEURL: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] ?? "")
            .appendingPathComponent("ltx-2.5-audio-vae-bf16.safetensors")
    }

    @Test func loadsEveryParameterFromTheCheckpoint() throws {
        // Throws if any parameter is unfed or any shape disagrees, which is the
        // whole point: a partially-loaded vocoder emits plausible noise.
        let vocoder = try BigVGANWeightLoader.load(from: Self.audioVAEURL)

        #expect(vocoder.inputSampleRate == 16000)
        #expect(vocoder.outputSampleRate == 48000)

        // Spot-check the three trees really landed, with the shapes the config implies.
        let params = Dictionary(uniqueKeysWithValues: vocoder.parameters().flattened())
        #expect(params["vocoder.conv_pre.weight"]?.shape == [1536, 7, 128])
        #expect(params["bwe_generator.conv_pre.weight"]?.shape == [512, 7, 128])
        #expect(params["mel_stft.mel_basis"]?.shape == [64, 257])
        // Snake parameters are per-channel and stored in log scale.
        #expect(params["vocoder.act_post.act.alpha"]?.shape == [24])
    }

    /// The LTX-2-era file must be refused rather than silently half-loaded.
    @Test func refusesTheOldVocoderFile() throws {
        let legacy = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/ltx-video-mlx/ltx-vocoder/diffusion_pytorch_model.safetensors")
        try #require(FileManager.default.fileExists(atPath: legacy.path),
                     "legacy vocoder not cached; skipping")
        #expect(throws: LTXError.self) { _ = try BigVGANWeightLoader.load(from: legacy) }
    }

    @Test func liftsSixteenKilohertzToFortyEight() throws {
        let vocoder = try BigVGANWeightLoader.load(from: Self.audioVAEURL)

        // 8 latent mel frames of stereo 64-bin input. The base generator upsamples
        // by 5x2x2x2x2x2 = 160 — one sample per 10 ms hop at 16 kHz, matching the
        // audio VAE's hop length — and the BWE stage then triples the rate.
        let frames = 8
        let mel = MLXRandom.normal([1, 2, frames, 64]) * 0.1
        let waveform = vocoder(mel)
        MLX.eval(waveform)

        #expect(waveform.dim(0) == 1)
        #expect(waveform.dim(1) == 2, "stereo out")
        #expect(waveform.dim(2) == frames * 160 * 3)

        // Finite and inside the clamp the final stage applies.
        let peak = MLX.abs(waveform).max().item(Float.self)
        #expect(peak.isFinite && peak <= 1.0)
    }
}
