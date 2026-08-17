// Gemma4TextEncoderE2ETests.swift — Gated E2E for the LTX-2.5 text encoder
// Copyright 2026
//
// Loads the real 24 GB `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` and runs a
// prompt through it. Nothing here can be faked: the point is to prove that the
// bundled config decodes, that every declared Gemma parameter is fed by the
// checkpoint, that the tokenizer round-trips out of its byte tensors, and that
// the 49 hidden states come out finite and correctly shaped.
//
// Gated behind LTX25_TE_PATH pointing at the file (~24 GB resident in bf16).
//
// Run:
//   LTX25_TE_PATH=/Volumes/Lexar/models/ltx-2.5/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Release test \
//     -only-testing:LTXVideoTests/Gemma4TextEncoderE2ETests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("LTX-2.5 Gemma 4 text encoder (gated: LTX25_TE_PATH)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_TE_PATH"] != nil),
       .serialized)
struct Gemma4TextEncoderE2ETests {

    static var encoderPath: String {
        ProcessInfo.processInfo.environment["LTX25_TE_PATH"] ?? ""
    }

    static var tokenizerCache: URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx25-gemma4-tokenizer", isDirectory: true)
    }

    @Test func bundleDeclaresTheLTXGemmaDerivative() throws {
        let assets = try LTX25TextEncoderAssets(fileURL: URL(fileURLWithPath: Self.encoderPath))

        #expect(assets.gemmaVersion == "gemma4-12b-ltx-v1")

        let config = try assets.textConfig()
        #expect(config.numHiddenLayers == 48)
        #expect(config.hiddenSize == 3840)

        // The transformer names the encoder it was trained with; a mismatch is fatal.
        #expect(throws: Never.self) {
            try assets.verifyPairing(withTransformerMetadata: [
                "gemma_source_checkpoint": #"{"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"}"#
            ])
        }
        #expect(throws: LTXError.self) {
            try assets.verifyPairing(withTransformerMetadata: [
                "gemma_source_checkpoint": #"{"ltx_version": "2.5.0", "gemma_version": "gemma3-12b"}"#
            ])
        }
        // A 2.3-era checkpoint declares no pairing — nothing to check, not an error.
        #expect(throws: Never.self) {
            try assets.verifyPairing(withTransformerMetadata: ["model_version": "2.3.0"])
        }
    }

    @Test func projectionsMatchTheLTX23Shapes() throws {
        let assets = try LTX25TextEncoderAssets(fileURL: URL(fileURLWithPath: Self.encoderPath))
        let projections = assets.projectionWeights()

        // 188160 = 49 hidden states x 3840 — unchanged from LTX-2.3, which is why
        // the Feature-Extractor-V2 path carries over untouched.
        let video = try #require(projections["feature_extractor.video_aggregate_embed.weight"])
        #expect(video.shape == [4096, 188160])
        let audio = try #require(projections["feature_extractor.audio_aggregate_embed.weight"])
        #expect(audio.shape == [2048, 188160])
    }

    /// Shapes and finiteness would survive a mis-mapped checkpoint — a permuted
    /// layer order or a swapped q/k projection still yields finite numbers. Meaning
    /// does not: a broken stack cannot rank a paraphrase above an unrelated
    /// sentence. This is the strongest self-contained check available here;
    /// elementwise parity against the PyTorch reference is still open.
    ///
    /// Note on what is deliberately *not* tested: greedy decoding. This checkpoint
    /// is an encoder fine-tune with tied embeddings, and its final-norm scale has
    /// drifted far above stock Gemma 4's (mean 20 vs 7.9, max 600 vs 14). The
    /// hidden state that reaches the tied head therefore lands deep in the
    /// `final_logit_softcapping` saturation zone and the head ranks by embedding
    /// magnitude, emitting single capital letters. Harmless — LTX never reads
    /// logits, and the feature extractor per-token RMS-normalises the states,
    /// which is exactly why the scale was free to drift during LTX's training.
    @Test func encodesMeaningNotJustNumbers() async throws {
        let encoder = try await Gemma4TextEncoder.load(
            fileURL: URL(fileURLWithPath: Self.encoderPath),
            tokenizerCacheDirectory: Self.tokenizerCache
        )

        func pooled(_ prompt: String) throws -> MLXArray {
            let (states, mask) = try encoder.encode(prompt: prompt, maxLength: 64)
            let last = states[states.count - 1].asType(.float32)
            let weights = mask.asType(.float32).expandedDimensions(axis: -1)
            let summed = (last * weights).sum(axis: 1)
            let vector = summed / weights.sum(axis: 1)
            return vector / MLX.sqrt((vector * vector).sum(axis: -1, keepDims: true))
        }
        func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
            let value = (a * b).sum()
            MLX.eval(value)
            return value.item(Float.self)
        }

        let reference = try pooled("a dog running through a sunny park")
        let paraphrase = try pooled("a puppy runs across a sunlit park")
        let unrelated = try pooled("quantum chromodynamics equations on a blackboard")

        let near = cosine(reference, paraphrase)
        let far = cosine(reference, unrelated)
        #expect(near > far,
                "paraphrase similarity \(near) should exceed unrelated \(far) — the stack is mis-mapped")

        // Encoding is deterministic: the same prompt gives the same vector.
        #expect(cosine(reference, try pooled("a dog running through a sunny park")) > 0.999)
    }

    /// The stack must stay numerically stable across all 48 layers: a scale that
    /// explodes or collapses is the signature of a mis-applied norm convention
    /// (Gemma stores RMSNorm weights as direct scales here, not as `1 + w` offsets).
    @Test func hiddenStateScaleStaysStable() async throws {
        let encoder = try await Gemma4TextEncoder.load(
            fileURL: URL(fileURLWithPath: Self.encoderPath),
            tokenizerCacheDirectory: Self.tokenizerCache
        )
        let maxLength = 16
        let (states, mask) = try encoder.encode(
            prompt: "The capital of France is", maxLength: maxLength)
        let realTokens = Int(mask.asArray(Float.self).reduce(0, +))

        var scales: [Float] = []
        for state in states {
            let real = state[0..., (maxLength - realTokens)..., 0...].asType(.float32)
            MLX.eval(real)
            scales.append(MLX.sqrt((real * real).mean()).item(Float.self))
        }

        // Layers 0…47 stay within an order of magnitude of the embedding scale;
        // only the last entry is post-final-norm, which deliberately rescales.
        for (index, rms) in scales.dropLast().enumerated() {
            #expect(rms > 0.1 && rms < 50, "state \(index) rms=\(rms) is out of band")
        }
        #expect(scales[scales.count - 1] > 1 && scales[scales.count - 1] < 500)
    }

    @Test func encodesAPromptIntoFortyNineHiddenStates() async throws {
        let encoder = try await Gemma4TextEncoder.load(
            fileURL: URL(fileURLWithPath: Self.encoderPath),
            tokenizerCacheDirectory: Self.tokenizerCache
        )
        #expect(encoder.numHiddenLayers == 48)

        let maxLength = 128
        let prompt = "A golden retriever running through a sunny meadow, cinematic lighting"
        let (states, mask) = try encoder.encode(prompt: prompt, maxLength: maxLength)

        #expect(states.count == 49)
        for state in states {
            #expect(state.shape == [1, maxLength, 3840])
        }

        // Left padding: the mask is a run of zeros followed by a run of ones.
        let maskValues = mask.asArray(Float.self)
        let realTokens = Int(maskValues.reduce(0, +))
        #expect(realTokens > 0 && realTokens < maxLength)
        #expect(maskValues.prefix(maxLength - realTokens).allSatisfy { $0 == 0 })
        #expect(maskValues.suffix(realTokens).allSatisfy { $0 == 1 })

        // Padded slots are exactly zero; real tokens are finite and non-degenerate.
        // A NaN here means a parameter stayed at its random initialisation.
        for index in [0, 1, 24, 47, 48] {
            let state = states[index].asType(.float32)
            MLX.eval(state)
            let padSlice = state[0..., 0 ..< (maxLength - realTokens), 0...]
            #expect(MLX.abs(padSlice).max().item(Float.self) == 0)

            let real = state[0..., (maxLength - realTokens)..., 0...]
            let mean = real.mean().item(Float.self)
            let rms = MLX.sqrt((real * real).mean()).item(Float.self)
            #expect(mean.isFinite && rms.isFinite, "state \(index) is not finite")
            #expect(rms > 0, "state \(index) is all zeros")
        }
    }
}
