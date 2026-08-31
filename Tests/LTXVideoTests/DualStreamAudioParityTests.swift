// DualStreamAudioParityTests.swift — LTX2Transformer's dual video/audio blocks
// against the reference implementation
// Copyright 2026
//
// Sub-task 5 of issue #57's breakdown (sub-tasks 1-4: video VAE decoder/encoder,
// text connector, audio VAE + vocoder — PRs #76, #77, #78, #80). Extends
// `scripts/transformer_reference.py` (already covering the video-only stream)
// with the "av" variant: Lightricks' own `LTXModel` built with
// `LTXModelType.AudioVideo`, small dims, fixed weights, and — critically —
// *different* sigmas per stream (video 0.7, audio 0.3). A bug that feeds the
// wrong modality's sigma into a cross-modal AdaLN is a no-op when both sigmas
// are equal, which is exactly why
// docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md went
// undetected on real (matched-sigma, non-LipDub) generations for months.
//
//   PYTHONPATH=<ltx-core>/src python3 scripts/transformer_reference.py ref.safetensors av
//   TEST_RUNNER_LTX_TRANSFORMER_AUDIO_REFERENCE=$PWD/ref.safetensors xcodebuild ... test \
//     -only-testing:LTXVideoTests/DualStreamAudioParityTests

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

@Suite("Dual-stream (video+audio) transformer parity",
       .enabled(if: ProcessInfo.processInfo.environment["LTX_TRANSFORMER_AUDIO_REFERENCE"] != nil))
struct DualStreamAudioParityTests {

    static var referencePath: String? {
        ProcessInfo.processInfo.environment["LTX_TRANSFORMER_AUDIO_REFERENCE"]
    }

    /// Mirrors `scripts/transformer_reference.py`'s "av" branch constants.
    static let config = LTXTransformerConfig(
        numLayers: 2, numAttentionHeads: 2, attentionHeadDim: 8,
        inChannels: 4, outChannels: 4,
        crossAttentionDim: 16, captionChannels: 16,
        ropeTheta: 10000.0, maxPos: [20, 2048, 2048],
        timestepScaleMultiplier: 1000, normEps: 1e-6,
        audioNumAttentionHeads: 2, audioAttentionHeadDim: 4,
        audioInChannels: 4, audioOutChannels: 4,
        audioMaxPos: [20],
        gatedAttention: true,
        crossAttentionAdaLN: true,
        captionProjBeforeConnector: true)
    static let videoShape = (frames: 2, height: 2, width: 3)   // -> 12 video tokens
    static let audioFrames = 5                                  // -> 5 audio tokens
    // Deliberately different — see the file header.
    static let sigmaVideo: Float = 0.7
    static let sigmaAudio: Float = 0.3

    func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
        let scale = MLX.abs(b.asType(.float32)).mean().item(Float.self)
        return scale > 0 ? diff / scale : diff
    }

    /// Loads the reference dump's weights onto a real `LTX2Transformer`, with the
    /// same key-coverage discipline as `TransformerParityTests`: every declared
    /// parameter must get a checkpoint value, except the affine-free block norms
    /// (no weight in the checkpoint — the AdaLN scale/shift carries the affine
    /// term instead, docs/knowledge/pitfalls/affine-free-norms-expected-missing.md)
    /// and `keyframes_abs_pos_embedding` (absent because
    /// `use_keyframes_abs_pos_embedding=False` in the reference).
    func loadModel(_ tensors: [String: MLXArray]) throws -> LTX2Transformer {
        var weights: [String: MLXArray] = [:]
        for (key, value) in tensors where key.hasPrefix("weight.") {
            weights[String(key.dropFirst("weight.".count))] = value.asType(.float32)
        }
        #expect(!weights.isEmpty, "reference carries no weights")

        let model = LTX2Transformer(config: Self.config, ropeType: .split)
        let mapped = LTXWeightLoader.mapTransformerWeights(weights, includeAudio: true)
        let declared = Set(model.parameters().flattened().map { $0.0 })
        let unexpected = mapped.keys.filter { !declared.contains($0) }.sorted()
        #expect(unexpected.isEmpty, "reference keys with no parameter: \(unexpected)")
        let missing = declared.filter { mapped[$0] == nil }.sorted()
            .filter {
                !$0.hasSuffix(".norm1.weight") && !$0.hasSuffix(".norm2.weight")
                    && !$0.hasSuffix(".norm3.weight")
                    && !$0.hasSuffix(".audio_norm1.weight") && !$0.hasSuffix(".audio_norm2.weight")
                    && !$0.hasSuffix(".audio_norm3.weight")
                    && !$0.hasSuffix(".audio_to_video_norm.weight")
                    && !$0.hasSuffix(".video_to_audio_norm.weight")
                    && $0 != "keyframes_abs_pos_embedding"
            }
        #expect(missing.isEmpty, "parameters the reference does not feed: \(missing)")
        _ = model.update(parameters: ModuleParameters.unflattened(mapped))
        MLX.eval(model.parameters())
        return model
    }

    /// Full forward pass, asserting the video and audio outputs *separately* —
    /// per the plan, a 2% threshold on a single combined number is not enough
    /// here: a mis-scaled cross-modal gate can leave video nearly correct while
    /// destroying audio (or vice versa), and averaging the two would hide it.
    @Test func videoAndAudioOutputsMatchReferenceSeparately() throws {
        let (tensors, _) = try MLX.loadArraysAndMetadata(
            url: URL(fileURLWithPath: Self.referencePath!))
        let model = try loadModel(tensors)

        let videoLatent = try #require(tensors["input.video_latent"]).asType(.float32)
        let videoContext = try #require(tensors["input.video_context"]).asType(.float32)
        let audioLatent = try #require(tensors["input.audio_latent"]).asType(.float32)
        let audioContext = try #require(tensors["input.audio_context"]).asType(.float32)
        let expectedVideo = try #require(tensors["output.video_velocity"]).asType(.float32)
        let expectedAudio = try #require(tensors["output.audio_velocity"]).asType(.float32)

        let tokens = Self.videoShape.frames * Self.videoShape.height * Self.videoShape.width
        let videoTimesteps = MLXArray.full([1, tokens], values: MLXArray(Self.sigmaVideo))
        let audioTimesteps = MLXArray.full([1, Self.audioFrames], values: MLXArray(Self.sigmaAudio))

        let (videoOut, audioOut) = model(
            videoLatent: videoLatent,
            audioLatent: audioLatent,
            videoContext: videoContext,
            audioContext: audioContext,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            videoLatentShape: Self.videoShape,
            audioNumFrames: Self.audioFrames
        )
        MLX.eval(videoOut, audioOut)

        #expect(videoOut.shape == expectedVideo.shape, "video \(videoOut.shape) vs ref \(expectedVideo.shape)")
        #expect(audioOut.shape == expectedAudio.shape, "audio \(audioOut.shape) vs ref \(expectedAudio.shape)")

        let videoErr = relativeError(videoOut, expectedVideo)
        let audioErr = relativeError(audioOut, expectedAudio)
        print("PARITY av video output: relative error \(videoErr)")
        print("PARITY av audio output: relative error \(audioErr)")
        #expect(videoErr < 0.02, "video stream diverges from the reference: \(videoErr)")
        #expect(audioErr < 0.02, "audio stream diverges from the reference: \(audioErr)")
    }

    /// Isolates the cross-modal AdaLN modules from the rest of the block stack:
    /// each is called exactly once per forward pass (shared by every block), so
    /// this pins down whether a divergence in the full-forward test above
    /// originates in the AdaLN inputs (a sigma swap, a missing av_ca_factor) or
    /// somewhere downstream in the block math itself. Feeds each module both
    /// candidate inputs (this modality's own sigma, and the other modality's) so
    /// a failure here names which one the reference actually used.
    @Test func crossModalAdaLNInputsMatchReference() throws {
        let (tensors, _) = try MLX.loadArraysAndMetadata(
            url: URL(fileURLWithPath: Self.referencePath!))
        let model = try loadModel(tensors)

        let scaleShiftMultiplier = MLXArray(Float(Self.config.timestepScaleMultiplier))
        let ownVideo = MLXArray([Self.sigmaVideo]) * scaleShiftMultiplier
        let ownAudio = MLXArray([Self.sigmaAudio]) * scaleShiftMultiplier
        // av_ca_timestep_scale_multiplier defaults to 1 on both sides (matching
        // the reference script), so the gate input is un-scaled sigma.
        let gateVideo = MLXArray([Self.sigmaVideo])
        let gateAudio = MLXArray([Self.sigmaAudio])

        // Both candidates are checked and printed for every module — not just the
        // one this test asserts on — so a future regression that swaps a pair
        // back reads as a clean "matches the other candidate instead" rather
        // than a bare threshold failure.
        func checkAgainst(
            _ name: String, own: MLXArray, cross: MLXArray, module: AdaLayerNormSingle, correctIsOwn: Bool
        ) throws {
            let ref = try #require(tensors["stage.\(name)"]).asType(.float32)
            let (ownEmb, _) = module(own)
            let (crossEmb, _) = module(cross)
            MLX.eval(ownEmb, crossEmb)
            let ownErr = relativeError(ownEmb, ref)
            let crossErr = relativeError(crossEmb, ref)
            print("PARITY av \(name): own-sigma relative error \(ownErr), cross-sigma relative error \(crossErr)")
            let matchErr = correctIsOwn ? ownErr : crossErr
            let expectedSide = correctIsOwn ? "own" : "cross"
            let message = "\(name) should match \(expectedSide)-modality sigma: own \(ownErr), cross \(crossErr)"
            #expect(matchErr < 0.02, "\(message)")
        }

        // Scale/shift AdaLNs take THIS modality's own sigma; the gate AdaLNs
        // take the OTHER modality's — see the matching comment in
        // LTX2Transformer.swift's cross-modal timestep embeddings section.
        try checkAgainst("cross_scale_shift_video", own: ownVideo, cross: ownAudio,
                          module: model.avCrossAttnVideoScaleShift, correctIsOwn: true)
        try checkAgainst("cross_scale_shift_audio", own: ownAudio, cross: ownVideo,
                          module: model.avCrossAttnAudioScaleShift, correctIsOwn: true)
        try checkAgainst("cross_gate_a2v", own: gateVideo, cross: gateAudio,
                          module: model.avCrossAttnVideoA2VGate, correctIsOwn: false)
        try checkAgainst("cross_gate_v2a", own: gateAudio, cross: gateVideo,
                          module: model.avCrossAttnAudioV2AGate, correctIsOwn: false)
    }
}
