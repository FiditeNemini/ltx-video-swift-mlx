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

    /// Mirrors `scripts/transformer_reference.py`'s "av" branch exactly: half the
    /// video tokens and one of five audio tokens held below the modality's active
    /// sigma, simulating I2V/keyframe/LipDub conditioning tokens mixed with real
    /// denoising ones. Uniform per-token timesteps can't distinguish a port that
    /// collapses them to one broadcast value from one that doesn't — see the
    /// cross_scale_shift_video/audio taps this feeds into
    /// `crossModalAdaLNInputsMatchReference`.
    static func nonUniformTimesteps() -> (video: MLXArray, audio: MLXArray) {
        let videoTokens = videoShape.frames * videoShape.height * videoShape.width
        let half = videoTokens / 2
        let video = MLX.concatenated([
            MLXArray.full([1, half], values: MLXArray(sigmaVideo * 0.4)),
            MLXArray.full([1, videoTokens - half], values: MLXArray(sigmaVideo))
        ], axis: 1)
        let audio = MLX.concatenated([
            MLXArray.full([1, 1], values: MLXArray(Float(0))),
            MLXArray.full([1, audioFrames - 1], values: MLXArray(sigmaAudio))
        ], axis: 1)
        return (video, audio)
    }

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

        let (videoTimesteps, audioTimesteps) = Self.nonUniformTimesteps()

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
        // 2e-4, matching TransformerParityTests's video-only threshold, not the
        // family's usual 2%: this is a pure float32 synthetic model with no
        // legitimate large noise source, and 2% was measured to still pass with
        // the scale-shift collapsed-to-scalar regression present (1.1e-3 /
        // 5.4e-3 against a clean baseline of ~1e-6) — exactly the insensitivity
        // that let that bug ship unnoticed.
        #expect(videoErr < 2e-4, "video stream diverges from the reference: \(videoErr)")
        #expect(audioErr < 2e-4, "audio stream diverges from the reference: \(audioErr)")
    }

    /// Isolates the cross-modal AdaLN modules from the rest of the block stack:
    /// each is called exactly once per forward pass (shared by every block), so
    /// this pins down whether a divergence in the full-forward test above
    /// originates in the AdaLN inputs (own/cross sigma, a missing av_ca_factor,
    /// a wrongly-collapsed per-token tensor) or somewhere downstream in the
    /// block math itself. Feeds each module *exactly* the formula
    /// `LTX2Transformer.callAsFunction` computes internally, so a shape
    /// mismatch alone (own modality's token count vs. the reference's) is
    /// enough to catch a regression back to a scalar-collapsed scale/shift.
    @Test func crossModalAdaLNInputsMatchReference() throws {
        let (tensors, _) = try MLX.loadArraysAndMetadata(
            url: URL(fileURLWithPath: Self.referencePath!))
        let model = try loadModel(tensors)

        let (videoTimesteps, audioTimesteps) = Self.nonUniformTimesteps()
        let scaleShiftMultiplier = Float(Self.config.timestepScaleMultiplier)
        // Scale/shift: THIS modality's own per-token timesteps, unreduced.
        let videoScaleShiftInput = (videoTimesteps * scaleShiftMultiplier).flattened()
        let audioScaleShiftInput = (audioTimesteps * scaleShiftMultiplier).flattened()
        // Gate: the OTHER modality's active scalar sigma (av_ca_factor = 1
        // with the reference script's defaults, so no further scaling).
        let scalarVideoSigma = videoTimesteps.max(axis: 1).flattened()
        let scalarAudioSigma = audioTimesteps.max(axis: 1).flattened()

        func check(_ name: String, _ module: AdaLayerNormSingle, _ input: MLXArray) throws {
            let ref = try #require(tensors["stage.\(name)"]).asType(.float32)
            let (embedding, _) = module(input)
            MLX.eval(embedding)
            #expect(embedding.shape == ref.shape, "\(name) shape \(embedding.shape) vs ref \(ref.shape)")
            guard embedding.shape == ref.shape else { return }
            let err = relativeError(embedding, ref)
            print("PARITY av \(name): relative error \(err)")
            #expect(err < 2e-4, "\(name) diverges from the reference: \(err)")
        }

        try check("cross_scale_shift_video", model.avCrossAttnVideoScaleShift, videoScaleShiftInput)
        try check("cross_scale_shift_audio", model.avCrossAttnAudioScaleShift, audioScaleShiftInput)
        try check("cross_gate_a2v", model.avCrossAttnVideoA2VGate, scalarAudioSigma)
        try check("cross_gate_v2a", model.avCrossAttnAudioV2AGate, scalarVideoSigma)
    }
}
