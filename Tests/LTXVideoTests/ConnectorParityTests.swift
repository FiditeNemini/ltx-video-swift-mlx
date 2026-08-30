// ConnectorParityTests.swift — element-wise parity against the reference connector
// Copyright 2026
//
// Every prompt goes through this pipeline: 49 Gemma hidden-state layers ->
// FeatureExtractorV2 (per-token RMS norm + concat -> video_aggregate_embed,
// 3840*49 -> 4096) -> Embeddings1DConnector (register replacement for padded
// tokens -> 8-layer 1D transformer, SPLIT RoPE, gated attention) -> final RMS
// norm. Sub-task 3 of #57's breakdown.
//
// Ground truth comes from Lightricks' own FeatureExtractorV2 and
// Embeddings1DConnector, run on CPU float32 over synthetic Gemma hidden
// states (bypassing the real ~24 GB Gemma 4 model on purpose — this tests
// the connector math, not Gemma, which Gemma4TextEncoderE2ETests already
// covers) — dumped by scripts/connector_reference.py.
//
//   TEST_RUNNER_LTX25_GEMMA_PROJ=<gemma4-with-proj-bf16.safetensors> \
//   TEST_RUNNER_LTX25_CONNECTOR_UNIFIED=<unified-transformer-bf16.safetensors> \
//   TEST_RUNNER_LTX25_CONNECTOR_REF=<connector_reference.safetensors> \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/ConnectorParityTests

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

@Suite("Text connector parity vs the reference implementation",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_CONNECTOR_REF"] != nil),
       .serialized)
struct ConnectorParityTests {

    func env(_ key: String) -> String { ProcessInfo.processInfo.environment[key]! }

    func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
        let scale = MLX.abs(b.asType(.float32)).mean().item(Float.self)
        return scale > 0 ? diff / scale : diff
    }

    /// Video only: the feature extractor declares no `audioAggregateEmbed`
    /// parameter when `includeAudioConnector: false`, so feeding it audio
    /// weights is a hard MLXNN error ("item: none") — filter them out.
    func loadModel() throws -> VideoGemmaTextEncoderModel {
        let model = createTextEncoder(includeAudioConnector: false)

        // Feature extractor: video_aggregate_embed lives in the Gemma-with-proj
        // bundle, keyed text_embedding_projection.*.
        let gemmaRaw = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_GEMMA_PROJ")))
        let projOnly = gemmaRaw.filter { $0.key.hasPrefix("text_embedding_projection.video_aggregate_embed.") }
        let feWeights = LTXWeightLoader.mapTextEncoderWeights(projOnly)

        // Connector: video_embeddings_connector.* lives in the unified
        // transformer checkpoint, under model.diffusion_model.*.
        let split = try LTXWeightLoader.splitUnifiedWeightsFile(
            path: env("LTX25_CONNECTOR_UNIFIED"), includeAudio: false)

        var combined = feWeights
        for (k, v) in split.connector { combined[k] = v }

        // MLXNN's no-`verify:` update() silently no-ops on unmatched or
        // missing keys — a broken key mapping would leave a parameter at its
        // random init and still "load" without error, invalidating every
        // parity number below without saying so.
        let declared = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let missing = declared.keys.filter { combined[$0] == nil }.sorted()
        guard missing.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "ConnectorParityTests: \(missing.count) model parameters got no checkpoint value, "
                + "e.g. \(missing.prefix(5).joined(separator: ", "))")
        }

        _ = model.update(parameters: ModuleParameters.unflattened(combined))
        eval(model.parameters())
        return model
    }

    /// Unstack reference["hidden_states"] ([B, T, D, L]) into the per-layer
    /// list the Swift API expects, matching scripts/connector_reference.py's
    /// torch.stack(hidden_states, dim=-1).
    func loadHiddenStates(_ reference: [String: MLXArray]) -> [MLXArray] {
        let stacked = reference["hidden_states"]!.asType(DType.float32)
        let numLayers = stacked.dim(-1)
        return (0..<numLayers).map { stacked[0..., 0..., 0..., $0] }
    }

    /// Walks the feature extractor and each of the 8 transformer blocks,
    /// reporting where the port first departs from the reference. The
    /// transformer blocks are fed the reference's own post-register-replacement
    /// hidden state directly (bypassing the Swift port's private register
    /// substitution) — isolating the 8-block math from register replacement,
    /// which is exactly how the register-replacement bug below was found: the
    /// blocks matched to ~1e-5 in isolation while the full pipeline was 135%
    /// off (measured with this repo's actual left-padding convention — see
    /// connectorOutputMatchesReference).
    @Test func bisectFirstDivergence() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_CONNECTOR_REF")))
        let model = try loadModel()
        let hiddenStates = loadHiddenStates(reference)
        let attentionMask = reference["attention_mask"]!.asType(DType.float32)

        let feOutput = model.featureExtractor.extractFromHiddenStates(
            hiddenStates: hiddenStates, attentionMask: attentionMask)
        MLX.eval(feOutput)
        let refFE = reference["feature_extractor_output"]!.asType(DType.float32)
        #expect(feOutput.shape == refFE.shape, "feature_extractor shape \(feOutput.shape) vs ref \(refFE.shape)")
        let feError = relativeError(feOutput, refFE)
        print("PARITY feature_extractor: relative error \(feError)")
        #expect(feError < 0.02, "feature_extractor diverges from the reference: \(feError)")

        let postRegister = reference["post_register_hidden"]!.asType(DType.float32)
        let connector = model.embeddingsConnector
        let seqLen = postRegister.dim(1)
        let indicesGrid = MLXArray(0..<seqLen).asType(DType.float32).reshaped([1, 1, seqLen])
        var freqsCis = precomputeFreqsCis(
            indicesGrid: indicesGrid,
            dim: connector.innerDim,
            theta: connector.positionalEmbeddingTheta,
            maxPos: connector.positionalEmbeddingMaxPos,
            numAttentionHeads: connector.numAttentionHeads,
            ropeType: .split,
            doublePrecision: true
        )
        freqsCis = (cos: freqsCis.cos.asType(DType.float32), sin: freqsCis.sin.asType(DType.float32))

        var x = postRegister
        for (i, block) in connector.transformer1DBlocks.enumerated() {
            x = block(x, mask: nil, pe: freqsCis)
            MLX.eval(x)
            guard let refBlock = reference["block_\(i)"]?.asType(DType.float32) else { continue }
            #expect(x.shape == refBlock.shape, "block_\(i) shape \(x.shape) vs ref \(refBlock.shape)")
            let blockError = relativeError(x, refBlock)
            print("PARITY block_\(i): relative error \(blockError)")
            #expect(blockError < 0.02, "block_\(i) diverges from the reference: \(blockError)")
        }
    }

    @Test func connectorOutputMatchesReference() throws {
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: env("LTX25_CONNECTOR_REF")))
        let model = try loadModel()
        let hiddenStates = loadHiddenStates(reference)
        let attentionMask = reference["attention_mask"]!.asType(DType.float32)

        let output = try model.encodeFromHiddenStates(
            hiddenStates: hiddenStates, attentionMask: attentionMask)
        MLX.eval(output.videoEncoding)

        let refConnector = reference["connector_output"]!.asType(DType.float32)
        #expect(output.videoEncoding.shape == refConnector.shape,
                "output \(output.videoEncoding.shape) vs reference \(refConnector.shape)")
        let err = relativeError(output.videoEncoding, refConnector)
        print("PARITY connector output relative error: \(err)")
        #expect(err < 0.02, "connector diverges from the reference: \(err)")
    }
}
