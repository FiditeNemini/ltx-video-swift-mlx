// LTX25CheckpointSourceE2ETests.swift — Gated E2E for split-checkpoint loading
// Copyright 2026
//
// Guards the failure mode that cost a full generation to find: reading a
// component from the wrong file of a split checkpoint returns *zero* keys rather
// than raising, so the module keeps its random initialisation and produces
// plausible-looking garbage. That is exactly what happened to the VAE encoder —
// the run finished, the video was coherent, and the conditioning image had
// simply been encoded to noise.
//
// Gated behind LTX25_MODELS_DIR pointing at a directory holding the LTX-2.5
// files under their published names.
//
// Run:
//   TEST_RUNNER_LTX25_MODELS_DIR=/Volumes/Lexar/models/ltx-2.5 \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/LTX25CheckpointSourceE2ETests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("LTX-2.5 split checkpoint (gated: LTX25_MODELS_DIR)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] != nil),
       .serialized)
struct LTX25CheckpointSourceE2ETests {

    static var modelsDirectory: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] ?? "")
    }

    static func source(for model: LTXModel = .v25Distilled) -> LTXCheckpointSource {
        let directory = modelsDirectory
        let file = { (path: String) in
            directory.appendingPathComponent((path as NSString).lastPathComponent)
        }
        let shared = model.family.sharedComponentFiles
        return LTXCheckpointSource(
            model: model,
            paths: LTXCheckpointPaths(
                transformer: file(model.unifiedWeightsFilename),
                videoVAE: file(shared.first { $0.kind == .videoVAE }!.path),
                textEncoder: file(shared.first { $0.kind == .textEncoder }!.path)))
    }

    @Test func componentsComeFromTheirOwnFiles() throws {
        let (transformer, vae, connector) = try Self.source().loadComponents()

        // The transformer shard holds the DiT and both connectors, and nothing else.
        #expect(transformer.count > 1000)
        #expect(transformer["patchify_proj.weight"] != nil)
        // LTX-2.5 is bias-free in the block FFNs; 2.3 was not.
        #expect(transformer["transformer_blocks.0.ff.project_in.proj.weight"] != nil)
        #expect(transformer["transformer_blocks.0.ff.project_in.proj.bias"] == nil)

        // The VAE comes from the VAE file, not the (VAE-less) transformer shard.
        #expect(vae["mean_of_means"] != nil)
        #expect(vae.keys.contains { $0.hasPrefix("up_blocks_") })

        // The aggregate projections ship with the text encoder in 2.5, the
        // connector blocks with the transformer. Both must land in one bucket.
        #expect(connector["feature_extractor.video_aggregate_embed.weight"] != nil)
        #expect(connector.keys.contains { $0.hasPrefix("embeddings_connector.") })
    }

    /// The regression guard: a wrong-file lookup returns an empty dictionary,
    /// which used to sail through and leave the encoder randomly initialised.
    @Test func vaeEncoderWeightsAreFoundAndComplete() throws {
        let encoder = try Self.source().loadVAEEncoderWeights()
        #expect(!encoder.isEmpty)
        #expect(encoder["conv_in.weight"] != nil)
        #expect(encoder.keys.contains { $0.hasPrefix("down_blocks_") })

        // Every parameter the module declares must be fed — the whole point is
        // that a silently-unfed encoder still "works" and emits noise.
        let model = VideoEncoder()
        let declared = Set(model.parameters().flattened().map(\.0))
        let missing = declared.subtracting(encoder.keys)
        #expect(missing.isEmpty, "unfed VAE encoder parameters: \(missing.sorted().prefix(5))")
    }

    @Test func theTransformerNamesItsTextEncoder() throws {
        let source = Self.source()
        let metadata = try source.transformerMetadata()
        #expect(metadata["model_version"] == LTXModelFamily.ltx25.checkpointModelVersion)

        let assets = try LTX25TextEncoderAssets(
            fileURL: source.paths.textEncoder!)
        #expect(throws: Never.self) {
            try assets.verifyPairing(withTransformerMetadata: metadata)
        }
    }
}
