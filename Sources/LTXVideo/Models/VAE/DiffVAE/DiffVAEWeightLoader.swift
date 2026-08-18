// DiffVAEWeightLoader.swift - Strict loading for the diffusion video decoder
// Copyright 2026

import Foundation
@preconcurrency import MLX
import MLXNN

/// Loads the diffusion decoder from an LTX-2.5 video-VAE bundle.
///
/// Strict by construction, like every loader added after the split-checkpoint
/// incident: a declared parameter that no checkpoint tensor feeds is an error,
/// not a silently randomly-initialised layer. That class of bug cost a whole
/// generation run once (docs/knowledge: split-checkpoint silent empty load).
public enum DiffVAEWeightLoader {

    /// Whether this file carries a diffusion decoder (as opposed to the
    /// convolutional one, which ships in a sibling file).
    public static func isDiffusionVAE(path: String) -> Bool {
        guard let (_, metadata) = try? MLX.loadArraysAndMetadata(url: URL(fileURLWithPath: path))
        else { return false }
        guard let raw = metadata["config"], let data = raw.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let vae = root["vae"] as? [String: Any]
        else { return false }
        return (vae["_class_name"] as? String)?.contains("Diffusion") == true
    }

    /// Build and populate a decoder from `path`.
    public static func load(from path: String) throws -> DiffusionVideoDecoder {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            throw LTXError.fileNotFound(path)
        }
        let (weights, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let config = try DiffVAEConfig.fromCheckpointMetadata(metadata)
        let decoder = DiffusionVideoDecoder(config: config)
        try apply(weights: weights, to: decoder)
        return decoder
    }

    /// Map checkpoint keys onto the module tree and apply them.
    ///
    /// Key shapes match one-to-one apart from two rewrites:
    /// * the `decoder.` prefix is dropped;
    /// * the latent statistics are named `per_channel_statistics.*-of-means` in
    ///   the file and are addressed by the same name here, so they are kept as
    ///   flat parameters rather than a submodule.
    static func apply(weights: [String: MLXArray], to decoder: DiffusionVideoDecoder) throws {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in weights {
            guard key.hasPrefix("decoder.") || key.hasPrefix("per_channel_statistics.") else {
                continue   // encoder tensors live in the same file; not ours
            }
            var name = key.hasPrefix("decoder.")
                ? String(key.dropFirst("decoder.".count)) : key
            // The checkpoint's t_embedder is a torch Sequential whose index 1 is
            // the activation; its second Linear is therefore ".2". Swift holds
            // the two Linears in a plain array, so the activation has no index.
            if name.hasPrefix("t_embedder.mlp.2.") {
                name = name.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.mlp.1.")
            }
            mapped[name] = value
        }
        guard !mapped.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "No decoder tensors in this file — wrong VAE bundle for the diffusion decoder")
        }

        let declared = Dictionary(uniqueKeysWithValues: decoder.parameters().flattened())
        var updates: [String: MLXArray] = [:]
        var unexpected: [String] = []
        for (key, value) in mapped {
            if let target = declared[key] {
                guard target.shape == value.shape else {
                    throw LTXError.weightLoadingFailed(
                        "Shape mismatch for \(key): model \(target.shape), checkpoint \(value.shape)")
                }
                updates[key] = value
            } else {
                unexpected.append(key)
            }
        }
        let missing = declared.keys.filter { updates[$0] == nil }.sorted()
        guard missing.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "Diffusion decoder is missing \(missing.count) parameters, e.g. "
                + missing.prefix(5).joined(separator: ", "))
        }
        if !unexpected.isEmpty {
            LTXDebug.log("[DiffVAE] \(unexpected.count) unused checkpoint keys, e.g. "
                + unexpected.prefix(3).joined(separator: ", "))
        }

        _ = decoder.update(parameters: ModuleParameters.unflattened(updates))
        eval(decoder.parameters())
        LTXDebug.log("[DiffVAE] applied \(updates.count) weights "
            + "(stages \(decoder.config.stageChannels), steps \(decoder.config.numInferenceSteps))")
    }
}
