// ModelVariantParsing.swift - CLI model selection
// Copyright 2025

import ArgumentParser
import LTXVideo

/// Resolve a `--model` string, rejecting variants this build cannot run.
///
/// Catalogued-but-unimplemented variants fail here with the reason rather than
/// after a multi-gigabyte download.
func parseModelVariant(_ raw: String) throws -> LTXModel {
    guard let variant = LTXModel(rawValue: raw) else {
        let known = LTXModel.allCases.map(\.rawValue).joined(separator: ", ")
        throw ValidationError("Unknown model '\(raw)'. Available: \(known)")
    }
    if case .notImplemented(let reason) = variant.support {
        throw ValidationError("\(variant.displayName) is not runnable yet: \(reason)")
    }
    return variant
}
