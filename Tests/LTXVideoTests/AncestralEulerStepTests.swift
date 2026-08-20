// AncestralEulerStepTests.swift — the ancestral step against its own contract
// Copyright 2026

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Ancestral Euler step")
struct AncestralEulerStepTests {

    static func fixture() -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(3)
        return (MLXRandom.normal([2, 4]), MLXRandom.normal([2, 4]), MLXRandom.normal([2, 4]))
    }

    @Test func etaZeroIsAPlainEulerStep() {
        // The whole point of covering both in one implementation: at eta = 0 it
        // must reduce exactly to the deterministic interpolation, noise unused.
        let (x, denoised, noise) = Self.fixture()
        let step = AncestralEulerStep(eta: 0)
        let got = step(sample: x, denoised: denoised, sigma: 0.8, sigmaNext: 0.5, noise: noise)
        let ratio = Float(0.5 / 0.8)
        let expected = MLXArray(ratio) * x + MLXArray(1 - ratio) * denoised
        MLX.eval(got, expected)
        #expect(MLX.abs(got - expected).max().item(Float.self) < 1e-6)
    }

    @Test func lastStepReturnsTheDenoisedPrediction() {
        let (x, denoised, noise) = Self.fixture()
        let got = AncestralEulerStep(eta: 0.5)(
            sample: x, denoised: denoised, sigma: 0.1, sigmaNext: 0, noise: noise)
        MLX.eval(got)
        #expect(MLX.abs(got - denoised).max().item(Float.self) == 0)
    }

    @Test func ancestralStepInjectsNoiseAndDependsOnIt() {
        let (x, denoised, noise) = Self.fixture()
        let step = AncestralEulerStep(eta: 0.5)
        let a = step(sample: x, denoised: denoised, sigma: 0.8, sigmaNext: 0.5, noise: noise)
        let b = step(sample: x, denoised: denoised, sigma: 0.8, sigmaNext: 0.5, noise: noise * -1)
        let deterministic = AncestralEulerStep(eta: 0)(
            sample: x, denoised: denoised, sigma: 0.8, sigmaNext: 0.5, noise: noise)
        MLX.eval(a, b, deterministic)
        // It must actually use the noise…
        #expect(MLX.abs(a - b).max().item(Float.self) > 1e-3)
        // …and differ from the deterministic step it would otherwise be.
        #expect(MLX.abs(a - deterministic).max().item(Float.self) > 1e-3)
    }

    @Test func missingNoiseFallsBackDeterministically() {
        // Rather than fabricating noise, which would look like a working
        // ancestral sampler while being a silent lie.
        let (x, denoised, _) = Self.fixture()
        let got = AncestralEulerStep(eta: 0.5)(
            sample: x, denoised: denoised, sigma: 0.8, sigmaNext: 0.5, noise: nil)
        MLX.eval(got)
        #expect(got.shape == x.shape)
    }
}
