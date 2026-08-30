//
//  RetakeDevRecipeTests.swift
//  ltx-video-swift-mlx
//
//  Locks LTXPipeline.retakeUsesDevRecipe to capability (isForTraining), not
//  identity (model == .dev) — the latter silently ran LTX-2.5 dev through the
//  distilled 8-step schedule.
//

import Testing
@testable import LTXVideo

@Suite("RetakeDevRecipe")
struct RetakeDevRecipeTests {

    @Test func testDevCheckpointsUseTheDevRecipe() {
        #expect(LTXPipeline.retakeUsesDevRecipe(.dev))
        #expect(LTXPipeline.retakeUsesDevRecipe(.v25Dev))
    }

    @Test func testDistilledCheckpointsDoNotUseTheDevRecipe() {
        #expect(!LTXPipeline.retakeUsesDevRecipe(.distilled))
        #expect(!LTXPipeline.retakeUsesDevRecipe(.v25Distilled))
    }

    // Regression lock: a future model variant must not silently fall back to
    // an identity comparison (`model == .dev`) that misses it.
    @Test func testMatchesIsForTrainingForEveryModel() {
        for model in LTXModel.allCases {
            #expect(LTXPipeline.retakeUsesDevRecipe(model) == model.isForTraining,
                    "\(model) diverges from isForTraining")
        }
    }
}
