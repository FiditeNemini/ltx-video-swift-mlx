// TrainingControlCommand.swift - CLI commands for training pause/resume/stop
// Copyright 2026

import ArgumentParser
import Foundation
import LTXVideo

struct TrainingControl: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "training",
        abstract: "Control a running LoRA training session",
        subcommands: [Pause.self, Resume.self, Stop.self, Status.self]
    )

    struct Pause: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Pause training (saves checkpoint, resumes with 'training resume')"
        )

        @Argument(help: "Training output directory")
        var outputDir: String

        func run() {
            TrainingController.pauseTraining(outputDir: outputDir)
            print("Pause signal sent to \(outputDir)")
            print("Training will pause at the next step and save a checkpoint.")
        }
    }

    struct Resume: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Resume a paused training session"
        )

        @Argument(help: "Training output directory")
        var outputDir: String

        func run() {
            TrainingController.resumeTraining(outputDir: outputDir)
            print("Resume signal sent to \(outputDir)")
        }
    }

    struct Stop: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Stop training gracefully (saves final checkpoint)"
        )

        @Argument(help: "Training output directory")
        var outputDir: String

        func run() {
            TrainingController.stopTraining(outputDir: outputDir)
            print("Stop signal sent to \(outputDir)")
            print("Training will save a checkpoint and exit.")
        }
    }

    struct Status: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Show training status and progress"
        )

        @Argument(help: "Training output directory")
        var outputDir: String

        func run() throws {
            // Find latest checkpoint
            guard let latest = TrainingState.findLatestCheckpoint(in: outputDir) else {
                print("No training checkpoints found in \(outputDir)")
                return
            }

            let state = try TrainingState.load(from: latest.path)
            let pct = Double(state.currentStep) / Double(state.totalSteps) * 100

            print("Training Status")
            print("===============")
            print("  Step: \(state.currentStep)/\(state.totalSteps) (\(String(format: "%.1f", pct))%)")
            print("  Status: \(state.status.rawValue)")
            print("  Model: \(state.modelType)")
            print("  Rank: \(state.loraRank), Alpha: \(state.loraAlpha)")
            print("  Best loss: \(String(format: "%.6f", state.bestLoss)) (step \(state.bestLossStep))")
            print("  Avg loss (last 20): \(String(format: "%.6f", state.averageLoss))")
            print("  Training time: \(String(format: "%.1f", state.totalTrainingTime))s")

            let eta = state.estimatedTimeRemaining
            if eta > 0 {
                let hours = Int(eta) / 3600
                let mins = (Int(eta) % 3600) / 60
                print("  ETA: \(hours)h\(String(format: "%02d", mins))m")
            }

            if TrainingController.isPaused(outputDir: outputDir) {
                print("\n  ⏸ Training is PAUSED. Use 'ltx-video training resume \(outputDir)' to continue.")
            }
        }
    }
}
