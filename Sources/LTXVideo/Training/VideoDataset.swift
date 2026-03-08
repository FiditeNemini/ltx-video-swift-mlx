// VideoDataset.swift - Video + Caption dataset loading for LoRA training
// Copyright 2026

import AVFoundation
import CoreImage
import Foundation
@preconcurrency import MLX

/// A training sample: video frames + caption + optional audio waveform
struct TrainingSample: Sendable {
    /// Video frames as MLXArray (1, 3, T, H, W) in [-1, 1]
    let frames: MLXArray
    /// Text caption
    let caption: String
    /// Audio waveform (optional), float32 1D array
    let audioWaveform: MLXArray?
    /// Source file name (for logging)
    let filename: String
}

/// Loads training data from a directory of videos + captions.
///
/// Expected format (ostris convention):
/// ```
/// dataset/
///   video1.mp4
///   video1.txt
///   video2.mp4
///   video2.txt
/// ```
///
/// Each `.txt` file contains the caption for the corresponding video.
class VideoDataset {
    /// All available samples
    let samples: [SampleInfo]

    /// Target resolution
    let width: Int
    let height: Int

    /// Target frame count
    let numFrames: Int

    /// Whether to extract audio
    let extractAudio: Bool

    /// Audio sample rate for extraction
    let audioSampleRate: Int

    struct SampleInfo: Sendable {
        let videoPath: String
        let captionPath: String
        let filename: String
    }

    init(
        directory: String,
        width: Int,
        height: Int,
        numFrames: Int,
        extractAudio: Bool = false,
        audioSampleRate: Int = 16000
    ) throws {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.extractAudio = extractAudio
        self.audioSampleRate = audioSampleRate

        let fm = FileManager.default
        guard fm.fileExists(atPath: directory) else {
            throw TrainingError.datasetError("Directory not found: \(directory)")
        }

        let contents = try fm.contentsOfDirectory(atPath: directory)
        let videoExtensions = Set(["mp4", "mov", "m4v"])

        var foundSamples: [SampleInfo] = []
        for file in contents {
            let ext = (file as NSString).pathExtension.lowercased()
            guard videoExtensions.contains(ext) else { continue }

            let baseName = (file as NSString).deletingPathExtension
            let captionFile = baseName + ".txt"
            let captionPath = (directory as NSString).appendingPathComponent(captionFile)

            guard fm.fileExists(atPath: captionPath) else {
                print("Warning: No caption file for \(file), skipping")
                continue
            }

            foundSamples.append(SampleInfo(
                videoPath: (directory as NSString).appendingPathComponent(file),
                captionPath: captionPath,
                filename: file
            ))
        }

        guard !foundSamples.isEmpty else {
            throw TrainingError.datasetError("No video+caption pairs found in \(directory)")
        }

        self.samples = foundSamples.sorted { $0.filename < $1.filename }
    }

    /// Load a training sample at the given index
    func loadSample(at index: Int) async throws -> TrainingSample {
        let info = samples[index % samples.count]

        // Read caption
        let caption = try String(contentsOfFile: info.captionPath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Extract video frames
        let frames = try await extractFrames(from: info.videoPath)

        // Extract audio if needed
        var audioWaveform: MLXArray? = nil
        if extractAudio {
            audioWaveform = try await extractAudioWaveform(from: info.videoPath)
        }

        return TrainingSample(
            frames: frames,
            caption: caption,
            audioWaveform: audioWaveform,
            filename: info.filename
        )
    }

    /// Number of samples
    var count: Int { samples.count }

    // MARK: - Frame Extraction

    /// Extract frames from a video file using AVFoundation
    private func extractFrames(from path: String) async throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        let asset = AVURLAsset(url: url)

        // Get video duration and frame rate
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        // Calculate frame times — evenly spaced across video duration
        let frameCount = numFrames
        var frameTimes: [CMTime] = []
        for i in 0..<frameCount {
            let t = durationSeconds * Double(i) / Double(frameCount - 1)
            frameTimes.append(CMTime(seconds: t, preferredTimescale: 600))
        }

        // Set up image generator
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.05, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.05, preferredTimescale: 600)
        generator.maximumSize = CGSize(width: width, height: height)

        // Extract frames
        var frameArrays: [MLXArray] = []
        let ciContext = CIContext()

        for time in frameTimes {
            let (image, _) = try await generator.image(at: time)

            // Convert CGImage to MLXArray (3, H, W) in [-1, 1]
            let ciImage = CIImage(cgImage: image)

            // Render to exact target size
            let scaleX = CGFloat(width) / ciImage.extent.width
            let scaleY = CGFloat(height) / ciImage.extent.height
            let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

            // Get pixel data as RGBA
            let w = width
            let h = height
            var pixelData = [UInt8](repeating: 0, count: w * h * 4)
            ciContext.render(
                scaledImage,
                toBitmap: &pixelData,
                rowBytes: w * 4,
                bounds: CGRect(x: 0, y: 0, width: w, height: h),
                format: .RGBA8,
                colorSpace: CGColorSpaceCreateDeviceRGB()
            )

            // Convert to float32 CHW format in [-1, 1]
            var rChannel = [Float](repeating: 0, count: w * h)
            var gChannel = [Float](repeating: 0, count: w * h)
            var bChannel = [Float](repeating: 0, count: w * h)

            for i in 0..<(w * h) {
                rChannel[i] = Float(pixelData[i * 4]) / 127.5 - 1.0
                gChannel[i] = Float(pixelData[i * 4 + 1]) / 127.5 - 1.0
                bChannel[i] = Float(pixelData[i * 4 + 2]) / 127.5 - 1.0
            }

            // Shape: (3, H, W)
            let r = MLXArray(rChannel, [1, h, w])
            let g = MLXArray(gChannel, [1, h, w])
            let b = MLXArray(bChannel, [1, h, w])
            let frame = MLX.concatenated([r, g, b], axis: 0)
            frameArrays.append(frame)
        }

        // Stack frames: (T, 3, H, W) → (1, 3, T, H, W)
        let stacked = MLX.stacked(frameArrays, axis: 0)  // (T, 3, H, W)
        let transposed = stacked.transposed(1, 0, 2, 3)   // (3, T, H, W)
        let batched = transposed.expandedDimensions(axis: 0)  // (1, 3, T, H, W)

        return batched.asType(.float32)
    }

    // MARK: - Audio Extraction

    /// Extract audio waveform from video using AVAssetReader
    private func extractAudioWaveform(from path: String) async throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        let asset = AVURLAsset(url: url)

        // Check for audio track
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = audioTracks.first else {
            // No audio track — return silence
            let duration = try await asset.load(.duration)
            let numSamples = Int(CMTimeGetSeconds(duration) * Double(audioSampleRate))
            return MLXArray.zeros([numSamples]).asType(.float32)
        }

        // Configure reader for PCM float32 mono
        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: audioSampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: outputSettings)
        reader.add(output)
        reader.startReading()

        var allSamples: [Float] = []
        while let sampleBuffer = output.copyNextSampleBuffer() {
            if let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
                let length = CMBlockBufferGetDataLength(blockBuffer)
                var data = [UInt8](repeating: 0, count: length)
                CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: &data)

                // Interpret as Float32
                let floatCount = length / MemoryLayout<Float>.size
                data.withUnsafeBufferPointer { ptr in
                    ptr.baseAddress!.withMemoryRebound(to: Float.self, capacity: floatCount) { floatPtr in
                        allSamples.append(contentsOf: UnsafeBufferPointer(start: floatPtr, count: floatCount))
                    }
                }
            }
        }

        guard reader.status == .completed else {
            throw TrainingError.datasetError("Audio extraction failed for \(path)")
        }

        return MLXArray(allSamples)
    }
}
