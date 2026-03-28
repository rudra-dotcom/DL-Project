import CoreML
import Foundation
import UIKit

enum RepViTPredictionError: LocalizedError {
    case labelsMissing
    case modelUnavailable
    case imageConversionFailed
    case outputUnavailable

    var errorDescription: String? {
        switch self {
        case .labelsMissing:
            return "ImageNet labels could not be loaded from the app bundle."
        case .modelUnavailable:
            return "The RepViT Core ML model could not be loaded."
        case .imageConversionFailed:
            return "The selected image could not be prepared for Core ML."
        case .outputUnavailable:
            return "The model did not return logits for classification."
        }
    }
}

actor RepViTPredictionEngine {
    private let labels: [String]
    private let model: MLModel
    private let modelName = "RepViT_M1_1"
    private let inputFeatureName = "x_1"
    private let outputFeatureName = "var_1177"

    init() throws {
        guard let labelsURL = Bundle.main.url(forResource: "imagenet_classes", withExtension: "txt"),
              let labelText = try? String(contentsOf: labelsURL, encoding: .utf8) else {
            throw RepViTPredictionError.labelsMissing
        }

        let labels = labelText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        let generatedModel = try RepViT_M1_1(configuration: configuration)
        let rawModel = generatedModel.model

        self.labels = labels
        self.model = rawModel
    }

    func classify(image: UIImage, inputSourceLabel: String) throws -> PredictionResult {
        guard let normalizedImage = image.normalizedCGImage(),
              let preparedImage = image.fittedSquareCGImage() else {
            throw RepViTPredictionError.imageConversionFailed
        }

        let featureValue = try MLFeatureValue(
            cgImage: preparedImage,
            pixelsWide: 224,
            pixelsHigh: 224,
            pixelFormatType: kCVPixelFormatType_32ARGB,
            options: nil
        )

        let featureProvider = try MLDictionaryFeatureProvider(dictionary: [
            inputFeatureName: featureValue,
        ])

        let start = ContinuousClock.now
        let output = try model.prediction(from: featureProvider)
        let latency = start.duration(to: .now)

        guard let logits = output.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw RepViTPredictionError.outputUnavailable
        }

        let predictions = topPredictions(from: logits, topK: 5)
        let predictionLatencyMs =
            (Double(latency.components.seconds) * 1_000) +
            (Double(latency.components.attoseconds) / 1e15)
        let runtimeMetrics = RuntimeMetrics(
            predictionLatencyMs: predictionLatencyMs,
            selectedImageSize: CGSize(width: normalizedImage.width, height: normalizedImage.height),
            compiledModelSizeLabel: compiledModelSize(),
            inputSourceLabel: inputSourceLabel
        )

        return PredictionResult(topPredictions: predictions, runtimeMetrics: runtimeMetrics)
    }

    private func topPredictions(from logits: MLMultiArray, topK: Int) -> [ClassPrediction] {
        let count = logits.count
        guard count > 0 else {
            return []
        }

        let values = (0..<count).map { index in
            Double(truncating: logits[index])
        }
        let maxLogit = values.max() ?? 0
        let expValues = values.map { Foundation.exp($0 - maxLogit) }
        let denominator = expValues.reduce(0, +)
        let probabilities = expValues.map { $0 / denominator }

        let ranked = probabilities
            .enumerated()
            .sorted { lhs, rhs in lhs.element > rhs.element }
            .prefix(topK)

        return ranked.enumerated().map { rank, entry in
            let label = labels.indices.contains(entry.offset) ? labels[entry.offset] : "Class \(entry.offset)"
            return ClassPrediction(rank: rank + 1, label: label, confidence: entry.element)
        }
    }

    private func compiledModelSize() -> String {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            return "Unavailable"
        }

        let size = directorySize(at: modelURL)
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        return formatter.string(fromByteCount: Int64(size))
    }

    private func directorySize(at url: URL) -> Int64 {
        let resourceKeys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: Array(resourceKeys),
            options: [.skipsHiddenFiles]
        )

        var totalSize: Int64 = 0
        while let fileURL = enumerator?.nextObject() as? URL {
            guard let values = try? fileURL.resourceValues(forKeys: resourceKeys),
                  values.isRegularFile == true,
                  let fileSize = values.fileSize else {
                continue
            }

            totalSize += Int64(fileSize)
        }
        return totalSize
    }
}
