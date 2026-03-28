import CoreGraphics
import Foundation

struct RepViTModelProfile: Identifiable, Hashable {
    let id: String
    let displayName: String
    let trainingVariant: String
    let datasetName: String
    let classCount: Int
    let inputResolution: CGSize
    let top1Accuracy: Double
    let top5Accuracy: Double
    let parameterCountLabel: String
    let macsLabel: String
    let publishedLatencyMs: Double
    let publishedLatencyContext: String
    let sourceSummary: String

    static let bundled = RepViTModelProfile(
        id: "repvit-m1.1-450e",
        displayName: "RepViT-M1.1",
        trainingVariant: "450e",
        datasetName: "ImageNet-1K",
        classCount: 1_000,
        inputResolution: CGSize(width: 224, height: 224),
        top1Accuracy: 81.2,
        top5Accuracy: 95.49,
        parameterCountLabel: "8.2M",
        macsLabel: "1.3G",
        publishedLatencyMs: 1.1,
        publishedLatencyContext: "Measured on iPhone 12, iOS 16, with the Xcode 14 benchmark tool.",
        sourceSummary: "Bundled from the published Core ML release and aligned with the repo's ImageNet-1K metrics."
    )
}

struct RuntimeMetrics: Equatable {
    let predictionLatencyMs: Double
    let selectedImageSize: CGSize
    let compiledModelSizeLabel: String
    let inputSourceLabel: String
}

struct ClassPrediction: Identifiable, Equatable {
    let id = UUID()
    let rank: Int
    let label: String
    let confidence: Double
}

struct PredictionResult: Equatable {
    let topPredictions: [ClassPrediction]
    let runtimeMetrics: RuntimeMetrics
}
