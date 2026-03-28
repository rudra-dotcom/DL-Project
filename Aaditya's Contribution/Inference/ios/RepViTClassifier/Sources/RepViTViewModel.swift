import Foundation
import UIKit

@MainActor
final class RepViTViewModel: ObservableObject {
    @Published private(set) var selectedImage: UIImage?
    @Published private(set) var selectedImageSource = "Demo image"
    @Published private(set) var topPredictions: [ClassPrediction] = []
    @Published private(set) var runtimeMetrics: RuntimeMetrics?
    @Published private(set) var isLoading = false
    @Published private(set) var errorMessage: String?

    let modelProfile = RepViTModelProfile.bundled
    private var engine: RepViTPredictionEngine?

    func loadBundledTestImage() async {
        guard let imageURL = Bundle.main.url(forResource: "test_image_corgi", withExtension: "jpg"),
              let image = UIImage(contentsOfFile: imageURL.path) else {
            errorMessage = "The bundled demo image could not be loaded."
            return
        }

        await classify(image: image, sourceLabel: "Demo image")
    }

    func classifySelectedImage(_ image: UIImage?) async {
        await classify(image: image, sourceLabel: "Uploaded image")
    }

    private func classify(image: UIImage?, sourceLabel: String) async {
        guard let image else {
            return
        }
        isLoading = true
        errorMessage = nil

        do {
            selectedImage = image
            selectedImageSource = sourceLabel

            if engine == nil {
                engine = try RepViTPredictionEngine()
            }

            guard let engine else {
                throw RepViTPredictionError.modelUnavailable
            }

            let result = try await engine.classify(image: image, inputSourceLabel: sourceLabel)
            topPredictions = result.topPredictions
            runtimeMetrics = result.runtimeMetrics
        } catch {
            topPredictions = []
            runtimeMetrics = nil
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }
}
