import CoreGraphics
import UIKit

extension UIImage {
    func normalizedCGImage() -> CGImage? {
        if imageOrientation == .up, let cgImage, !cgImageHasAlpha(cgImage) {
            return cgImage
        }

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let rendered = renderer.image { _ in
            UIColor.white.setFill()
            UIBezierPath(rect: CGRect(origin: .zero, size: size)).fill()
            draw(in: CGRect(origin: .zero, size: size))
        }
        return rendered.cgImage
    }

    func fittedSquareCGImage() -> CGImage? {
        guard let source = normalizedCGImage() else {
            return nil
        }

        if source.width == source.height {
            return source
        }

        let side = max(source.width, source.height)
        let canvasSize = CGSize(width: side, height: side)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: canvasSize, format: format)
        let rendered = renderer.image { _ in
            UIColor.white.setFill()
            UIBezierPath(rect: CGRect(origin: .zero, size: canvasSize)).fill()

            let imageSize = CGSize(width: source.width, height: source.height)
            let origin = CGPoint(
                x: (canvasSize.width - imageSize.width) / 2,
                y: (canvasSize.height - imageSize.height) / 2
            )
            UIImage(cgImage: source).draw(in: CGRect(origin: origin, size: imageSize))
        }
        return rendered.cgImage
    }

    private func cgImageHasAlpha(_ cgImage: CGImage) -> Bool {
        switch cgImage.alphaInfo {
        case .first, .last, .premultipliedFirst, .premultipliedLast:
            return true
        default:
            return false
        }
    }
}
