import SwiftUI

struct ContentView: View {
    @AppStorage(AppAppearance.storageKey) private var appearanceMode = AppAppearance.system.rawValue
    @Environment(\.colorScheme) private var colorScheme
    @StateObject private var viewModel = RepViTViewModel()
    @State private var isPhotoPickerPresented = false

    private let metricColumns = [
        GridItem(.flexible(), spacing: 14),
        GridItem(.flexible(), spacing: 14),
    ]

    var body: some View {
        ZStack {
            LinearGradient(
                colors: theme.backgroundGradient,
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    header
                    pickerCard
                    previewCard
                    resultCard
                    technicalDetailsCard
                }
                .padding(20)
            }
        }
        .sheet(isPresented: $isPhotoPickerPresented) {
            PhotoLibraryPicker { image in
                Task {
                    await viewModel.classifySelectedImage(image)
                }
            }
        }
        .task {
            guard ProcessInfo.processInfo.arguments.contains("-repvitLoadDemoImage") else {
                return
            }

            await viewModel.loadBundledTestImage()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("RepViT on iPhone")
                .font(.system(size: 34, weight: .black, design: .rounded))
                .foregroundStyle(theme.titleColor)

            Text("Load the demo image or choose your own photo. The app runs the model on-device and shows the best match for that single image.")
                .font(.system(.body, design: .rounded))
                .foregroundStyle(theme.primaryTextColor.opacity(0.78))

            HStack(spacing: 10) {
                StatCapsule(label: viewModel.modelProfile.displayName)
                StatCapsule(label: "Core ML")
                StatCapsule(label: "On-device")
            }

            appearancePicker
        }
    }

    private var appearancePicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Appearance")
                .font(.system(.caption, design: .rounded).weight(.bold))
                .foregroundStyle(theme.secondaryTextColor)

            HStack(spacing: 10) {
                ForEach(AppAppearance.allCases) { mode in
                    Button {
                        withAnimation(.snappy(duration: 0.25)) {
                            appearanceMode = mode.rawValue
                        }
                    } label: {
                        Text(mode.title)
                            .font(.system(.subheadline, design: .rounded).weight(.bold))
                            .foregroundStyle(selectedAppearance == mode ? .white : theme.titleColor)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(selectedAppearance == mode ? theme.accentColor : theme.secondarySurface)
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var pickerCard: some View {
        card {
            VStack(alignment: .leading, spacing: 14) {
                Text("Image Input")
                    .font(.system(.title3, design: .rounded).weight(.bold))

                HStack(spacing: 12) {
                    actionButton(
                        title: "Upload Photo",
                        systemImage: "photo.badge.plus",
                        fillColor: theme.accentColor
                    ) {
                        isPhotoPickerPresented = true
                    }

                    actionButton(
                        title: "Load Demo Image",
                        systemImage: "sparkles",
                        fillColor: theme.secondaryActionColor
                    ) {
                        Task {
                            await viewModel.loadBundledTestImage()
                        }
                    }
                }

                if let errorMessage = viewModel.errorMessage {
                    Text(errorMessage)
                        .font(.system(.footnote, design: .rounded))
                        .foregroundStyle(Color.red.opacity(0.85))
                }

                Text("Upload Photo opens your photo library and runs the model on the one image you choose. Load Demo Image uses the sample image bundled inside the app.")
                    .font(.system(.footnote, design: .rounded))
                    .foregroundStyle(theme.primaryTextColor.opacity(0.74))
            }
            .foregroundStyle(theme.titleColor)
        }
    }

    private var previewCard: some View {
        card {
            VStack(alignment: .leading, spacing: 14) {
                Text("Preview")
                    .font(.system(.title3, design: .rounded).weight(.bold))

                Group {
                    if let image = viewModel.selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                            .frame(height: 280)
                            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                    } else {
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .stroke(style: StrokeStyle(lineWidth: 1.5, dash: [8, 8]))
                            .foregroundStyle(theme.secondaryTextColor.opacity(0.45))
                            .frame(height: 220)
                            .overlay {
                                VStack(spacing: 10) {
                                    Image(systemName: "sparkles.rectangle.stack")
                                        .font(.system(size: 32))
                                    Text("Load the demo image or choose your own photo.")
                                        .font(.system(.headline, design: .rounded))
                                }
                                .foregroundStyle(theme.secondaryTextColor)
                            }
                    }
                }

                if viewModel.selectedImage != nil {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundStyle(theme.accentColor)
                        Text("Showing results for \(viewModel.selectedImageSource)")
                            .font(.system(.subheadline, design: .rounded).weight(.semibold))
                            .foregroundStyle(theme.primaryTextColor.opacity(0.78))
                    }
                }

                if viewModel.isLoading {
                    HStack(spacing: 12) {
                        ProgressView()
                        Text("Running Core ML inference...")
                            .font(.system(.subheadline, design: .rounded))
                    }
                }
            }
            .foregroundStyle(theme.titleColor)
        }
    }

    private var resultCard: some View {
        card {
            VStack(alignment: .leading, spacing: 16) {
                Text("Result")
                    .font(.system(.title3, design: .rounded).weight(.bold))

                if let best = viewModel.topPredictions.first {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Best Match")
                            .font(.system(.caption, design: .rounded).weight(.bold))
                            .foregroundStyle(theme.secondaryTextColor)

                        Text(best.label)
                            .font(.system(size: 30, weight: .black, design: .rounded))
                            .lineLimit(2)

                        Text("Model score: \(best.confidence.formatted(.percent.precision(.fractionLength(1))))")
                            .font(.system(.subheadline, design: .rounded).weight(.semibold))
                            .foregroundStyle(theme.accentColor)
                    }

                    HStack(spacing: 12) {
                        resultChip(title: "Source", value: viewModel.selectedImageSource)

                        if let runtime = viewModel.runtimeMetrics {
                            resultChip(
                                title: "Speed",
                                value: "\(runtime.predictionLatencyMs.formatted(.number.precision(.fractionLength(2)))) ms"
                            )
                        }
                    }

                    Text("The model compares your photo against 1,000 ImageNet labels. This score shows which label it preferred most, but cut-out graphics, checkerboard backgrounds, and non-photo images can still confuse it.")
                        .font(.system(.footnote, design: .rounded))
                        .foregroundStyle(theme.primaryTextColor.opacity(0.78))

                    if viewModel.topPredictions.count > 1 {
                        DisclosureGroup {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("These are backup guesses for the same photo, not extra images.")
                                    .font(.system(.footnote, design: .rounded))
                                    .foregroundStyle(theme.secondaryTextColor)

                                otherMatchesList
                            }
                            .padding(.top, 8)
                        } label: {
                            Text("Show other possible matches")
                                .font(.system(.subheadline, design: .rounded).weight(.bold))
                        }
                        .tint(theme.titleColor)
                    }
                } else {
                    Text("Load the demo image or choose your own photo to see the model's best guess.")
                        .font(.system(.body, design: .rounded))
                        .foregroundStyle(theme.secondaryTextColor)
                }
            }
            .foregroundStyle(theme.titleColor)
        }
    }

    private var technicalDetailsCard: some View {
        card {
            VStack(alignment: .leading, spacing: 16) {
                Text("Technical Details")
                    .font(.system(.title3, design: .rounded).weight(.bold))

                Text("This section is optional. It explains the ImageNet label set, benchmark accuracy, and model size details.")
                    .font(.system(.footnote, design: .rounded))
                    .foregroundStyle(theme.primaryTextColor.opacity(0.74))

                DisclosureGroup {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("ImageNet is the fixed list of 1,000 labels this model was trained to recognize. The numbers below are model benchmarks and live runtime info, not extra answers for your photo.")
                            .font(.system(.footnote, design: .rounded))
                            .foregroundStyle(theme.primaryTextColor.opacity(0.78))

                        metricsGrid

                        Text(viewModel.modelProfile.publishedLatencyContext)
                            .font(.system(.footnote, design: .rounded))
                            .foregroundStyle(theme.primaryTextColor.opacity(0.65))
                    }
                    .padding(.top, 8)
                } label: {
                    Text("Show advanced model info")
                        .font(.system(.subheadline, design: .rounded).weight(.bold))
                }
                .tint(theme.titleColor)
            }
            .foregroundStyle(theme.titleColor)
        }
    }

    private var metricsGrid: some View {
        let metrics = dashboardMetrics()
        return LazyVGrid(columns: metricColumns, spacing: 14) {
            ForEach(metrics, id: \.title) { metric in
                metricTile(metric)
            }
        }
    }

    private var otherMatchesList: some View {
        VStack(spacing: 12) {
            ForEach(Array(viewModel.topPredictions.dropFirst())) { prediction in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("#\(prediction.rank)")
                            .font(.system(.caption, design: .rounded).weight(.bold))
                            .foregroundStyle(theme.secondaryTextColor)
                        Text(prediction.label)
                            .font(.system(.headline, design: .rounded))
                            .lineLimit(1)
                        Spacer()
                        Text(prediction.confidence.formatted(.percent.precision(.fractionLength(1))))
                            .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    }

                    GeometryReader { proxy in
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(theme.progressTrackColor)
                            Capsule()
                                .fill(
                                    LinearGradient(
                                        colors: [
                                            Color(red: 0.11, green: 0.55, blue: 0.49),
                                            Color(red: 0.16, green: 0.37, blue: 0.54),
                                        ],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: proxy.size.width * prediction.confidence)
                        }
                    }
                    .frame(height: 10)
                }
            }
        }
    }

    private func dashboardMetrics() -> [MetricEntry] {
        let profile = viewModel.modelProfile
        let runtime = viewModel.runtimeMetrics

        return [
            MetricEntry(
                title: "Published Top-1",
                value: "\(profile.top1Accuracy.formatted(.number.precision(.fractionLength(1))))%",
                detail: "Best single-label benchmark from the repo"
            ),
            MetricEntry(
                title: "Published Top-5",
                value: "\(profile.top5Accuracy.formatted(.number.precision(.fractionLength(2))))%",
                detail: "How often the right label is in the first 5 guesses"
            ),
            MetricEntry(
                title: "Live Latency",
                value: runtime.map { "\($0.predictionLatencyMs.formatted(.number.precision(.fractionLength(2)))) ms" } ?? "\(profile.publishedLatencyMs.formatted(.number.precision(.fractionLength(1)))) ms",
                detail: runtime == nil ? "Published iPhone 12 benchmark" : "Measured live on this device"
            ),
            MetricEntry(
                title: "Parameters",
                value: profile.parameterCountLabel,
                detail: "Learned weights in the model"
            ),
            MetricEntry(
                title: "MACs",
                value: profile.macsLabel,
                detail: "Approximate compute cost from the repo"
            ),
            MetricEntry(
                title: "Input Size",
                value: "\(Int(profile.inputResolution.width)) x \(Int(profile.inputResolution.height))",
                detail: "Image size expected by the model"
            ),
            MetricEntry(
                title: "Input Source",
                value: runtime?.inputSourceLabel ?? viewModel.selectedImageSource,
                detail: "Demo image or your uploaded photo"
            ),
            MetricEntry(
                title: "Selected Image",
                value: runtime.map { "\(Int($0.selectedImageSize.width)) x \(Int($0.selectedImageSize.height))" } ?? "Waiting",
                detail: "Photo size before it is resized for the model"
            ),
            MetricEntry(
                title: "Model Size",
                value: runtime?.compiledModelSizeLabel ?? "Loading",
                detail: "Compiled Core ML size in the app bundle"
            ),
        ]
    }

    private func resultChip(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title.uppercased())
                .font(.system(.caption2, design: .rounded).weight(.bold))
                .foregroundStyle(theme.secondaryTextColor)
            Text(value)
                .font(.system(.subheadline, design: .rounded).weight(.bold))
                .foregroundStyle(theme.titleColor)
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(theme.secondarySurface)
        )
    }

    private func metricTile(_ metric: MetricEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(metric.title.uppercased())
                .font(.system(.caption, design: .rounded).weight(.bold))
                .foregroundStyle(theme.secondaryTextColor)

            Text(metric.value)
                .font(.system(.title3, design: .rounded).weight(.black))
                .foregroundStyle(theme.titleColor)

            Text(metric.detail)
                .font(.system(.footnote, design: .rounded))
                .foregroundStyle(theme.primaryTextColor.opacity(0.7))
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(theme.secondarySurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .stroke(theme.cardBorder.opacity(0.8), lineWidth: 1)
                )
        )
    }

    private func card<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(theme.cardFill)
                    .overlay(
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .stroke(theme.cardBorder, lineWidth: 1)
                    )
            )
            .shadow(color: theme.shadowColor, radius: 18, x: 0, y: 10)
    }

    private func actionButton(title: String, systemImage: String, fillColor: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: systemImage)
                    .font(.headline)
                Text(title)
                    .font(.system(.subheadline, design: .rounded).weight(.bold))
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(fillColor)
            )
        }
        .buttonStyle(.plain)
    }

    private var selectedAppearance: AppAppearance {
        AppAppearance.launchOverride ?? AppAppearance(rawValue: appearanceMode) ?? .system
    }

    private var theme: VisualTheme {
        VisualTheme(colorScheme: selectedAppearance.colorScheme ?? colorScheme)
    }
}

private struct MetricEntry {
    let title: String
    let value: String
    let detail: String
}

private struct StatCapsule: View {
    let label: String

    var body: some View {
        Text(label)
            .font(.system(.caption, design: .rounded).weight(.bold))
            .foregroundStyle(Color(red: 0.10, green: 0.34, blue: 0.31))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(0.6))
            )
    }
}

private struct VisualTheme {
    let backgroundGradient: [Color]
    let cardFill: Color
    let cardBorder: Color
    let shadowColor: Color
    let titleColor: Color
    let primaryTextColor: Color
    let secondaryTextColor: Color
    let accentColor: Color
    let secondaryActionColor: Color
    let secondarySurface: Color
    let progressTrackColor: Color

    init(colorScheme: ColorScheme) {
        if colorScheme == .dark {
            backgroundGradient = [
                Color(red: 0.07, green: 0.10, blue: 0.15),
                Color(red: 0.06, green: 0.20, blue: 0.20),
                Color(red: 0.14, green: 0.28, blue: 0.32),
            ]
            cardFill = Color(red: 0.15, green: 0.20, blue: 0.22).opacity(0.94)
            cardBorder = Color.white.opacity(0.12)
            shadowColor = Color.black.opacity(0.35)
            titleColor = Color(red: 0.93, green: 0.96, blue: 0.97)
            primaryTextColor = Color.white
            secondaryTextColor = Color.white.opacity(0.68)
            accentColor = Color(red: 0.19, green: 0.57, blue: 0.51)
            secondaryActionColor = Color(red: 0.23, green: 0.39, blue: 0.61)
            secondarySurface = Color.white.opacity(0.08)
            progressTrackColor = Color.white.opacity(0.12)
        } else {
            backgroundGradient = [
                Color(red: 0.95, green: 0.92, blue: 0.85),
                Color(red: 0.84, green: 0.90, blue: 0.86),
                Color(red: 0.70, green: 0.82, blue: 0.84),
            ]
            cardFill = Color.white.opacity(0.82)
            cardBorder = Color.white.opacity(0.7)
            shadowColor = Color.black.opacity(0.08)
            titleColor = Color(red: 0.08, green: 0.20, blue: 0.23)
            primaryTextColor = Color(red: 0.16, green: 0.25, blue: 0.27)
            secondaryTextColor = Color(red: 0.16, green: 0.25, blue: 0.27).opacity(0.72)
            accentColor = Color(red: 0.10, green: 0.34, blue: 0.31)
            secondaryActionColor = Color(red: 0.22, green: 0.41, blue: 0.58)
            secondarySurface = Color.white.opacity(0.55)
            progressTrackColor = Color(red: 0.20, green: 0.29, blue: 0.31).opacity(0.12)
        }
    }
}

#Preview {
    ContentView()
}
