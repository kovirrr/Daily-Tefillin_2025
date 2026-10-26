import CoreML
import CoreFoundation
import CoreVideo
import PhotosUI
import SwiftUI
import UIKit

@available(iOS 16.0, *)
struct CoreMLBenchmarkView: View {
  @State private var selectedItem: PhotosPickerItem?
  @State private var selectedImage: UIImage?
  @State private var isRunning = false
  @State private var status = "Choose a photo stored on this iPhone."
  @State private var report = ""
  @State private var tefillinBox: CGRect?

  var body: some View {
    NavigationView {
      ScrollView {
        VStack(alignment: .leading, spacing: 16) {
          Text("iPhone Core ML benchmark")
            .font(.title2.bold())
          Text("Both models run on this iPhone. Photos are not uploaded or saved by this test.")
            .font(.subheadline)

          PhotosPicker(selection: $selectedItem, matching: .images) {
            Text("Choose test photo")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.bordered)

          if let selectedImage {
            Image(uiImage: selectedImage)
              .resizable()
              .scaledToFit()
              .frame(maxHeight: 280)
              .clipShape(RoundedRectangle(cornerRadius: 12))
              .overlay {
                if let detection = tefillinBox {
                  GeometryReader { geometry in
                    let imageAspect = selectedImage.size.width / selectedImage.size.height
                    let displayWidth = min(geometry.size.width, geometry.size.height * imageAspect)
                    let displayHeight = displayWidth / imageAspect
                    let offsetX = (geometry.size.width - displayWidth) / 2
                    let offsetY = (geometry.size.height - displayHeight) / 2

                    Rectangle()
                      .stroke(.green, lineWidth: 3)
                      .frame(
                        width: displayWidth * detection.width,
                        height: displayHeight * detection.height
                      )
                      .position(
                        x: offsetX + displayWidth * (detection.minX + detection.width / 2),
                        y: offsetY + displayHeight * (detection.minY + detection.height / 2)
                      )
                  }
                }
              }

            Button {
              runBenchmark()
            } label: {
              if isRunning {
                ProgressView()
                  .frame(maxWidth: .infinity)
              } else {
                Text("Run both models")
                  .frame(maxWidth: .infinity)
              }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)
          }

          Text(status)
            .font(.callout)
          if !report.isEmpty {
            Text(report)
              .font(.system(.footnote, design: .monospaced))
              .textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
        }
        .padding()
      }
      .navigationTitle("Offline model test")
      .navigationBarTitleDisplayMode(.inline)
    }
    .onChange(of: selectedItem) { newItem in
      Task {
        guard let newItem else { return }
        do {
          guard let data = try await newItem.loadTransferable(type: Data.self),
                let image = UIImage(data: data) else {
            status = "Could not load that photo. Choose another one stored on the phone."
            return
          }
          selectedImage = image
          report = ""
          tefillinBox = nil
          status = "Photo loaded on-device. Turn on Airplane Mode, then run both models."
        } catch {
          status = "Could not load photo: \(error.localizedDescription)"
        }
      }
    }
  }

  private func runBenchmark() {
    guard let selectedImage else { return }
    isRunning = true
    status = "Loading Core ML models and measuring inference…"

    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let result = try CoreMLBenchmarkRunner.run(on: selectedImage)
        DispatchQueue.main.async {
          report = result.report
          tefillinBox = result.tefillinBox
          status = "Finished. Inference ran on-device; no network request was made."
          isRunning = false
        }
      } catch {
        DispatchQueue.main.async {
          status = "Benchmark failed: \(error.localizedDescription)"
          isRunning = false
        }
      }
    }
  }
}

private enum CoreMLBenchmarkRunner {
  private static let warmupRuns = 3
  private static let measuredRuns = 25
  private static let inputSize = 640

  static func run(on image: UIImage) throws -> BenchmarkReport {
    let preprocessStart = ProcessInfo.processInfo.systemUptime
    let (inputImage, transform) = try makeLetterboxedInput(from: image)
    let preprocessMilliseconds = (ProcessInfo.processInfo.systemUptime - preprocessStart) * 1_000
    let imageFeature = try MLFeatureValue(
      cgImage: inputImage,
      pixelsWide: inputSize,
      pixelsHigh: inputSize,
      pixelFormatType: kCVPixelFormatType_32BGRA
    )
    let imageInputs = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])
    let tefillinInputs = try MLDictionaryFeatureProvider(dictionary: [
      "image": imageFeature,
      "iouThreshold": MLFeatureValue(double: 0.45),
      "confidenceThreshold": MLFeatureValue(double: 0.25),
    ])
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all

    let hairlineModel = try loadModel(named: "hairlineAI", configuration: configuration)
    let tefillinModel = try loadModel(named: "best", configuration: configuration)
    let hairlineResult = try measure(model: hairlineModel, named: "hairlineAI", input: imageInputs)
    let tefillinResult = try measure(model: tefillinModel, named: "best", input: tefillinInputs)
    let bestDetection = bestTefillinDetection(
      from: tefillinResult.finalOutput,
      transform: transform
    )

    let report = """
    Device: \(UIDevice.current.model), iOS \(UIDevice.current.systemVersion)
    Available compute units: CPU, GPU, Neural Engine (Core ML chooses)
    Input: \(transform.imageWidth) x \(transform.imageHeight), letterboxed to 640 x 640
    Preprocessing: \(String(format: "%.1f", preprocessMilliseconds)) ms once per photo
    Inference runs: \(warmupRuns) warm-up + \(measuredRuns) measured per model
    Compiled model files: \(String(format: "%.1f", (hairlineResult.modelBytes + tefillinResult.modelBytes) / 1_000_000)) MB

    HAIRLINE SEGMENTATION
    Median inference: \(String(format: "%.1f", hairlineResult.medianMilliseconds)) ms
    P95 inference: \(String(format: "%.1f", hairlineResult.p95Milliseconds)) ms
    Output: \(hairlineResult.outputSummary)

    TEFILLIN DETECTION
    Median inference: \(String(format: "%.1f", tefillinResult.medianMilliseconds)) ms
    P95 inference: \(String(format: "%.1f", tefillinResult.p95Milliseconds)) ms
    Output: \(bestDetection.summary)

    Timings measure model inference only. The green box is Core ML's highest-confidence detection.
    This does not test MediaPipe or the full tef_good() check. Validate detections across varied photos.
    """
    return BenchmarkReport(report: report, tefillinBox: bestDetection.normalizedBox)
  }

  private static func loadModel(named name: String, configuration: MLModelConfiguration) throws -> MLModel {
    guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
      throw BenchmarkError.modelNotBundled(name)
    }
    return try MLModel(contentsOf: url, configuration: configuration)
  }

  private static func measure(
    model: MLModel,
    named name: String,
    input: MLFeatureProvider
  ) throws -> ModelMeasurement {
    for _ in 0..<warmupRuns {
      _ = try model.prediction(from: input)
    }

    var durations: [Double] = []
    var finalOutput: MLFeatureProvider?
    for _ in 0..<measuredRuns {
      let start = ProcessInfo.processInfo.systemUptime
      finalOutput = try model.prediction(from: input)
      durations.append((ProcessInfo.processInfo.systemUptime - start) * 1_000)
    }

    guard let finalOutput else {
      throw BenchmarkError.noPrediction
    }
    durations.sort()
    let median = durations[durations.count / 2]
    let p95Index = min(durations.count - 1, Int((Double(durations.count) * 0.95).rounded(.up)) - 1)
    return ModelMeasurement(
      medianMilliseconds: median,
      p95Milliseconds: durations[p95Index],
      modelBytes: compiledModelSize(named: name),
      outputSummary: summarize(finalOutput),
      finalOutput: finalOutput
    )
  }

  private static func summarize(_ output: MLFeatureProvider) -> String {
    let arrays = output.featureNames.sorted().compactMap { name -> (String, MLMultiArray)? in
      guard let array = output.featureValue(for: name)?.multiArrayValue else { return nil }
      return (name, array)
    }
    let shapes = arrays.map { name, array in
      "\(name):[\(array.shape.map(\.stringValue).joined(separator: ","))]"
    }.joined(separator: ", ")
    guard let raw = arrays.map(\.1).first(where: {
      $0.shape.count == 3 && $0.shape[1].intValue >= 5
    }) else {
      return shapes.isEmpty ? "no multi-array outputs" : shapes
    }
    var scores: [Double] = []
    for index in 0..<raw.shape[2].intValue {
      scores.append(raw[
        [NSNumber(value: 0), NSNumber(value: 4), NSNumber(value: index)]
      ].doubleValue)
    }
    return "\(shapes); max class confidence: \(String(format: "%.3f", scores.max() ?? 0)), raw candidates >= 0.25: \(scores.filter { $0 >= 0.25 }.count)"
  }

  private static func makeLetterboxedInput(from image: UIImage) throws -> (CGImage, LetterboxTransform) {
    let sourceSize = image.size
    guard sourceSize.width > 0, sourceSize.height > 0 else {
      throw BenchmarkError.imageUnavailable
    }
    let scale = min(
      CGFloat(inputSize) / sourceSize.width,
      CGFloat(inputSize) / sourceSize.height
    )
    let resizedWidth = (sourceSize.width * scale).rounded(.toNearestOrEven)
    let resizedHeight = (sourceSize.height * scale).rounded(.toNearestOrEven)
    let paddingX = ((CGFloat(inputSize) - resizedWidth) / 2 - 0.1).rounded(.toNearestOrEven)
    let paddingY = ((CGFloat(inputSize) - resizedHeight) / 2 - 0.1).rounded(.toNearestOrEven)

    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    format.opaque = true
    let renderer = UIGraphicsImageRenderer(
      size: CGSize(width: inputSize, height: inputSize),
      format: format
    )
    let renderedImage = renderer.image { context in
      UIColor(
        red: 114.0 / 255.0,
        green: 114.0 / 255.0,
        blue: 114.0 / 255.0,
        alpha: 1
      ).setFill()
      context.fill(CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
      image.draw(in: CGRect(
        x: paddingX,
        y: paddingY,
        width: resizedWidth,
        height: resizedHeight
      ))
    }
    guard let cgImage = renderedImage.cgImage else {
      throw BenchmarkError.imageUnavailable
    }

    return (
      cgImage,
      LetterboxTransform(
        scale: scale,
        paddingX: paddingX,
        paddingY: paddingY,
        imageWidth: Int(sourceSize.width),
        imageHeight: Int(sourceSize.height)
      )
    )
  }

  private static func bestTefillinDetection(
    from output: MLFeatureProvider,
    transform: LetterboxTransform
  ) -> DetectionSummary {
    guard let confidence = output.featureValue(for: "confidence")?.multiArrayValue,
          let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue,
          confidence.count > 0, coordinates.count >= 4 else {
      return DetectionSummary(summary: "no tefillin detected", normalizedBox: nil)
    }

    let score = confidence[0].doubleValue
    guard score >= 0.25 else {
      return DetectionSummary(summary: "no tefillin detected", normalizedBox: nil)
    }
    let centerX = coordinates[0].doubleValue * Double(inputSize)
    let centerY = coordinates[1].doubleValue * Double(inputSize)
    let width = coordinates[2].doubleValue * Double(inputSize)
    let height = coordinates[3].doubleValue * Double(inputSize)
    let x1 = min(Double(transform.imageWidth), max(0, (centerX - width / 2 - transform.paddingX) / transform.scale))
    let y1 = min(Double(transform.imageHeight), max(0, (centerY - height / 2 - transform.paddingY) / transform.scale))
    let x2 = min(Double(transform.imageWidth), max(0, (centerX + width / 2 - transform.paddingX) / transform.scale))
    let y2 = min(Double(transform.imageHeight), max(0, (centerY + height / 2 - transform.paddingY) / transform.scale))
    let box = CGRect(
      x: CGFloat(x1 / Double(transform.imageWidth)),
      y: CGFloat(y1 / Double(transform.imageHeight)),
      width: CGFloat(max(0, x2 - x1) / Double(transform.imageWidth)),
      height: CGFloat(max(0, y2 - y1) / Double(transform.imageHeight))
    )
    return DetectionSummary(
      summary: "confidence: \(String(format: "%.3f", score))",
      normalizedBox: box
    )
  }

  private static func compiledModelSize(named name: String) -> Double {
    guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
          let files = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey]
          ) else {
      return 0
    }
    var totalBytes: Int64 = 0
    for case let fileURL as URL in files {
      totalBytes += Int64((try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
    return Double(totalBytes)
  }
}

private struct ModelMeasurement {
  let medianMilliseconds: Double
  let p95Milliseconds: Double
  let modelBytes: Double
  let outputSummary: String
  let finalOutput: MLFeatureProvider
}

private struct LetterboxTransform {
  let scale: CGFloat
  let paddingX: CGFloat
  let paddingY: CGFloat
  let imageWidth: Int
  let imageHeight: Int
}

private struct DetectionSummary {
  let summary: String
  let normalizedBox: CGRect?
}

private struct BenchmarkReport {
  let report: String
  let tefillinBox: CGRect?
}

private enum BenchmarkError: LocalizedError {
  case imageUnavailable
  case modelNotBundled(String)
  case noPrediction

  var errorDescription: String? {
    switch self {
    case .imageUnavailable:
      return "The selected photo has no readable image data."
    case .modelNotBundled(let name):
      return "The compiled Core ML model \(name) is missing from the app."
    case .noPrediction:
      return "Core ML returned no prediction results."
    }
  }
}
