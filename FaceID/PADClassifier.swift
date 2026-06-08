import CoreML
import CoreImage
import CoreVideo
import Foundation

/// 反欺骗(PAD)分类器:在人脸区域(带 margin,含屏幕边/手等上下文)跑 MobileNetV3,
/// 输出「假体概率」。与 ArcFace 不同:输入 224×224、**ImageNet 归一化**、不做对齐。
/// 模型文件缺失(还没训好导入)时 init 返回 nil,上层自动降级为纯深度活体。
final class PADClassifier {
    private let model: MLModel
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let size = 224
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std:  [Float] = [0.229, 0.224, 0.225]

    init?() {
        guard let url = Bundle.main.url(forResource: "FaceSpoofMBV3", withExtension: "mlmodelc"),
              let m = try? MLModel(contentsOf: url) else { return nil }
        model = m
    }

    /// spoof 概率(0~1,越大越像假体)。normalizedBox: Vision 框(左下原点,归一化)。
    func spoofProbability(pixelBuffer: CVPixelBuffer, normalizedBox box: CGRect, margin: CGFloat = 0.4) -> Float? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let W = ci.extent.width, H = ci.extent.height
        let mx = box.width * margin, my = box.height * margin
        var rect = CGRect(x: (box.minX - mx) * W, y: (box.minY - my) * H,
                          width: (box.width + 2*mx) * W, height: (box.height + 2*my) * H)
        rect = rect.intersection(ci.extent)
        guard !rect.isNull, rect.width > 8, rect.height > 8 else { return nil }
        let cropped = ci.cropped(to: rect)
        let toOrigin = cropped.transformed(
            by: CGAffineTransform(translationX: -cropped.extent.origin.x, y: -cropped.extent.origin.y))
        let scaled = toOrigin.transformed(
            by: CGAffineTransform(scaleX: CGFloat(size)/cropped.extent.width,
                                  y: CGFloat(size)/cropped.extent.height))
        guard let cg = ciContext.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: size, height: size)),
              let input = preprocess(cg),
              let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: input)]),
              let out = try? model.prediction(from: provider),
              let arr = out.featureValue(for: "logits")?.multiArrayValue, arr.count >= 2 else { return nil }
        let l0 = Float(truncating: arr[0]), l1 = Float(truncating: arr[1])
        let m = max(l0, l1)
        let e0 = expf(l0 - m), e1 = expf(l1 - m)
        return e1 / (e0 + e1)            // softmax 的 spoof(类 1)概率
    }

    private func preprocess(_ cg: CGImage) -> MLMultiArray? {
        let s = size
        var rgba = [UInt8](repeating: 0, count: s*s*4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &rgba, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: 4*s, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: s, height: s))
        guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: s), NSNumber(value: s)],
                                          dataType: .float32) else { return nil }
        let plane = s*s
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: 3*plane)
        for p in 0..<plane {
            ptr[0*plane+p] = (Float32(rgba[p*4])   / 255 - mean[0]) / std[0]
            ptr[1*plane+p] = (Float32(rgba[p*4+1]) / 255 - mean[1]) / std[1]
            ptr[2*plane+p] = (Float32(rgba[p*4+2]) / 255 - mean[2]) / std[2]
        }
        return arr
    }
}
