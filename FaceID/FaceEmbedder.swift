import CoreML
import CoreImage
import CoreVideo

/// 用 Core ML 跑 ArcFace(r50,自训 Glint360K + 鲁棒增广,2026-06-08 换上)。
/// 优先走 **5 点对齐**(Vision 关键点 → 相似变换到 ArcFace 标准模板 → 112×112),
/// 对齐能显著提升同人分数(尤其侧脸/眼镜);拿不到关键点时退回纯框裁剪。
/// 输出未归一化;比对由 C++ 引擎做 L2 归一化 + 余弦。
final class FaceEmbedder {

    private let model: MLModel
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let inputSize = 112

    /// ArcFace 112 标准模板,已翻成「左下原点」以匹配 Vision/CIImage 坐标系。
    /// 顺序:左眼、右眼、鼻、左嘴角、右嘴角(112 - 原 top-left y)。
    private let dstTemplate: [CGPoint] = [
        CGPoint(x: 38.2946, y: 112 - 51.6963),
        CGPoint(x: 73.5318, y: 112 - 51.5014),
        CGPoint(x: 56.0252, y: 112 - 71.7366),
        CGPoint(x: 41.5493, y: 112 - 92.3655),
        CGPoint(x: 70.7299, y: 112 - 92.2041),
    ]

    init?() {
        guard let url = Bundle.main.url(forResource: "ArcFaceR50", withExtension: "mlmodelc"),
              let m = try? MLModel(contentsOf: url) else { return nil }
        model = m
    }

    /// 对齐路径:src5 = 5 个关键点(图像像素坐标,左下原点),顺序同 dstTemplate。
    func embedAligned(pixelBuffer: CVPixelBuffer, src5 pts: [CGPoint]) -> [Float]? {
        guard pts.count == 5 else { return nil }

        // 闭式最小二乘相似变换(src -> dst):x' = a·x - b·y + tx ; y' = b·x + a·y + ty
        var mx = 0.0, my = 0.0, Mx = 0.0, My = 0.0
        for i in 0..<5 {
            mx += Double(pts[i].x); my += Double(pts[i].y)
            Mx += Double(dstTemplate[i].x); My += Double(dstTemplate[i].y)
        }
        mx /= 5; my /= 5; Mx /= 5; My /= 5
        var sxx = 0.0, sa = 0.0, sb = 0.0
        for i in 0..<5 {
            let xp = Double(pts[i].x) - mx, yp = Double(pts[i].y) - my
            let Xp = Double(dstTemplate[i].x) - Mx, Yp = Double(dstTemplate[i].y) - My
            sxx += xp * xp + yp * yp
            sa += xp * Xp + yp * Yp
            sb += xp * Yp - yp * Xp
        }
        guard sxx > 1e-6 else { return nil }
        let a = sa / sxx, b = sb / sxx
        let tx = Mx - (a * mx - b * my), ty = My - (b * mx + a * my)
        let t = CGAffineTransform(a: a, b: b, c: -b, d: a, tx: tx, ty: ty)

        let aligned = CIImage(cvPixelBuffer: pixelBuffer).transformed(by: t)
        guard let cg = ciContext.createCGImage(
            aligned, from: CGRect(x: 0, y: 0, width: inputSize, height: inputSize)) else { return nil }
        // 测试时翻转增强(TTA):脸 + 水平镜像各编码取平均 → 嵌入更稳。引擎再 L2 归一化。
        guard let i0 = preprocess(cgImage: cg, flip: false), let v0 = run(i0) else { return nil }
        guard let i1 = preprocess(cgImage: cg, flip: true), let v1 = run(i1) else { return v0 }
        return zip(v0, v1).map { ($0 + $1) * 0.5 }
    }

    /// 兜底路径:无关键点时按 Vision 框裁剪(左下原点归一化框)。
    func embed(pixelBuffer: CVPixelBuffer, normalizedBox: CGRect) -> [Float]? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let W = ci.extent.width, H = ci.extent.height
        var rect = CGRect(x: normalizedBox.minX * W, y: normalizedBox.minY * H,
                          width: normalizedBox.width * W, height: normalizedBox.height * H)
        rect = rect.intersection(ci.extent)
        guard !rect.isNull, rect.width > 4, rect.height > 4 else { return nil }
        let cropped = ci.cropped(to: rect)
        let toOrigin = cropped.transformed(
            by: CGAffineTransform(translationX: -cropped.extent.origin.x, y: -cropped.extent.origin.y))
        let scaled = toOrigin.transformed(
            by: CGAffineTransform(scaleX: CGFloat(inputSize) / cropped.extent.width,
                                  y: CGFloat(inputSize) / cropped.extent.height))
        guard let cg = ciContext.createCGImage(
            scaled, from: CGRect(x: 0, y: 0, width: inputSize, height: inputSize)) else { return nil }
        guard let input = preprocess(cgImage: cg) else { return nil }
        return run(input)
    }

    // MARK: - 共享:预处理 + 推理

    private func run(_ input: MLMultiArray) -> [Float]? {
        guard let provider = try? MLDictionaryFeatureProvider(
                dictionary: ["input": MLFeatureValue(multiArray: input)]),
              let out = try? model.prediction(from: provider),
              let arr = out.featureValue(for: "embedding")?.multiArrayValue else { return nil }
        return (0..<arr.count).map { Float(truncating: arr[$0]) }
    }

    /// 112×112 CGImage → RGB → (x-127.5)/127.5 → MLMultiArray[1,3,112,112] (NCHW)。flip=水平镜像(TTA)。
    private func preprocess(cgImage: CGImage, flip: Bool = false) -> MLMultiArray? {
        let s = inputSize
        var rgba = [UInt8](repeating: 0, count: s * s * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &rgba, width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: 4 * s, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: s, height: s))
        guard let arr = try? MLMultiArray(
            shape: [1, 3, NSNumber(value: s), NSNumber(value: s)], dataType: .float32) else { return nil }
        let plane = s * s
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: 3 * plane)
        for row in 0..<s {
            for col in 0..<s {
                let src = (row * s + (flip ? (s - 1 - col) : col)) * 4   // 翻转=读镜像列
                let dst = row * s + col
                ptr[0 * plane + dst] = (Float32(rgba[src])     - 127.5) / 127.5
                ptr[1 * plane + dst] = (Float32(rgba[src + 1]) - 127.5) / 127.5
                ptr[2 * plane + dst] = (Float32(rgba[src + 2]) - 127.5) / 127.5
            }
        }
        return arr
    }
}
