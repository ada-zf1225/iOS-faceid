import AVFoundation
import Vision
import SwiftUI
import UIKit

/// 一张被识别的人脸:框(屏幕坐标)+ 标签 + 是否认出
struct RecognizedFace: Identifiable {
    let id = UUID()
    let box: CGRect
    let label: String
    let recognized: Bool
}

/// 相机 + 检测 + 识别核心。
/// 流程:相机帧(已随朝向转正+镜像)→ Vision 检测人脸 → 逐张裁剪 → FaceNet 编码
///       → C++ 引擎余弦比对 → 标签(✅姓名 / ❓陌生人)。识别管线节流到 ~7fps。
final class CameraModel: NSObject, ObservableObject {

    @Published var faces: [RecognizedFace] = []
    @Published var enrolledCount: Int = 0
    @Published var pendingEnroll: [Float]? = nil   // 待命名的人脸向量(点了录入后)

    /// 余弦相似度 ≥ 此值判为同一人。ArcFace 余弦整体偏低、陌生人≈0.1,
    /// 默认 0.35(给眼镜/姿态留余量),真机用滑块实时标定。
    @Published var threshold: Float = 0.35

    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "faceid.camera.session")
    private let frameQueue = DispatchQueue(label: "faceid.camera.frames")

    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var device: AVCaptureDevice?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?

    private let embedder = FaceEmbedder()
    private let engine: FaceEngineBridge
    private var captureNext = false
    private var lastProcess = Date.distantPast
    private let interval: TimeInterval = 0.14   // ~7fps 识别(预览是另一条流,不受影响)

    override init() {
        let dir = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first
            ?? NSTemporaryDirectory()
        let dbPath = (dir as NSString).appendingPathComponent("faces.db")
        engine = FaceEngineBridge(dbPath: dbPath)
        super.init()
        enrolledCount = engine.count()
    }

    func setPreviewLayer(_ layer: AVCaptureVideoPreviewLayer) {
        previewLayer = layer
        sessionQueue.async { self.setupRotationCoordinatorIfReady() }
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard granted, let self else { return }
            self.sessionQueue.async { self.configure() }
        }
    }

    /// 点「录入」:下一帧把最大的人脸向量交给界面去命名
    func requestEnroll() { captureNext = true }

    /// 命名确认后真正录入
    func enroll(name: String) {
        guard let vec = pendingEnroll, !name.isEmpty else { pendingEnroll = nil; return }
        engine.enrollName(name, embedding: vec.map { NSNumber(value: $0) })
        enrolledCount = engine.count()
        pendingEnroll = nil
    }

    func cancelEnroll() { pendingEnroll = nil }

    /// 清空整个录入库
    func resetDB() {
        engine.clear()
        enrolledCount = 0
    }

    // MARK: - 相机配置

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .high

        let cam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video)
        if let cam, let input = try? AVCaptureDeviceInput(device: cam), session.canAddInput(input) {
            session.addInput(input)
            device = cam
        }

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: frameQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        if let conn = videoOutput.connection(with: .video), conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = true   // 前置镜像,和镜像的自拍预览一致
        }

        session.commitConfiguration()
        session.startRunning()
        setupRotationCoordinatorIfReady()
    }

    private func setupRotationCoordinatorIfReady() {
        guard let device, previewLayer != nil, rotationCoordinator == nil else { return }
        let coord = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        rotationCoordinator = coord
        applyRotation()
        rotationObservation = coord.observe(\.videoRotationAngleForHorizonLevelPreview,
                                            options: [.new]) { [weak self] _, _ in
            self?.applyRotation()
        }
    }

    private func applyRotation() {
        guard let coord = rotationCoordinator else { return }
        let angle = coord.videoRotationAngleForHorizonLevelPreview
        DispatchQueue.main.async {
            if let conn = self.previewLayer?.connection, conn.isVideoRotationAngleSupported(angle) {
                conn.videoRotationAngle = angle
            }
        }
        sessionQueue.async {
            if let conn = self.videoOutput.connection(with: .video), conn.isVideoRotationAngleSupported(angle) {
                conn.videoRotationAngle = angle
            }
        }
    }
}

extension CameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {

    /// 从 Vision 关键点取 5 个对齐点(图像像素坐标,左下原点),顺序:左眼、右眼、鼻、左嘴角、右嘴角。
    /// 眼/嘴角按 x 定左右(不靠 Vision 的 left/right 标签,避开镜像歧义)。
    private func landmarkPoints(_ obs: VNFaceObservation, bufW: CGFloat, bufH: CGFloat) -> [CGPoint]? {
        guard let lm = obs.landmarks else { return nil }
        let sz = CGSize(width: bufW, height: bufH)
        func firstPt(_ r: VNFaceLandmarkRegion2D?) -> CGPoint? { r?.pointsInImage(imageSize: sz).first }
        func centroid(_ r: VNFaceLandmarkRegion2D?) -> CGPoint? {
            guard let p = r?.pointsInImage(imageSize: sz), !p.isEmpty else { return nil }
            let sx = p.reduce(0) { $0 + $1.x }, sy = p.reduce(0) { $0 + $1.y }
            return CGPoint(x: sx / CGFloat(p.count), y: sy / CGFloat(p.count))
        }
        guard let eA = firstPt(lm.leftPupil) ?? centroid(lm.leftEye),
              let eB = firstPt(lm.rightPupil) ?? centroid(lm.rightEye),
              let nose = centroid(lm.nose),
              let lips = lm.outerLips?.pointsInImage(imageSize: sz), !lips.isEmpty,
              let mL = lips.min(by: { $0.x < $1.x }),
              let mR = lips.max(by: { $0.x < $1.x }) else { return nil }
        let (eyeL, eyeR) = eA.x <= eB.x ? (eA, eB) : (eB, eA)
        return [eyeL, eyeR, nose, mL, mR]
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // 节流(检测+编码较重);点了录入时这一帧强制处理
        let now = Date()
        if now.timeIntervalSince(lastProcess) < interval && !captureNext { return }
        lastProcess = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let bufW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try? handler.perform([request])
        let observations = request.results ?? []

        // 逐张:(优先)5点对齐裁剪 /(兜底)框裁剪 → 编码 → 比对
        var built: [(box: CGRect, label: String, recognized: Bool)] = []
        var largest: (vec: [Float], area: CGFloat)? = nil

        for obs in observations {
            let nb = obs.boundingBox
            var label = "模型未加载"
            var recognized = false

            // 只走「5点对齐」这一条路,保证录入/查询向量永远同一空间(跨尺寸一致)。
            // 拿不到关键点(脸太小/太偏)就提示靠近,不做比对、也不可录入。
            if let embedder, let pts = landmarkPoints(obs, bufW: bufW, bufH: bufH),
               let vec = embedder.embedAligned(pixelBuffer: pixelBuffer, src5: pts) {
                let match = engine.findBest(vec.map { NSNumber(value: $0) })
                recognized = (match?.score ?? -2) >= threshold
                if let m = match {
                    label = recognized ? String(format: "%@ %.2f", m.name, m.score)
                                       : String(format: "陌生人 %.2f", m.score)
                } else {
                    label = "库为空"
                }
                let area = nb.width * nb.height
                if captureNext, largest == nil || area > largest!.area {
                    largest = (vec, area)
                }
            } else {
                label = "靠近 · 正对镜头"
            }
            built.append((nb, label, recognized))
        }

        // 处理录入抓取
        if captureNext {
            captureNext = false
            if let lg = largest {
                DispatchQueue.main.async { self.pendingEnroll = lg.vec }
            }
        }

        // 归一化框 → 屏幕坐标(复刻预览 .resizeAspectFill),发布
        DispatchQueue.main.async { [weak self] in
            guard let self, let layer = self.previewLayer else { return }
            let vw = layer.bounds.width, vh = layer.bounds.height
            let scale = max(vw / bufW, vh / bufH)
            let sw = bufW * scale, sh = bufH * scale
            let ox = (vw - sw) / 2, oy = (vh - sh) / 2
            self.faces = built.map { b in
                RecognizedFace(
                    box: CGRect(x: b.box.minX * sw + ox,
                                y: (1 - b.box.maxY) * sh + oy,
                                width: b.box.width * sw,
                                height: b.box.height * sh),
                    label: b.label,
                    recognized: b.recognized)
            }
        }
    }
}
