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
/// 流程:相机帧(已随朝向转正+镜像)→ Vision 检测/关键点 → 5点对齐 → ArcFace 编码
///       → C++ 引擎余弦比对 → 标签(姓名 / 陌生人)。识别管线节流到 ~7fps。
final class CameraModel: NSObject, ObservableObject {

    @Published var faces: [RecognizedFace] = []
    @Published var enrolledCount: Int = 0
    @Published var enrolledNames: [String] = []
    @Published var pendingEnroll: [[Float]]? = nil   // 待命名的「一批」模板(多帧录入)
    @Published var hint: String = ""
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
    private var lastProcess = Date.distantPast
    private let interval: TimeInterval = 0.14

    // 多帧录入(multi-shot):一次点击采 burstTarget 张合格脸 → 多模板
    private var burstRemaining = 0
    private var burstVecs: [[Float]] = []
    private let burstTarget = 5

    // 录入质量门(太远/太偏不收)
    private let minFaceWidth: CGFloat = 0.14
    private let maxPoseRad: Float = 0.5

    // 时序防闪烁
    private var labelHistory: [String] = []

    override init() {
        let dir = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first
            ?? NSTemporaryDirectory()
        let dbPath = (dir as NSString).appendingPathComponent("faces.db")
        engine = FaceEngineBridge(dbPath: dbPath)
        super.init()
        enrolledNames = engine.names()
        enrolledCount = enrolledNames.count
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

    /// 点「录入」:连采 burstTarget 张合格脸作多模板
    func requestEnroll() {
        burstVecs = []
        burstRemaining = burstTarget
        DispatchQueue.main.async { self.hint = "录入中…保持正脸,轻微转动头部" }
    }

    func enroll(name: String) {
        defer { pendingEnroll = nil }
        guard let vecs = pendingEnroll, !name.isEmpty else { return }
        for v in vecs { engine.enrollName(name, embedding: v.map { NSNumber(value: $0) }) }
        refreshEnrolled()
    }

    func cancelEnroll() { pendingEnroll = nil }

    func deletePerson(_ name: String) { engine.removeName(name); refreshEnrolled() }
    func renamePerson(_ old: String, to newName: String) {
        let t = newName.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty, t != old else { return }
        engine.rename(from: old, to: t); refreshEnrolled()
    }

    func resetDB() { engine.clear(); refreshEnrolled() }

    func templateCount(_ name: String) -> Int { Int(engine.templateCount(of: name)) }

    private func refreshEnrolled() {
        let ns = engine.names()
        DispatchQueue.main.async { self.enrolledNames = ns; self.enrolledCount = ns.count }
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
            conn.isVideoMirrored = true   // 前置镜像,和自拍预览一致
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

    /// 从 Vision 关键点取 5 个对齐点(图像像素坐标,左下原点):左眼、右眼、鼻、左嘴角、右嘴角。
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

    /// 录入质量门:脸够大 + 姿态够正。识别不受此限制。
    private func qualityOK(_ obs: VNFaceObservation) -> Bool {
        if obs.boundingBox.width < minFaceWidth { return false }
        if let y = obs.yaw?.floatValue, abs(y) > maxPoseRad { return false }
        if let r = obs.roll?.floatValue, abs(r) > maxPoseRad { return false }
        return true
    }

    /// 主脸标签滑窗投票:返回出现最多的姓名(仅统计认出的帧),减少抖动。
    private func smoothedName(_ current: String?) -> String? {
        labelHistory.append(current ?? "")
        if labelHistory.count > 7 { labelHistory.removeFirst() }
        var tally: [String: Int] = [:]
        for s in labelHistory where !s.isEmpty { tally[s, default: 0] += 1 }
        guard let (name, n) = tally.max(by: { $0.value < $1.value }), n >= 3 else { return nil }
        return name
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let now = Date()
        if now.timeIntervalSince(lastProcess) < interval { return }
        lastProcess = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let bufW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try? handler.perform([request])
        let observations = request.results ?? []

        var built: [(box: CGRect, label: String, recognized: Bool)] = []
        var names: [String?] = []
        var vecs: [[Float]?] = []
        var maxIdx = -1; var maxArea: CGFloat = -1

        for (i, obs) in observations.enumerated() {
            let nb = obs.boundingBox
            let area = nb.width * nb.height
            if area > maxArea { maxArea = area; maxIdx = i }

            var label = "靠近 · 正对镜头"
            var recognized = false
            var nm: String? = nil
            var qvec: [Float]? = nil

            // 只走「5点对齐」,保证录入/查询向量同一空间(跨尺寸一致)。
            if let embedder, let pts = landmarkPoints(obs, bufW: bufW, bufH: bufH),
               let vec = embedder.embedAligned(pixelBuffer: pixelBuffer, src5: pts) {
                let match = engine.findBest(vec.map { NSNumber(value: $0) })
                let hit = (match?.score ?? -2) >= threshold
                if let m = match {
                    label = hit ? String(format: "%@ %.2f", m.name, m.score)
                                : String(format: "陌生人 %.2f", m.score)
                    recognized = hit
                    if hit { nm = m.name }
                } else {
                    label = "库为空"
                }
                if qualityOK(obs) { qvec = vec }
            }
            built.append((nb, label, recognized)); names.append(nm); vecs.append(qvec)
        }

        // 多帧录入:每帧收主脸合格向量,够 burstTarget 张交界面命名
        if burstRemaining > 0, maxIdx >= 0, let v = vecs[maxIdx] {
            burstVecs.append(v)
            burstRemaining -= 1
            if burstRemaining == 0 {
                let batch = burstVecs
                DispatchQueue.main.async { self.pendingEnroll = batch; self.hint = "" }
            }
        }

        // 时序防闪烁(主脸)
        if maxIdx >= 0, let sm = smoothedName(names[maxIdx]) {
            built[maxIdx].label = sm + " ✓"
            built[maxIdx].recognized = true
        } else if maxIdx < 0 {
            _ = smoothedName(nil)
        }

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
