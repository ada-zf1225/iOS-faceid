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
    @Published var enrolledNames: [String] = []      // 库中姓名(管理面板用)
    @Published var pendingEnroll: [[Float]]? = nil   // 待命名的「一批」模板(多帧录入)
    @Published var livenessOK: Bool = false          // 最近检测到眨眼 = 真人(防照片)
    @Published var hint: String = ""                 // 顶部即时提示(质量门 / 活体)

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
    private var lastProcess = Date.distantPast
    private let interval: TimeInterval = 0.14   // ~7fps 识别(预览是另一条流,不受影响)

    // 多帧录入(multi-shot):一次点击采集 burstTarget 张「合格」人脸,作为多模板录入
    private var burstRemaining = 0
    private var burstVecs: [[Float]] = []
    private let burstTarget = 5

    // 质量门:太远/太偏的脸不录入(归一化框宽下限、姿态弧度上限 ~28°)
    private let minFaceWidth: CGFloat = 0.14
    private let maxPoseRad: Float = 0.5

    // 活体(眨眼):跟最大脸的眼开度,先闭后睁判为一次眨眼
    private var sawClosed = false
    private var noFaceFrames = 0

    // 时序防闪烁:主脸标签滑窗投票
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

    /// 点「录入」:先要求活体(防照片),再连采 burstTarget 张合格脸作多模板
    func requestEnroll() {
        guard livenessOK else {
            DispatchQueue.main.async { self.hint = "请正对镜头并眨一下眼 👁 确认是真人" }
            return
        }
        burstVecs = []
        burstRemaining = burstTarget
        DispatchQueue.main.async { self.hint = "录入中…保持正脸,轻微转动头部" }
    }

    /// 命名确认后真正录入(一批模板都挂到同一姓名下)
    func enroll(name: String) {
        defer { pendingEnroll = nil }
        guard let vecs = pendingEnroll, !name.isEmpty else { return }
        for v in vecs { engine.enrollName(name, embedding: v.map { NSNumber(value: $0) }) }
        refreshEnrolled()
    }

    func cancelEnroll() { pendingEnroll = nil }

    /// 删除 / 改名某人(管理面板用)
    func deletePerson(_ name: String) { engine.removeName(name); refreshEnrolled() }
    func renamePerson(_ old: String, to newName: String) {
        let t = newName.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty, t != old else { return }
        engine.rename(from: old, to: t); refreshEnrolled()
    }

    /// 清空整个录入库
    func resetDB() { engine.clear(); refreshEnrolled() }

    /// 某人有几条模板(管理面板显示用)
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

    /// 录入质量门:脸够大(不太远)+ 姿态够正(yaw/roll 在阈内)。识别不受此限制。
    private func qualityOK(_ obs: VNFaceObservation) -> Bool {
        if obs.boundingBox.width < minFaceWidth { return false }
        if let y = obs.yaw?.floatValue, abs(y) > maxPoseRad { return false }
        if let r = obs.roll?.floatValue, abs(r) > maxPoseRad { return false }
        return true
    }

    /// 眼开度(眼轮廓 高/宽 比);睁眼≈0.25+,闭眼≈<0.15。取双眼均值。
    private func eyeOpenness(_ obs: VNFaceObservation) -> Float? {
        guard let lm = obs.landmarks else { return nil }
        func ratio(_ r: VNFaceLandmarkRegion2D?) -> Float? {
            guard let p = r?.normalizedPoints, p.count >= 4 else { return nil }
            let xs = p.map { $0.x }, ys = p.map { $0.y }
            let w = (xs.max()! - xs.min()!), h = (ys.max()! - ys.min()!)
            return w > 1e-5 ? Float(h / w) : nil
        }
        let vals = [ratio(lm.leftEye), ratio(lm.rightEye)].compactMap { $0 }
        return vals.isEmpty ? nil : vals.reduce(0, +) / Float(vals.count)
    }

    /// 用最大脸的眼开度更新活体状态(先闭后睁=一次眨眼);久无脸则失效。
    private func updateLiveness(primary: VNFaceObservation?) {
        if let p = primary, let o = eyeOpenness(p) {
            noFaceFrames = 0
            if o < 0.16 { sawClosed = true }
            if sawClosed && o > 0.24 { sawClosed = false; setLive(true) }
        } else {
            noFaceFrames += 1
            if noFaceFrames > 20 { sawClosed = false; setLive(false) }
        }
    }
    private func setLive(_ v: Bool) {
        DispatchQueue.main.async { if self.livenessOK != v { self.livenessOK = v } }
    }

    /// 主脸标签滑窗投票:返回出现最多的姓名(仅统计已认出的帧),减少抖动。
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
        // 节流(检测+编码较重)
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

        // 逐张:5点对齐裁剪 → 编码 → 比对;记录每张脸的姓名/合格录入向量,并挑「最大脸」
        var built: [(box: CGRect, label: String, recognized: Bool)] = []
        var names: [String?] = []        // 每脸认出的姓名(nil=未认出)
        var vecs: [[Float]?] = []        // 每脸的「合格」录入向量(nil=不合格/无关键点)
        var maxIdx = -1; var maxArea: CGFloat = -1

        for (i, obs) in observations.enumerated() {
            let nb = obs.boundingBox
            let area = nb.width * nb.height
            if area > maxArea { maxArea = area; maxIdx = i }

            var label = "模型未加载"
            var recognized = false
            var nm: String? = nil
            var qvec: [Float]? = nil

            // 只走「5点对齐」这一条路,保证录入/查询向量永远同一空间(跨尺寸一致)。
            if let embedder, let pts = landmarkPoints(obs, bufW: bufW, bufH: bufH),
               let vec = embedder.embedAligned(pixelBuffer: pixelBuffer, src5: pts) {
                let match = engine.findBest(vec.map { NSNumber(value: $0) })
                recognized = (match?.score ?? -2) >= threshold
                if let m = match {
                    label = recognized ? String(format: "%@ %.2f", m.name, m.score)
                                       : String(format: "陌生人 %.2f", m.score)
                    if recognized { nm = m.name }
                } else {
                    label = "库为空"
                }
                if qualityOK(obs) { qvec = vec }   // 录入只收合格(够大够正)的脸
            } else {
                label = "靠近 · 正对镜头"
            }
            built.append((nb, label, recognized)); names.append(nm); vecs.append(qvec)
        }

        // 活体:用最大脸的眨眼更新
        updateLiveness(primary: maxIdx >= 0 ? observations[maxIdx] : nil)

        // 多帧录入:每帧收主脸的合格向量,够 burstTarget 张就交界面命名
        if burstRemaining > 0, maxIdx >= 0, let v = vecs[maxIdx] {
            burstVecs.append(v)
            burstRemaining -= 1
            if burstRemaining == 0 {
                let batch = burstVecs
                DispatchQueue.main.async { self.pendingEnroll = batch; self.hint = "" }
            }
        }

        // 时序防闪烁:主脸姓名滑窗投票,稳定后覆盖显示
        if maxIdx >= 0, let sm = smoothedName(names[maxIdx]) {
            built[maxIdx].label = sm + " ✓"
            built[maxIdx].recognized = true
        } else if maxIdx < 0 {
            _ = smoothedName(nil)
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
