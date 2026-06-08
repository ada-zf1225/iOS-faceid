import AVFoundation
import Vision
import SwiftUI
import UIKit
import CoreImage

/// 轻量触感反馈
enum Haptics {
    static func success() { DispatchQueue.main.async { UINotificationFeedbackGenerator().notificationOccurred(.success) } }
}

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
    @Published var pendingThumb: UIImage? = nil      // 随 pendingEnroll 的人脸缩略图
    @Published var hint: String = ""
    @Published var threshold: Float = 0.35

    private var lastBurstThumb: UIImage? = nil
    private var enrollTargetName: String? = nil      // 非 nil = 补录到已有人,跳过命名
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private lazy var thumbsDir: URL = {
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("thumbs")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()

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

    // 多脸跟踪 + 逐脸时序防闪烁(IoU 关联,每条轨迹各自投票)
    private struct Track { var box: CGRect; var history: [String]; var missed: Int }
    private var tracks: [Int: Track] = [:]
    private var nextTrackId = 0

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

    /// 点「录入」:连采 burstTarget 张合格脸作多模板。forName 非空 = 给已有人补录(跳过命名)。
    func requestEnroll(forName name: String? = nil) {
        enrollTargetName = name
        burstVecs = []; lastBurstThumb = nil
        burstRemaining = burstTarget
        let p = name == nil ? "录入中… " : "补录 \(name!)… "
        DispatchQueue.main.async { self.hint = p + "0/\(self.burstTarget)" }
    }

    func enroll(name: String) {
        defer { pendingEnroll = nil; pendingThumb = nil }
        guard let vecs = pendingEnroll, !name.isEmpty else { return }
        for v in vecs { engine.enrollName(name, embedding: v.map { NSNumber(value: $0) }) }
        if let t = pendingThumb { saveThumb(t, name: name) }
        refreshEnrolled(); Haptics.success()
    }

    func cancelEnroll() { pendingEnroll = nil; pendingThumb = nil }

    func deletePerson(_ name: String) {
        engine.removeName(name)
        try? FileManager.default.removeItem(at: thumbURL(name))
        refreshEnrolled()
    }
    func renamePerson(_ old: String, to newName: String) {
        let t = newName.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty, t != old else { return }
        engine.rename(from: old, to: t)
        try? FileManager.default.moveItem(at: thumbURL(old), to: thumbURL(t))
        refreshEnrolled()
    }

    func resetDB() {
        engine.clear()
        try? FileManager.default.removeItem(at: thumbsDir)
        try? FileManager.default.createDirectory(at: thumbsDir, withIntermediateDirectories: true)
        refreshEnrolled()
    }

    func templateCount(_ name: String) -> Int { Int(engine.templateCount(of: name)) }

    func thumbnail(for name: String) -> UIImage? { UIImage(contentsOfFile: thumbURL(name).path) }

    private func thumbURL(_ name: String) -> URL {
        let safe = name.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? "x"
        return thumbsDir.appendingPathComponent(safe + ".jpg")
    }
    private func saveThumb(_ img: UIImage, name: String) {
        if let d = img.jpegData(compressionQuality: 0.8) { try? d.write(to: thumbURL(name)) }
    }
    /// 从相机帧按人脸框裁一张 ~140px 缩略图(随预览镜像,自然)。
    private func makeThumb(_ pb: CVPixelBuffer, box: CGRect) -> UIImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        let W = ci.extent.width, H = ci.extent.height
        let mx = box.width * 0.15 * W, my = box.height * 0.15 * H
        let rect = CGRect(x: box.minX*W - mx, y: box.minY*H - my,
                          width: box.width*W + 2*mx, height: box.height*H + 2*my).intersection(ci.extent)
        guard !rect.isNull, rect.width > 8, rect.height > 8 else { return nil }
        let cropped = ci.cropped(to: rect)
        let s = 140.0 / max(cropped.extent.width, cropped.extent.height)
        let scaled = cropped.transformed(by: CGAffineTransform(scaleX: s, y: s))
        guard let cg = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cg)
    }

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

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull { return 0 }
        let ia = inter.width * inter.height
        return ia / (a.width*a.height + b.width*b.height - ia)
    }

    /// 把本帧每张脸按 IoU 关联到已有轨迹(无则新建),各轨迹独立做姓名滑窗投票。
    /// 返回每张脸的稳定姓名(出现≥3 次才认,否则 nil)。
    private func trackAndSmooth(boxes: [CGRect], names: [String?]) -> [String?] {
        var result = [String?](repeating: nil, count: boxes.count)
        var used = Set<Int>()
        var assigned = [Int](repeating: -1, count: boxes.count)
        for i in 0..<boxes.count {
            var bestId = -1; var best: CGFloat = 0.3
            for (id, tr) in tracks where !used.contains(id) {
                let o = iou(boxes[i], tr.box)
                if o > best { best = o; bestId = id }
            }
            if bestId < 0 { bestId = nextTrackId; nextTrackId += 1; tracks[bestId] = Track(box: boxes[i], history: [], missed: 0) }
            used.insert(bestId); assigned[i] = bestId
        }
        for i in 0..<boxes.count {
            let id = assigned[i]
            var tr = tracks[id]!
            tr.box = boxes[i]; tr.missed = 0
            tr.history.append(names[i] ?? "")
            if tr.history.count > 7 { tr.history.removeFirst() }
            tracks[id] = tr
            var tally: [String: Int] = [:]
            for s in tr.history where !s.isEmpty { tally[s, default: 0] += 1 }
            if let (nm, c) = tally.max(by: { $0.value < $1.value }), c >= 3 { result[i] = nm }
        }
        for id in Array(tracks.keys) where !used.contains(id) {
            tracks[id]!.missed += 1
            if tracks[id]!.missed > 8 { tracks.removeValue(forKey: id) }
        }
        return result
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

        // 多帧录入:每帧收主脸合格向量 + 缩略图,显示进度;够 burstTarget 张后命名/补录
        if burstRemaining > 0, maxIdx >= 0, let v = vecs[maxIdx] {
            burstVecs.append(v)
            lastBurstThumb = makeThumb(pixelBuffer, box: built[maxIdx].box) ?? lastBurstThumb
            burstRemaining -= 1
            let got = burstTarget - burstRemaining
            let tn = enrollTargetName
            DispatchQueue.main.async { self.hint = (tn == nil ? "录入中… " : "补录中… ") + "\(got)/\(self.burstTarget)" }
            if burstRemaining == 0 {
                let batch = burstVecs, thumb = lastBurstThumb
                if let tn = enrollTargetName {                 // 补录:直接入库,不弹命名
                    enrollTargetName = nil
                    for vv in batch { engine.enrollName(tn, embedding: vv.map { NSNumber(value: $0) }) }
                    if let thumb { saveThumb(thumb, name: tn) }
                    refreshEnrolled(); Haptics.success()
                    DispatchQueue.main.async { self.hint = "已补录 \(batch.count) 张到 \(tn)" }
                } else {                                        // 新人:交界面命名
                    DispatchQueue.main.async { self.pendingEnroll = batch; self.pendingThumb = thumb; self.hint = "" }
                }
            }
        }

        // 多脸跟踪 + 逐脸时序投票(每张脸都稳,不只主脸)
        let smoothed = trackAndSmooth(boxes: built.map { $0.box }, names: names)
        for i in 0..<built.count where smoothed[i] != nil {
            built[i].label = smoothed[i]! + " ✓"
            built[i].recognized = true
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
