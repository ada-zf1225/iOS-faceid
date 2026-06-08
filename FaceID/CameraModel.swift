import AVFoundation
import Vision
import SwiftUI
import UIKit

/// 一张被识别的人脸:框(屏幕坐标)+ 标签 + 是否认出 + 活体判定
struct RecognizedFace: Identifiable {
    let id = UUID()
    let box: CGRect
    let label: String
    let recognized: Bool
    let live: Bool?        // nil=未知(无深度/样本不足);true=真人;false=平面假体
}

/// 相机 + 检测 + 识别 + 活体核心。
/// 流程:TrueDepth 相机帧(视频+深度同步)→ Vision 检测/关键点 → 5点对齐 → ArcFace 编码
///       → C++ 引擎余弦比对 → 标签;同时按人脸区域采样深度图,平面拟合判真脸/平面(防照片+视频回放)。
final class CameraModel: NSObject, ObservableObject {

    @Published var faces: [RecognizedFace] = []
    @Published var enrolledCount: Int = 0
    @Published var enrolledNames: [String] = []
    @Published var pendingEnroll: [[Float]]? = nil
    @Published var livenessOK: Bool = false          // 眨眼活体(录入门控用)
    @Published var hint: String = ""
    @Published var threshold: Float = 0.35
    @Published var gateMode: Bool = false            // false=只显示活体/假体;true=门控(假体不给身份)
    @Published var depthAvailable: Bool = false      // 设备是否有 TrueDepth
    @Published var depthDebug: String = ""           // 主脸深度指标(真机标定用)

    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let sessionQueue = DispatchQueue(label: "faceid.camera.session")
    private let frameQueue = DispatchQueue(label: "faceid.camera.frames")

    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var device: AVCaptureDevice?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?

    private let embedder = FaceEmbedder()
    private let pad = PADClassifier()                 // 反欺骗 CNN(模型缺失=nil,自动降级为纯深度)
    private let engine: FaceEngineBridge
    private var lastProcess = Date.distantPast
    private let interval: TimeInterval = 0.14

    // 多帧录入
    private var burstRemaining = 0
    private var burstVecs: [[Float]] = []
    private let burstTarget = 5

    // 录入质量门
    private let minFaceWidth: CGFloat = 0.14
    private let maxPoseRad: Float = 0.5

    // 眨眼活体
    private var sawClosed = false
    private var noFaceFrames = 0

    // 时序防闪烁
    private var labelHistory: [String] = []

    // ── 深度活体可调参数(真机标定)──
    // 真脸有 3D 起伏 → 平面拟合残差大;照片/屏幕(哪怕倾斜)是平面 → 残差小。
    private let depthResidualThresh: Float = 0.008   // 米;> 此值判真人。太严→真人被拒,太松→照片漏过
    private let depthMinSamples = 12
    // 视频前置镜像、深度图通常不镜像 → 采样时水平翻转。若真机发现活体判反了,把它改成 false。
    private let depthFlipH = true

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

    func requestEnroll() {
        guard livenessOK else {
            DispatchQueue.main.async { self.hint = "请正对镜头并眨一下眼 👁 确认是真人" }
            return
        }
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

        // 优先 TrueDepth(前置带深度);退回普通前置 / 任意
        let cam = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video)
        if let cam, let input = try? AVCaptureDeviceInput(device: cam), session.canAddInput(input) {
            session.addInput(input)
            device = cam
        }

        let isTrueDepth = (device?.deviceType == .builtInTrueDepthCamera)
        if isTrueDepth, let cam = device {
            // 选「支持深度 + 视频分辨率最高」的格式
            let depthFmts = cam.formats.filter { !$0.supportedDepthDataFormats.isEmpty }
            if let best = depthFmts.max(by: {
                CMVideoFormatDescriptionGetDimensions($0.formatDescription).width <
                CMVideoFormatDescriptionGetDimensions($1.formatDescription).width
            }), (try? cam.lockForConfiguration()) != nil {
                cam.activeFormat = best
                let f32 = best.supportedDepthDataFormats.first {
                    CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
                }
                cam.activeDepthDataFormat = f32 ?? best.supportedDepthDataFormats.first
                cam.unlockForConfiguration()
            }
        } else {
            session.sessionPreset = .high
        }

        videoOutput.alwaysDiscardsLateVideoFrames = true
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        if let conn = videoOutput.connection(with: .video), conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = true
        }

        if isTrueDepth, session.canAddOutput(depthOutput) {
            session.addOutput(depthOutput)
            depthOutput.isFilteringEnabled = true
            depthOutput.connection(with: .depthData)?.isEnabled = true
            depthAvailable = true
        }

        session.commitConfiguration()

        // 有深度走「视频+深度同步」;否则退回纯视频回调
        if depthAvailable {
            let sync = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
            sync.setDelegate(self, queue: frameQueue)
            synchronizer = sync
        } else {
            videoOutput.setSampleBufferDelegate(self, queue: frameQueue)
        }

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
            // 深度连接同角度旋转,保证与视频帧坐标一致(只差一个水平镜像)
            if let conn = self.depthOutput.connection(with: .depthData), conn.isVideoRotationAngleSupported(angle) {
                conn.videoRotationAngle = angle
            }
        }
    }
}

// MARK: - 深度活体

extension CameraModel {
    /// 解 3x3 线性方程(平面拟合用),Cramer 法则。无解返回 nil。
    private func solve3x3(_ m: [[Float]], _ r: [Float]) -> (Float, Float, Float)? {
        func det3(_ a: [[Float]]) -> Float {
            a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
          - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
          + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
        }
        let D = det3(m)
        guard abs(D) > 1e-12 else { return nil }
        func sub(_ col: Int) -> [[Float]] {
            var c = m; for i in 0..<3 { c[i][col] = r[i] }; return c
        }
        return (det3(sub(0))/D, det3(sub(1))/D, det3(sub(2))/D)
    }

    /// 在深度图上按人脸框中心区域采样,平面拟合 → 残差。残差大=3D 真人,小=平面假体。
    /// 返回 (isLive, range米, residual米);样本不足返回 nil。
    private func depthLiveness(_ depthMap: CVPixelBuffer, faceBox nb: CGRect) -> (Bool, Float, Float)? {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        let w = CVPixelBufferGetWidth(depthMap), h = CVPixelBufferGetHeight(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)

        func depthAt(_ col: Int, _ row: Int) -> Float? {
            guard col >= 0, col < w, row >= 0, row < h else { return nil }
            let p = base.advanced(by: row * rowBytes + col * MemoryLayout<Float32>.size)
            let d = p.assumingMemoryBound(to: Float32.self).pointee
            guard d.isFinite, d > 0.05, d < 5.0 else { return nil }   // 合理深度(米)
            return d
        }

        // Vision 框:归一化、左下原点、镜像视频空间。深度:同角度旋转、不镜像 → 翻 x;像素行从上 → 翻 y。
        let cx = nb.midX, cy = nb.midY
        let rw = nb.width * 0.6, rh = nb.height * 0.6
        var us: [Float] = [], vs: [Float] = [], ds: [Float] = []
        let N = 9
        for i in 0..<N {
            for j in 0..<N {
                let fx = cx - rw/2 + rw * CGFloat(i)/CGFloat(N-1)
                let fy = cy - rh/2 + rh * CGFloat(j)/CGFloat(N-1)
                let nx = depthFlipH ? (1 - fx) : fx
                let col = Int(nx * CGFloat(w))
                let row = Int((1 - fy) * CGFloat(h))
                if let d = depthAt(col, row) {
                    us.append(Float(fx)); vs.append(Float(fy)); ds.append(d)
                }
            }
        }
        guard ds.count >= depthMinSamples else { return nil }

        // 最小二乘平面 d ≈ a·u + b·v + c
        var Suu: Float = 0, Suv: Float = 0, Su: Float = 0, Svv: Float = 0, Sv: Float = 0
        var Sud: Float = 0, Svd: Float = 0, Sd: Float = 0
        let n = Float(ds.count)
        for k in 0..<ds.count {
            let u = us[k], v = vs[k], d = ds[k]
            Suu += u*u; Suv += u*v; Su += u; Svv += v*v; Sv += v
            Sud += u*d; Svd += v*d; Sd += d
        }
        let M = [[Suu, Suv, Su], [Suv, Svv, Sv], [Su, Sv, n]]
        let R = [Sud, Svd, Sd]
        guard let (a, b, c) = solve3x3(M, R) else { return nil }
        var sse: Float = 0
        for k in 0..<ds.count { let e = ds[k] - (a*us[k] + b*vs[k] + c); sse += e*e }
        let residual = (sse / n).squareRoot()
        let range = (ds.max() ?? 0) - (ds.min() ?? 0)
        return (residual > depthResidualThresh, range, residual)
    }
}

// MARK: - 公共处理(视频 ± 深度都汇到这里)

extension CameraModel {
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

    private func qualityOK(_ obs: VNFaceObservation) -> Bool {
        if obs.boundingBox.width < minFaceWidth { return false }
        if let y = obs.yaw?.floatValue, abs(y) > maxPoseRad { return false }
        if let r = obs.roll?.floatValue, abs(r) > maxPoseRad { return false }
        return true
    }

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

    private func smoothedName(_ current: String?) -> String? {
        labelHistory.append(current ?? "")
        if labelHistory.count > 7 { labelHistory.removeFirst() }
        var tally: [String: Int] = [:]
        for s in labelHistory where !s.isEmpty { tally[s, default: 0] += 1 }
        guard let (name, n) = tally.max(by: { $0.value < $1.value }), n >= 3 else { return nil }
        return name
    }

    /// 视频(可选深度)→ 检测 → 深度活体 + 识别 → 主脸 CNN 反欺骗融合 → 发布
    fileprivate func process(sampleBuffer: CMSampleBuffer, depthMap: CVPixelBuffer?) {
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

        // 第一遍:每张脸的框 / 深度活体 / 识别结果 / 录入向量
        var boxes: [CGRect] = []
        var live: [Bool?] = []
        var depthInfo: [(Float, Float)?] = []
        var mName: [String?] = []
        var mScore: [Float?] = []
        var hasEmbed: [Bool] = []
        var qvecs: [[Float]?] = []
        var maxIdx = -1; var maxArea: CGFloat = -1

        for (i, obs) in observations.enumerated() {
            let nb = obs.boundingBox
            let area = nb.width * nb.height
            if area > maxArea { maxArea = area; maxIdx = i }

            var dl: Bool? = nil; var di: (Float, Float)? = nil
            if let dm = depthMap, let (isLive, rng, res) = depthLiveness(dm, faceBox: nb) {
                dl = isLive; di = (rng, res)
            }
            var nm: String? = nil; var ms: Float? = nil; var emb = false; var qv: [Float]? = nil
            if let embedder, let pts = landmarkPoints(obs, bufW: bufW, bufH: bufH),
               let vec = embedder.embedAligned(pixelBuffer: pixelBuffer, src5: pts) {
                emb = true
                if let m = engine.findBest(vec.map { NSNumber(value: $0) }) { nm = m.name; ms = m.score }
                if qualityOK(obs) { qv = vec }
            }
            boxes.append(nb); live.append(dl); depthInfo.append(di)
            mName.append(nm); mScore.append(ms); hasEmbed.append(emb); qvecs.append(qv)
        }

        // CNN 反欺骗:只跑主脸(开销大),与深度做「与」融合(都说活体才算活体)
        var dbg = ""
        if maxIdx >= 0 {
            if let (rng, res) = depthInfo[maxIdx] {
                dbg = String(format: "深度 res=%.3f range=%.3f", res, rng)
            }
            if let pad, let sp = pad.spoofProbability(pixelBuffer: pixelBuffer, normalizedBox: boxes[maxIdx]) {
                let cnnLive = sp < 0.5
                let dl = live[maxIdx]
                live[maxIdx] = (dl == nil) ? cnnLive : (dl! && cnnLive)
                dbg += String(format: "  CNN spoof=%.2f", sp)
            }
        }

        updateLiveness(primary: maxIdx >= 0 ? observations[maxIdx] : nil)

        if burstRemaining > 0, maxIdx >= 0, let v = qvecs[maxIdx] {
            burstVecs.append(v)
            burstRemaining -= 1
            if burstRemaining == 0 {
                let batch = burstVecs
                DispatchQueue.main.async { self.pendingEnroll = batch; self.hint = "" }
            }
        }

        // 第二遍:用融合后的活体合成标签 / 门控
        var built: [(box: CGRect, label: String, recognized: Bool, live: Bool?)] = []
        var primaryName: String? = nil
        for i in 0..<observations.count {
            let lv = live[i]
            let mark = lv == true ? " ·活体" : (lv == false ? " ·假体⚠" : "")
            var label = "靠近 · 正对镜头"
            var recognized = false
            if !hasEmbed[i] {
                label = "靠近 · 正对镜头"
            } else if let ms = mScore[i], let nm = mName[i] {
                if ms >= threshold {
                    if gateMode && lv == false { label = "假体 ⚠"; recognized = false }
                    else {
                        label = String(format: "%@ %.2f%@", nm, ms, mark)
                        recognized = true
                        if i == maxIdx { primaryName = nm }
                    }
                } else {
                    label = String(format: "陌生人 %.2f%@", ms, mark)
                }
            } else {
                label = "库为空"
            }
            built.append((boxes[i], label, recognized, lv))
        }

        // 时序防闪烁(主脸);门控下假体不覆盖姓名
        var sm: String? = nil
        if maxIdx >= 0 { sm = smoothedName(primaryName) } else { _ = smoothedName(nil) }
        if maxIdx >= 0, let sm, !(gateMode && built[maxIdx].live == false) {
            built[maxIdx].label = sm + (built[maxIdx].live == true ? " ✓活体" : " ✓")
            built[maxIdx].recognized = true
        }

        DispatchQueue.main.async { [weak self] in
            guard let self, let layer = self.previewLayer else { return }
            self.depthDebug = dbg
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
                    recognized: b.recognized,
                    live: b.live)
            }
        }
    }
}

// 纯视频回调(无深度设备的兜底)
extension CameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        process(sampleBuffer: sampleBuffer, depthMap: nil)
    }
}

// 视频+深度同步回调(TrueDepth)
extension CameraModel: AVCaptureDataOutputSynchronizerDelegate {
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let v = collection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
              !v.sampleBufferWasDropped else { return }
        var depthMap: CVPixelBuffer? = nil
        if let d = collection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData,
           !d.depthDataWasDropped {
            let conv = d.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
            depthMap = conv.depthDataMap
        }
        process(sampleBuffer: v.sampleBuffer, depthMap: depthMap)
    }
}
