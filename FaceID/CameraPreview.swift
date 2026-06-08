import SwiftUI
import AVFoundation

/// 一个 UIView,其底层 layer 就是相机预览层(AVCaptureVideoPreviewLayer)。
final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

/// 把上面的 UIView 包成 SwiftUI 视图。SwiftUI 没有原生相机预览,
/// 用 UIViewRepresentable 桥接 UIKit(类似安卓里用 AndroidView 嵌 PreviewView)。
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// 把创建好的 previewLayer 回传出去,模型用它做「检测框 → 屏幕坐标」换算。
    let onLayerReady: (AVCaptureVideoPreviewLayer) -> Void

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill   // 等比铺满
        onLayerReady(view.previewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}
