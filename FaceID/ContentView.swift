import SwiftUI

struct ContentView: View {
    @StateObject private var model = CameraModel()
    @State private var enrollName = ""

    var body: some View {
        ZStack {
            // 1) 相机预览
            CameraPreview(session: model.session) { layer in
                model.setPreviewLayer(layer)
            }
            .ignoresSafeArea()

            // 2) 人脸框 + 姓名标签(认出=绿,陌生人=红)
            ForEach(model.faces) { face in
                let color: Color = face.recognized ? .green : .red
                Rectangle()
                    .stroke(color, lineWidth: 3)
                    .frame(width: face.box.width, height: face.box.height)
                    .position(x: face.box.midX, y: face.box.midY)
                Text(face.label)
                    .font(.caption).bold()
                    .foregroundColor(.white)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(color)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .fixedSize()
                    .position(x: face.box.midX, y: max(face.box.minY - 14, 20))
            }
            .ignoresSafeArea()

            // 3) 顶部计数 + 底部录入按钮
            VStack {
                HStack(spacing: 12) {
                    Text("库中 \(model.enrolledCount) 人 · 当前 \(model.faces.count) 张脸")
                        .font(.headline)
                        .foregroundColor(.white)
                    Button(role: .destructive) { model.resetDB() } label: {
                        Image(systemName: "trash")
                    }
                    .tint(.red)
                }
                .padding(8)
                .background(.black.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.top, 60)

                Spacer()

                // 阈值滑块(ArcFace:同人~0.4-0.7,陌生人≈0.1)
                HStack {
                    Text("阈值 \(String(format: "%.2f", model.threshold))")
                        .font(.caption).foregroundColor(.white)
                        .frame(width: 72, alignment: .leading)
                    Slider(value: $model.threshold, in: 0.2...0.7)
                }
                .padding(.horizontal, 16).padding(.vertical, 8)
                .background(.black.opacity(0.45))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 40)
                .padding(.bottom, 12)

                Button {
                    model.requestEnroll()
                } label: {
                    Label("录入当前人脸", systemImage: "person.crop.circle.badge.plus")
                        .font(.headline)
                        .padding(.horizontal, 20).padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .padding(.bottom, 40)
            }
        }
        .onAppear { model.start() }
        .alert("录入人脸", isPresented: Binding(
            get: { model.pendingEnroll != nil },
            set: { if !$0 { model.cancelEnroll() } }
        )) {
            TextField("名字", text: $enrollName)
            Button("保存") { model.enroll(name: enrollName); enrollName = "" }
            Button("取消", role: .cancel) { model.cancelEnroll(); enrollName = "" }
        } message: {
            Text("给当前这张脸起个名字")
        }
    }
}

#Preview {
    ContentView()
}
