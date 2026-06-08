import SwiftUI

struct ContentView: View {
    @StateObject private var model = CameraModel()
    @State private var enrollName = ""
    @State private var showManage = false

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

            // 3) 顶部状态条 + 底部录入
            VStack {
                HStack(spacing: 10) {
                    Text("库中 \(model.enrolledCount) 人 · \(model.faces.count) 张脸")
                        .font(.subheadline).bold().foregroundColor(.white)
                    Spacer()
                    Button { showManage = true } label: {
                        Image(systemName: "person.2.crop.square.stack")
                    }.tint(.white)
                    Button(role: .destructive) { model.resetDB() } label: {
                        Image(systemName: "trash")
                    }.tint(.red)
                }
                .padding(10)
                .background(.black.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding(.top, 60).padding(.horizontal, 12)

                if !model.hint.isEmpty {
                    Text(model.hint)
                        .font(.caption).bold().foregroundColor(.white)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.black.opacity(0.55))
                        .clipShape(Capsule())
                        .padding(.top, 6)
                }

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
                    Label("录入当前人脸(多帧)", systemImage: "person.crop.circle.badge.plus")
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
            Text("已采集 \(model.pendingEnroll?.count ?? 0) 帧,作为多模板挂到同一姓名下")
        }
        .sheet(isPresented: $showManage) {
            ManagePeopleView(model: model)
        }
    }
}

/// 已录入人物管理:列出姓名 + 模板数,支持改名 / 滑动删除。
struct ManagePeopleView: View {
    @ObservedObject var model: CameraModel
    @Environment(\.dismiss) private var dismiss
    @State private var renaming: String? = nil
    @State private var renameText = ""

    var body: some View {
        NavigationStack {
            List {
                if model.enrolledNames.isEmpty {
                    Text("库为空,先去录入").foregroundStyle(.secondary)
                }
                ForEach(model.enrolledNames, id: \.self) { name in
                    HStack {
                        Image(systemName: "person.crop.circle").foregroundStyle(.tint)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(name).font(.body)
                            Text("\(model.templateCount(name)) 个模板")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button {
                            renameText = name; renaming = name
                        } label: { Image(systemName: "pencil") }
                        .buttonStyle(.borderless)
                    }
                    .swipeActions {
                        Button(role: .destructive) { model.deletePerson(name) } label: {
                            Label("删除", systemImage: "trash")
                        }
                    }
                }
            }
            .navigationTitle("已录入的人")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("完成") { dismiss() } }
            }
            .alert("改名", isPresented: Binding(
                get: { renaming != nil }, set: { if !$0 { renaming = nil } }
            )) {
                TextField("新名字", text: $renameText)
                Button("保存") {
                    if let old = renaming { model.renamePerson(old, to: renameText) }
                    renaming = nil
                }
                Button("取消", role: .cancel) { renaming = nil }
            }
        }
    }
}

#Preview {
    ContentView()
}
