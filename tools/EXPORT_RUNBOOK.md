# 训练完 → 导出 → 换进 App(明早照着跑)

训练产物:`~/insightface/recognition/arcface_torch/work_dirs/glint360k_r50_robust/model.pt`
(r50 骨干 state_dict,每个 epoch 覆盖保存;16×H200,~2h,16 epoch)

---

## ① hx2:确认训练完 + 导出 ONNX

```bash
ssh hx2
cd ~/insightface/recognition/arcface_torch
tail -5 glint-r50-robust-98560.out          # 看到 16 个 epoch 跑完 / job 结束
ls -la work_dirs/glint360k_r50_robust/model.pt

source ~/miniforge3/etc/profile.d/conda.sh && conda activate facetrain
python torch2onnx.py work_dirs/glint360k_r50_robust/model.pt \
    --network r50 --output ~/r50_glint_robust.onnx
ls -la ~/r50_glint_robust.onnx              # r50 ≈ 166MB
```

## ② 下载 ONNX 到 Mac

在 **Mac** 上跑(不是 hx2):
```bash
scp hx2:~/r50_glint_robust.onnx /Volumes/Data/Projects/ios/FaceID/tools/
```

## ③ Mac:ONNX → Core ML(.mlpackage)

```bash
cd /Volumes/Data/Projects/ios/FaceID/tools
/opt/anaconda3/bin/python onnx_to_coreml.py \
    r50_glint_robust.onnx ../FaceID/ArcFaceR50.mlpackage
# 末行应打印  fp16-vs-torch 余弦≈0.999
```

## ④ 换进 App(把 mbf 换成 r50)

r50 与 mbf 输入/输出契约**完全相同**([1,3,112,112]→512,input/embedding),
所以 `FaceEmbedder.swift` 预处理、对齐、C++ 引擎**都不用改**,只换模型文件名:

- `FaceEmbedder.swift` 第 26 行:
  `forResource: "ArcFaceMBF"` → `forResource: "ArcFaceR50"`
- 删掉旧的 `FaceID/ArcFaceMBF.mlpackage`(同步分组会自动从编译里移除)
- 新的 `FaceID/ArcFaceR50.mlpackage` 已在上一步落到位,自动纳入编译

```bash
# 验证仍能编译(CLI 只验代码,真机 Run 走 Xcode GUI 签名)
cd /Volumes/Data/Projects/ios/FaceID
xcodebuild -project FaceID.xcodeproj -scheme FaceID \
  -destination 'platform=iOS,id=00008027-000D71543CF0402E' \
  -derivedDataPath /Volumes/App/DeveloperData/DerivedData-FaceID-cli \
  CODE_SIGNING_ALLOWED=NO build
```

## ⑤ iPad 实测(关键提醒)

**换模型 = 向量空间变了,旧 faces.db 录入全部失效**:
进 App 先点右上角 🗑 清空,再重新录入,然后测
「同人不同尺寸 / 戴不戴眼镜」——这正是这次训练要改善的点。

---

### 备注
- r50 比 mbf 大(166MB vs ~5MB)、慢一点,但准确率/鲁棒性更高;iPad 上 Neural Engine 跑 fp16 完全够实时。
- 若想同时保留 mbf 做对比,可两个 .mlpackage 都留着,只改 `forResource` 切换。
- ONNX 导出脚本接口已确认:`torch2onnx.py <model.pt> --network r50 --output x.onnx`(输入名 `data`、动态 batch),转换脚本 `onnx_to_coreml.py` 会把 I/O 名映射回 `input`/`embedding`。
