#!/usr/bin/env python
"""
r50 ArcFace ONNX -> Core ML (.mlpackage),契约对齐 App 里的 FaceEmbedder.swift:
  输入  name="input"   shape=[1,3,112,112]  NCHW  (App 端已做 (x-127.5)/127.5)
  输出  name="embedding" 512 维  (App 端 C++ 引擎再 L2 归一化 + 余弦)

用法:
  /opt/anaconda3/bin/python onnx_to_coreml.py r50_glint_robust.onnx ArcFaceR50.mlpackage

链路:onnx -> onnx2torch -> trace -> coremltools(mlprogram, fp16)
和当初转 ArcFaceMBF 用的是同一条链,只是骨干从 mbf 换成 r50。
"""
import sys, numpy as np, torch
from onnx2torch import convert
import coremltools as ct

def main(onnx_path: str, out_path: str):
    print(f"[1/4] 载入 ONNX 并转 PyTorch: {onnx_path}")
    net = convert(onnx_path).eval()

    print("[2/4] trace (example [1,3,112,112])")
    ex = torch.rand(1, 3, 112, 112)
    with torch.no_grad():
        ref = net(ex)
        if isinstance(ref, (tuple, list)):
            ref = ref[0]
    traced = torch.jit.trace(net, ex)

    print("[3/4] coremltools.convert -> mlprogram fp16")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 112, 112), dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )

    print(f"[4/4] 保存: {out_path}")
    mlmodel.save(out_path)

    # 数值校验:Core ML 输出 vs PyTorch 参考,余弦应 ~1.0
    pred = mlmodel.predict({"input": ex.numpy()})["embedding"].reshape(-1)
    r = ref.numpy().reshape(-1)
    cos = float(np.dot(pred, r) / (np.linalg.norm(pred) * np.linalg.norm(r) + 1e-9))
    print(f"✅ done. dim={pred.shape[0]}  fp16-vs-torch 余弦={cos:.4f} (应≈0.999)")
    if cos < 0.99:
        print("⚠️  余弦偏低,检查输入归一化/通道顺序是否与 FaceEmbedder 一致")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
