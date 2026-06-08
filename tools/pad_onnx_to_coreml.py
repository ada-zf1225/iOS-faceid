#!/usr/bin/env python
"""
Anti-spoofing (PAD) ONNX -> Core ML (.mlpackage).

契约对齐 iOS 端 PADClassifier.swift:
  输入  name="image"  shape=[1,3,224,224]  NCHW  (App 端做 ImageNet 归一化)
  输出  name="logits" [1,2]                       (App 端 softmax → spoof 概率)

用法:
  /opt/anaconda3/bin/python pad_onnx_to_coreml.py pad_mbv3.onnx ../FaceID/FaceSpoofMBV3.mlpackage

链路同 ArcFace:onnx -> onnx2torch -> trace -> coremltools(mlprogram, fp16)。
"""
import sys, numpy as np, torch
from onnx2torch import convert
import coremltools as ct

def main(onnx_path, out_path):
    print(f"[1/4] 载入 ONNX → PyTorch: {onnx_path}")
    net = convert(onnx_path).eval()
    print("[2/4] trace [1,3,224,224]")
    ex = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        ref = net(ex)
        if isinstance(ref, (tuple, list)): ref = ref[0]
    traced = torch.jit.trace(net, ex)
    print("[3/4] coremltools.convert → mlprogram fp16")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, 224, 224), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    print(f"[4/4] 保存: {out_path}")
    mlmodel.save(out_path)
    pred = mlmodel.predict({"image": ex.numpy()})["logits"].reshape(-1)
    r = ref.numpy().reshape(-1)
    def softmax(x): e = np.exp(x - x.max()); return e / e.sum()
    print(f"✅ done. logits torch={r.round(3)}  coreml={pred.round(3)}  "
          f"spoof_p(torch)={softmax(r)[1]:.3f} spoof_p(coreml)={softmax(pred)[1]:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
