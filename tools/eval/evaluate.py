#!/usr/bin/env python
"""
FaceID 量化评测:把「感觉效果可以了」变成硬数据。

两部分:
  1) 标准 LFW 人脸验证      —— ROC / AUC / EER / 最佳阈值准确率 / TAR@FAR,
                               同人 vs 陌生人余弦分布。两模型对比。
  2) 鲁棒性压力测试(本项目卖点)—— 把对齐后人脸做「降采样(模拟远/小)」和
                               「眼区遮挡(模拟眼镜)」逐级退化,看同人余弦掉多少。
                               自训 r50(带鲁棒增广)应当掉得更慢。

对齐:用 insightface 本地 buffalo_s 检测 + face_align.norm_crop,
      和 iOS 端 FaceEmbedder.swift 同一套 5 点 ArcFace 模板,评测忠于部署。

跑:  /opt/anaconda3/bin/python evaluate.py            # 默认 LFW test 1000 对
     /opt/anaconda3/bin/python evaluate.py --subset 10_folds   # 全量 6000 对
     /opt/anaconda3/bin/python evaluate.py --limit 200         # 快速冒烟

LFW 由 sklearn 自动下载并缓存到 tools/eval/_work/(外置卷、已 gitignore)。
"""
import os, sys, argparse, json
import numpy as np
import cv2
import onnxruntime as ort
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_work")
RES  = os.path.join(HERE, "results")
os.makedirs(WORK, exist_ok=True); os.makedirs(RES, exist_ok=True)

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = {
    "ResNet-50 (ours)":        (os.path.join(REPO, "tools", "r50_glint_robust.onnx"), "data"),
    "MobileFaceNet (baseline)": (os.path.expanduser("~/.insightface/models/buffalo_s/w600k_mbf.onnx"), "input.1"),
}
COLORS = {"ResNet-50 (ours)": "#d62728", "MobileFaceNet (baseline)": "#1f77b4"}

# ----------------------------------------------------------------------------- models / alignment
def load_model(path, inp):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return {"sess": sess, "inp": inp, "out": sess.get_outputs()[0].name}

def embed(model, crop_rgb_u8):
    """crop_rgb_u8: HxWx3 uint8 RGB 112x112 -> L2-normalized 512-d."""
    x = crop_rgb_u8.astype(np.float32)
    x = (x - 127.5) / 127.5
    x = np.transpose(x, (2, 0, 1))[None]              # 1,3,112,112 (NCHW)
    v = model["sess"].run([model["out"]], {model["inp"]: x})[0][0]
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

_detector = None
def detector():
    global _detector
    if _detector is None:
        from insightface.app import FaceAnalysis
        _detector = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"])
        _detector.prepare(ctx_id=-1, det_size=(640, 640))   # ctx_id=-1 => CPU
    return _detector

def to_u8_rgb(img):
    a = np.asarray(img)
    if a.ndim == 2:                                   # gray -> 3ch
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.dtype != np.uint8:
        if a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a

def align_rgb112(img_rgb):
    """检测最大脸 -> 5点 norm_crop 到 112 RGB。检测失败则中心方块缩放(并记 miss)。"""
    from insightface.utils import face_align
    bgr = img_rgb[:, :, ::-1].copy()
    faces = detector().get(bgr)
    if faces:
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        aligned_bgr = face_align.norm_crop(bgr, f.kps, image_size=112)
        return aligned_bgr[:, :, ::-1].copy(), True
    h, w = img_rgb.shape[:2]; m = min(h, w)
    y0, x0 = (h-m)//2, (w-m)//2
    crop = img_rgb[y0:y0+m, x0:x0+m]
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA), False

# ----------------------------------------------------------------------------- degradations
def degrade_scale(crop, px):
    """降采样到 px 再放回 112,模拟远处/小尺寸人脸。px>=112 表示不退化。"""
    if px >= 112: return crop
    small = cv2.resize(crop, (px, px), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)

def degrade_occlude(crop, frac):
    """在眼区盖一条黑带(占脸高比例 frac),模拟墨镜/遮挡。frac=0 不退化。"""
    if frac <= 0: return crop
    c = crop.copy()
    y0 = int(112 * 0.34); h = int(112 * frac)
    c[y0:y0+h, :, :] = 0
    return c

# ----------------------------------------------------------------------------- metrics
def metrics(y, scores):
    fpr, tpr, thr = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2); eer_thr = float(thr[i])
    accs = [np.mean((scores >= t) == y) for t in thr]
    bi = int(np.argmax(accs)); best_acc = float(accs[bi]); best_thr = float(thr[bi])
    def tar_at(far):
        ok = fpr <= far
        return float(tpr[ok].max()) if ok.any() else float("nan")
    g = scores[y == 1]; im = scores[y == 0]
    return dict(auc=float(roc_auc), eer=eer, eer_thr=eer_thr,
                best_acc=best_acc, best_thr=best_thr,
                tar_far1e2=tar_at(1e-2), tar_far1e3=tar_at(1e-3),
                gen_mean=float(g.mean()), gen_std=float(g.std()),
                imp_mean=float(im.mean()), imp_std=float(im.std()),
                fpr=fpr, tpr=tpr)

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="test", choices=["test", "10_folds", "train"])
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 对(冒烟用)")
    args = ap.parse_args()

    print(f"[data] 加载 LFW pairs subset={args.subset} ...")
    # slice_=None 取完整 250×250 漏斗图(带背景),否则默认紧裁 125×94 检测不到脸。
    lfw = fetch_lfw_pairs(subset=args.subset, color=True, resize=1.0,
                          funneled=True, slice_=None,
                          data_home=os.path.join(WORK, "sklearn_data"))
    y_all = lfw.target.astype(int)
    if args.limit > 0:                      # 均衡取样(test 子集前半 genuine、后半 impostor)
        h = max(1, args.limit // 2)
        idx = np.concatenate([np.arange(h), np.arange(len(y_all) - h, len(y_all))])
    else:
        idx = np.arange(len(y_all))
    pairs, y = lfw.pairs[idx], y_all[idx]; n = len(y)
    print(f"[data] {n} 对 (genuine={int(y.sum())}, impostor={int((y==0).sum())})")

    models = {name: load_model(p, inp) for name, (p, inp) in MODELS.items()}

    # 对齐所有图(两模型共用),缓存裁剪
    print("[align] 检测+对齐(insightface norm_crop,同 App 模板)...")
    cropsA, cropsB, miss = [], [], 0
    for k in range(n):
        a, oka = align_rgb112(to_u8_rgb(pairs[k, 0]))
        b, okb = align_rgb112(to_u8_rgb(pairs[k, 1]))
        cropsA.append(a); cropsB.append(b); miss += (not oka) + (not okb)
        if (k+1) % 100 == 0: print(f"  aligned {k+1}/{n}")
    print(f"[align] 检测失败回退中心裁剪: {miss}/{2*n}")

    # 各模型:嵌入 + 配对余弦 + 指标
    results, embA, embB = {}, {}, {}
    for name, m in models.items():
        ea = np.stack([embed(m, c) for c in cropsA])
        eb = np.stack([embed(m, c) for c in cropsB])
        embA[name], embB[name] = ea, eb
        scores = np.sum(ea * eb, axis=1)              # 余弦(已归一化)
        mt = metrics(y, scores)
        results[name] = mt
        np.save(os.path.join(WORK, f"scores_{name.split()[0]}.npy"), scores)
        print(f"\n=== {name} ===")
        print(f"  AUC={mt['auc']:.4f}  EER={mt['eer']*100:.2f}%  "
              f"最佳准确率={mt['best_acc']*100:.2f}% @thr={mt['best_thr']:.3f}")
        print(f"  TAR@FAR=1e-2={mt['tar_far1e2']*100:.2f}%  TAR@FAR=1e-3={mt['tar_far1e3']*100:.2f}%")
        print(f"  同人余弦={mt['gen_mean']:.3f}±{mt['gen_std']:.3f}  "
              f"陌生人={mt['imp_mean']:.3f}±{mt['imp_std']:.3f}")

    # ---- 图1: ROC ----
    plt.figure(figsize=(5.2, 5))
    for name, mt in results.items():
        plt.plot(mt["fpr"], mt["tpr"], color=COLORS[name], lw=2,
                 label=f"{name}  (AUC={mt['auc']:.4f}, EER={mt['eer']*100:.2f}%)")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    plt.xscale("log"); plt.xlim(1e-3, 1); plt.ylim(0, 1.005)
    plt.xlabel("False Accept Rate (log)"); plt.ylabel("True Accept Rate")
    plt.title(f"LFW Verification ROC  (n={n} pairs)"); plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RES, "roc.png"), dpi=150); plt.close()

    # ---- 图2: 分数分布 ----
    fig, axes = plt.subplots(1, len(models), figsize=(5.2*len(models), 4), sharex=True, sharey=True)
    if len(models) == 1: axes = [axes]
    for ax, (name, m) in zip(axes, models.items()):
        sc = np.sum(embA[name]*embB[name], axis=1)
        ax.hist(sc[y==1], bins=40, alpha=0.6, color="#2ca02c", label="genuine")
        ax.hist(sc[y==0], bins=40, alpha=0.6, color="#9467bd", label="impostor")
        ax.axvline(results[name]["eer_thr"], color="k", ls="--", lw=1,
                   label=f"EER thr={results[name]['eer_thr']:.2f}")
        ax.set_title(name); ax.set_xlabel("cosine"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_ylabel("count"); plt.tight_layout()
    plt.savefig(os.path.join(RES, "score_dist.png"), dpi=150); plt.close()

    # ---- 鲁棒性:只用同人对,退化 B,看同人余弦均值 ----
    gidx = np.where(y == 1)[0]
    scale_px = [112, 64, 48, 32, 24, 16]
    occ_frac = [0.0, 0.1, 0.2, 0.3, 0.4]
    rob = {"scale": {}, "occ": {}}
    for name, m in models.items():
        ea = embA[name][gidx]
        rob["scale"][name] = []
        for px in scale_px:
            eb = np.stack([embed(m, degrade_scale(cropsB[i], px)) for i in gidx])
            rob["scale"][name].append(float(np.mean(np.sum(ea*eb, axis=1))))
        rob["occ"][name] = []
        for fr in occ_frac:
            eb = np.stack([embed(m, degrade_occlude(cropsB[i], fr)) for i in gidx])
            rob["occ"][name].append(float(np.mean(np.sum(ea*eb, axis=1))))

    def rob_plot(key, xs, xlabel, title, fname, invert=False):
        plt.figure(figsize=(5.4, 4.2))
        xv = [(112//x if invert else x) for x in xs] if False else xs
        for name in models:
            plt.plot(xs, rob[key][name], "o-", color=COLORS[name], lw=2, label=name)
        plt.xlabel(xlabel); plt.ylabel("mean genuine cosine"); plt.title(title)
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(RES, fname), dpi=150); plt.close()
    rob_plot("scale", scale_px, "downsampled face size (px) — smaller = farther",
             "Robustness to scale (genuine pairs)", "robustness_scale.png")
    rob_plot("occ", occ_frac, "eye-region occlusion (fraction of face height)",
             "Robustness to occlusion / glasses (genuine pairs)", "robustness_occlusion.png")

    # ---- 写 RESULTS.md ----
    def row(name):
        m = results[name]
        return (f"| {name} | {m['auc']:.4f} | {m['eer']*100:.2f}% | {m['best_acc']*100:.2f}% "
                f"| {m['best_thr']:.3f} | {m['tar_far1e3']*100:.2f}% "
                f"| {m['gen_mean']:.3f}±{m['gen_std']:.3f} | {m['imp_mean']:.3f}±{m['imp_std']:.3f} |")
    lines = [
        "# Evaluation results",
        "",
        f"Benchmark: **LFW** verification, `subset={args.subset}` ({n} pairs). "
        "Faces detected + 5-point aligned with the same InsightFace template the iOS app uses. "
        "Embeddings from the exported ONNX (numerically ≡ the shipped fp16 Core ML, cosine 0.9984).",
        "",
        "## Verification accuracy",
        "",
        "| Model | AUC | EER | Best acc | Best thr | TAR@FAR=1e-3 | genuine cos | impostor cos |",
        "|---|---|---|---|---|---|---|---|",
        row("MobileFaceNet (baseline)"),
        row("ResNet-50 (ours)"),
        "",
        "![ROC](results/roc.png)",
        "",
        "![Score distributions](results/score_dist.png)",
        "",
        "## Robustness stress test (the project's thesis)",
        "",
        "Genuine pairs only; one image is progressively degraded, and we plot how far the "
        "same-person cosine falls. The self-trained ResNet-50 was trained with downscale and "
        "random-erasing augmentation specifically to resist these two failure modes.",
        "",
        "Mean genuine cosine under **scale** degradation (smaller = farther away):",
        "",
        "| face px | " + " | ".join(str(p) for p in scale_px) + " |",
        "|---|" + "|".join("---" for _ in scale_px) + "|",
    ]
    for name in ["MobileFaceNet (baseline)", "ResNet-50 (ours)"]:
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in rob['scale'][name]) + " |")
    lines += ["", "![Scale robustness](results/robustness_scale.png)", "",
              "Mean genuine cosine under **eye-region occlusion** (glasses / mask proxy):", "",
              "| occlusion frac | " + " | ".join(str(p) for p in occ_frac) + " |",
              "|---|" + "|".join("---" for _ in occ_frac) + "|"]
    for name in ["MobileFaceNet (baseline)", "ResNet-50 (ours)"]:
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in rob['occ'][name]) + " |")
    lines += ["", "![Occlusion robustness](results/robustness_occlusion.png)", ""]
    with open(os.path.join(RES, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines))
    # 机器可读
    dump = {k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")} for k, v in results.items()}
    dump["_meta"] = dict(subset=args.subset, n=n, miss=miss, scale_px=scale_px, occ_frac=occ_frac, robustness=rob)
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\n✅ 写出 {RES}/RESULTS.md + roc.png + score_dist.png + robustness_*.png + results.json")

if __name__ == "__main__":
    main()
