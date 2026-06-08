#!/usr/bin/env python
"""
Standard face-verification benchmarks from InsightFace .bin files:
LFW, CFP-FP (cross-pose), AgeDB-30 (cross-age).

The .bin images are already aligned 112x112, so this SKIPS detection entirely —
just decode + (batched, flip-TTA) embed + cosine + 10-fold accuracy. Fast on CPU.

Models default to the same ONNX as tools/eval/evaluate.py; override via
FACEID_R50_ONNX / FACEID_MBF_ONNX. Bins dir via FACEID_BINS.

Run (on the box where the bins live, e.g. HX2):
    FACEID_BINS=~/casia/faces_webface_112x112 \
    FACEID_R50_ONNX=~/r50_glint_robust.onnx \
    python eval_bins.py
"""
import os, pickle, numpy as np, cv2, onnxruntime as ort
from sklearn.metrics import roc_curve, auc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BINS = os.path.expanduser(os.environ.get("FACEID_BINS", os.path.join(REPO, "tools", "eval", "_work", "bins")))
R50 = os.environ.get("FACEID_R50_ONNX", os.path.join(REPO, "tools", "r50_glint_robust.onnx"))
MBF = os.environ.get("FACEID_MBF_ONNX", os.path.expanduser("~/.insightface/models/buffalo_s/w600k_mbf.onnx"))
CTX = int(os.environ.get("FACEID_CTX", "-1"))
MODELS = {"MobileFaceNet (baseline)": (MBF, "input.1"), "ResNet-50 (ours)": (R50, "data")}
BENCHES = ["lfw", "cfp_fp", "agedb_30"]

def load_model(path, inp):
    # 显式设线程数:避免 onnxruntime 在受限环境(SLURM/login cgroup)里做线程亲和性绑定失败而退化成单线程
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("FACEID_THREADS", "8"))
    so.inter_op_num_threads = 1
    provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] if CTX >= 0 else ["CPUExecutionProvider"]
    s = ort.InferenceSession(path, sess_options=so, providers=provs)
    return {"sess": s, "inp": inp, "out": s.get_outputs()[0].name}

def load_bin(path):
    with open(path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    imgs = np.empty((len(bins), 112, 112, 3), np.uint8)
    for i, b in enumerate(bins):
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)  # BGR
        if im.shape[0] != 112: im = cv2.resize(im, (112, 112))
        imgs[i] = im[:, :, ::-1]                                          # -> RGB
    return imgs, np.array(issame, dtype=int)

def embed_all(model, imgs):                       # imgs N,112,112,3 RGB uint8 -> N,512 L2-norm (flip-TTA)
    def run(batch):
        x = (batch.astype(np.float32) - 127.5) / 127.5
        x = x.transpose(0, 3, 1, 2)
        out = []
        for i in range(0, len(x), 256):
            out.append(model["sess"].run([model["out"]], {model["inp"]: x[i:i+256]})[0])
        return np.concatenate(out)
    e = run(imgs) + run(imgs[:, :, ::-1, :])      # original + horizontal flip
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    return e

def kfold_acc(scores, y, folds=10):
    n = len(y); idx = np.arange(n); fs = n // folds
    cand = np.unique(scores); accs = []
    for f in range(folds):
        te = idx[f*fs:(f+1)*fs] if f < folds-1 else idx[f*fs:]
        tr = np.setdiff1d(idx, te)
        bt, ba = 0.0, 0.0
        for t in cand:
            a = np.mean((scores[tr] >= t) == y[tr])
            if a > ba: ba, bt = a, t
        accs.append(np.mean((scores[te] >= bt) == y[te]))
    return float(np.mean(accs)), float(np.std(accs))

def eer_of(scores, y):
    fpr, tpr, _ = roc_curve(y, scores); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(auc(fpr, tpr))

def main():
    models = {n: load_model(p, i) for n, (p, i) in MODELS.items()}
    table = {}   # bench -> model -> (acc, std, eer, aucv)
    for bench in BENCHES:
        path = os.path.join(BINS, bench + ".bin")
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在"); continue
        imgs, issame = load_bin(path)
        print(f"\n=== {bench}  ({len(issame)} pairs) ===")
        table[bench] = {}
        for name, m in models.items():
            emb = embed_all(m, imgs)
            scores = np.sum(emb[0::2] * emb[1::2], axis=1)
            acc, std = kfold_acc(scores, issame)
            eer, aucv = eer_of(scores, issame)
            table[bench][name] = (acc, std, eer, aucv)
            print(f"  {name:26s} acc={acc*100:.2f}%±{std*100:.2f}%  EER={eer*100:.2f}%  AUC={aucv:.4f}")

    # markdown
    lines = ["# Standard verification benchmarks", "",
             "InsightFace `.bin` sets (pre-aligned 112×112, flip-TTA, 10-fold accuracy). "
             "Same exported ONNX as the app.", "",
             "| Benchmark | " + " | ".join(MODELS) + " |",
             "|---|" + "|".join("---" for _ in MODELS) + "|"]
    pretty = {"lfw": "LFW", "cfp_fp": "CFP-FP (cross-pose)", "agedb_30": "AgeDB-30 (cross-age)"}
    for bench in BENCHES:
        if bench not in table: continue
        cells = [f"{table[bench][n][0]*100:.2f}% ± {table[bench][n][1]*100:.2f}%" for n in MODELS]
        lines.append(f"| {pretty[bench]} | " + " | ".join(cells) + " |")
    open(os.path.join(RES, "RESULTS_bench.md"), "w").write("\n".join(lines))

    # bar chart
    benches = [b for b in BENCHES if b in table]
    x = np.arange(len(benches)); w = 0.38
    plt.figure(figsize=(6.4, 4))
    for k, name in enumerate(MODELS):
        vals = [table[b][name][0]*100 for b in benches]
        plt.bar(x + (k-0.5)*w, vals, w, label=name)
        for xi, v in zip(x + (k-0.5)*w, vals): plt.text(xi, v+0.1, f"{v:.2f}", ha="center", fontsize=7)
    plt.xticks(x, [pretty[b] for b in benches]); plt.ylim(80, 100.5)
    plt.ylabel("10-fold accuracy (%)"); plt.title("Verification benchmarks: ResNet-50 (ours) vs MobileFaceNet")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RES, "benchmarks.png"), dpi=150)
    print(f"\n✅ {RES}/RESULTS_bench.md + benchmarks.png")

if __name__ == "__main__":
    main()
