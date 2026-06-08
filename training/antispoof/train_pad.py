#!/usr/bin/env python
"""
Face anti-spoofing (PAD) trainer — binary live vs. spoof on CelebA-Spoof.

The shipped iOS app fuses this RGB classifier with a TrueDepth planarity check.
This model's job: learn the *texture* cues of a presentation attack (screen
moiré, print grain, reflections, bezels) so that — even on devices without a
depth camera — a replayed video or printed photo scores as "spoof".

Data: Ar4ikov/celebA_spoof (HF mirror of CelebA-Spoof, parquet, ~526K imgs).
      columns: Filepath(Image), Bbox([x1,y1,x2,y2]), Class("live"|"spoof").
Crop: face box expanded by `margin` (context like screen edges matters for PAD).
Backbone: torchvision MobileNetV3-Large (ImageNet-pretrained), 224x224, 2-class.

Run (single GPU is plenty):
    python train_pad.py --epochs 4 --bs 256 --out work_dirs/pad_mbv3
Outputs: work_dirs/pad_mbv3/best.pt  +  pad_mbv3.onnx (input "image"[1,3,224,224], output "logits"[1,2])
"""
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
import datasets as hfds

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SIZE = 224

def build_tf(train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),     # mild — texture/color is a spoof cue
            T.RandomApply([T.GaussianBlur(3)], p=0.1),
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([T.Resize(SIZE + 32), T.CenterCrop(SIZE),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class CelebASpoof(Dataset):
    def __init__(self, split, train, margin=0.4):
        self.ds = hfds.load_dataset("Ar4ikov/celebA_spoof", split=split)
        self.tf = build_tf(train); self.margin = margin

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r["Filepath"]
        if img.mode != "RGB": img = img.convert("RGB")
        W, H = img.size
        bb = r["Bbox"]
        crop = img
        if bb and len(bb) == 4:
            x1, y1, x2, y2 = [float(v) for v in bb]      # treat as x1,y1,x2,y2
            if x2 > x1 and y2 > y1:
                w, h = x2 - x1, y2 - y1
                mx, my = w * self.margin, h * self.margin
                cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
                cx2 = min(W, int(x2 + mx)); cy2 = min(H, int(y2 + my))
                if cx2 - cx1 > 8 and cy2 - cy1 > 8:
                    crop = img.crop((cx1, cy1, cx2, cy2))
        label = 0 if str(r["Class"]).lower() == "live" else 1   # 0=live, 1=spoof
        return self.tf(crop), label

def make_model():
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, 2)
    return m

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); correct = tot = 0; tp = fp = fn = tn = 0
    for x, y in loader:
        x = x.to(dev, non_blocking=True); y = y.to(dev)
        p = model(x).argmax(1)
        correct += (p == y).sum().item(); tot += y.numel()
        tp += ((p == 1) & (y == 1)).sum().item(); tn += ((p == 0) & (y == 0)).sum().item()
        fp += ((p == 1) & (y == 0)).sum().item(); fn += ((p == 0) & (y == 1)).sum().item()
    acc = correct / max(1, tot)
    apcer = fp / max(1, fp + tn)   # live misclassified as... (attack presentation classification err proxies)
    bpcer = fn / max(1, fn + tp)
    return acc, apcer, bpcer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="work_dirs/pad_mbv3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = CelebASpoof("train", train=True)
    va = CelebASpoof("valid", train=False)
    print(f"[data] train={len(tr)}  valid={len(va)}")
    trl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                     pin_memory=True, drop_last=True, persistent_workers=True)
    val = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                     pin_memory=True, persistent_workers=True)

    model = make_model().to(dev)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model); print(f"[gpu] DataParallel x{torch.cuda.device_count()}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(trl))
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for it, (x, y) in enumerate(trl):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if it % 100 == 0:
                print(f"ep{ep} it{it}/{len(trl)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        acc, apcer, bpcer = evaluate(model, val, dev)
        print(f"[val] ep{ep} acc={acc*100:.2f}% APCER={apcer*100:.2f}% BPCER={bpcer*100:.2f}%", flush=True)
        sd = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
        torch.save(sd, os.path.join(args.out, "last.pt"))
        if acc > best:
            best = acc; torch.save(sd, os.path.join(args.out, "best.pt"))
            print(f"[val] new best {best*100:.2f}%", flush=True)

    # export ONNX (input already ImageNet-normalized NCHW; normalization done app-side)
    net = make_model()
    net.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location="cpu"))
    net.eval()
    onnx_path = args.out + ".onnx"
    torch.onnx.export(net, torch.randn(1, 3, SIZE, SIZE), onnx_path,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes=None, opset_version=13)
    print(f"✅ best acc {best*100:.2f}%  → {onnx_path}")

if __name__ == "__main__":
    main()
