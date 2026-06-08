# Anti-spoofing (liveness / PAD) model

A second trained model, separate from face recognition: a binary **live vs. spoof**
classifier. On-device it is **fused with a TrueDepth planarity check** — depth defeats
flat attacks (printed photo, replayed video on a screen) because a screen has no 3-D
structure; this RGB CNN adds a learned texture cue (moiré, print grain, reflections)
and works on devices without a depth camera.

## Recipe

| | |
|---|---|
| Data | [Ar4ikov/celebA_spoof](https://hf.co/datasets/Ar4ikov/celebA_spoof) — HF parquet mirror of CelebA-Spoof, ~420K train / 47K val, `Class ∈ {live, spoof}` + face `Bbox` |
| Crop | face box expanded by 40% margin (context like screen bezels matters for PAD), 224×224 |
| Backbone | torchvision MobileNetV3-Large (ImageNet-pretrained), 2-class head |
| Loss / opt | cross-entropy, AdamW lr 3e-4 cosine, AMP, 4 epochs |
| Hardware | 1 node × 2 H200 (data-loading bound; many workers) |
| Export | → ONNX (`image`[1,3,224,224] → `logits`[1,2]), then Core ML on the Mac |

Metrics reported: accuracy, **APCER** (attack accepted as live) and **BPCER** (live rejected).

**Result (4 epochs, CelebA-Spoof val, 47K images):** acc **99.93%**, APCER **0.10%**, BPCER **0.06%**.

| epoch | acc | APCER | BPCER |
|---|---|---|---|
| 0 | 99.85% | 0.41% | 0.01% |
| 1 | 99.88% | 0.16% | 0.10% |
| 2 | 99.91% | 0.14% | 0.06% |
| 3 | **99.93%** | **0.10%** | 0.06% |

> In-distribution accuracy is very high; cross-device generalization of pure-RGB PAD is the known
> hard part — which is exactly why the app fuses this with the TrueDepth planarity check rather
> than trusting it alone.

## Reproduce

```bash
conda activate facetrain
pip install datasets                       # torchvision already present
mkdir -p ~/antispoof && cd ~/antispoof
# pre-download (datasets caches the parquet):
python -c "import datasets; datasets.load_dataset('Ar4ikov/celebA_spoof', split='train'); datasets.load_dataset('Ar4ikov/celebA_spoof', split='valid')"
sbatch submit_pad.sh                       # → work_dirs/pad_mbv3/best.pt + pad_mbv3.onnx
```

## Into the app

ONNX → Core ML (separate from ArcFace; input is **ImageNet-normalized** 224×224, not the
ArcFace `(x-127.5)/127.5`). The app runs it per detected face (throttled) and fuses:

```
live  ⇔  depth says 3-D (TrueDepth planarity)   AND   CNN says live
```

with a UI toggle between *display-only* (show "who + live/spoof") and *gated* (a spoof is
never given an identity).

> Note: pure-RGB PAD generalizes imperfectly across unseen capture devices — that's exactly
> why it is fused with depth rather than trusted alone.
