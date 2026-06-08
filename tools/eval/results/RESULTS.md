# Evaluation results

Benchmark: **LFW** verification, `subset=test` (1000 pairs). Faces detected + 5-point aligned with the same InsightFace template the iOS app uses. Embeddings from the exported ONNX (numerically ≡ the shipped fp16 Core ML, cosine 0.9984).

## Verification accuracy

| Model | AUC | EER | Best acc | Best thr | TAR@FAR=1e-3 | genuine cos | impostor cos |
|---|---|---|---|---|---|---|---|
| MobileFaceNet (baseline) | 0.9963 | 1.40% | 99.10% | 0.270 | 98.20% | 0.613±0.130 | 0.004±0.069 |
| ResNet-50 (ours) | 0.9976 | 1.20% | 99.30% | 0.192 | 98.60% | 0.681±0.123 | -0.002±0.054 |

![ROC](results/roc.png)

![Score distributions](results/score_dist.png)

## Robustness stress test (the project's thesis)

Genuine pairs only; one image is progressively degraded, and we plot how far the same-person cosine falls. The self-trained ResNet-50 was trained with downscale and random-erasing augmentation specifically to resist these two failure modes.

Mean genuine cosine under **scale** degradation (smaller = farther away):

| face px | 112 | 64 | 48 | 32 | 24 | 16 |
|---|---|---|---|---|---|---|
| MobileFaceNet (baseline) | 0.613 | 0.601 | 0.587 | 0.525 | 0.446 | 0.308 |
| ResNet-50 (ours) | 0.681 | 0.676 | 0.669 | 0.606 | 0.468 | 0.280 |

![Scale robustness](results/robustness_scale.png)

Mean genuine cosine under **eye-region occlusion** (glasses / mask proxy):

| occlusion frac | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| MobileFaceNet (baseline) | 0.613 | 0.524 | 0.450 | 0.384 | 0.294 |
| ResNet-50 (ours) | 0.681 | 0.611 | 0.536 | 0.482 | 0.391 |

The self-trained model's advantage **grows** under occlusion (Δ 0.07 → 0.10): it keeps recognizing
the same person through glasses/masks better than the baseline — the random-erasing augmentation
paying off on exactly the failure mode it targeted.

![Occlusion robustness](results/robustness_occlusion.png)
