# Standard verification benchmarks

InsightFace `.bin` sets (pre-aligned 112×112, flip-TTA, 10-fold accuracy). Same exported ONNX as the app.

| Benchmark | MobileFaceNet (baseline) | ResNet-50 (ours) |
|---|---|---|
| LFW | 99.60% ± 0.25% | 99.77% ± 0.26% |
| CFP-FP (cross-pose) | 96.01% ± 1.10% | 97.44% ± 0.95% |
| AgeDB-30 (cross-age) | 96.35% ± 0.63% | 98.10% ± 0.73% |