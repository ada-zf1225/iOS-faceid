<div align="center">

# FaceID — On-Device Face Recognition for iOS

**Real-time face enrollment & recognition that runs entirely on the device — powered by a custom ArcFace model trained from scratch on 16× NVIDIA H200.**

[![Platform](https://img.shields.io/badge/platform-iOS%2018.5%2B-blue.svg)](https://developer.apple.com/ios/)
[![Swift](https://img.shields.io/badge/Swift-6-orange.svg)](https://swift.org)
[![Core ML](https://img.shields.io/badge/Core%20ML-ArcFace%20R50-green.svg)](https://developer.apple.com/machine-learning/core-ml/)
[![Engine](https://img.shields.io/badge/engine-C%2B%2B17-lightgrey.svg)](FaceID/FaceEngine.hpp)
[![License](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)

<!-- Demo:录一段真机演示存成 docs/demo.gif(见 docs/HOWTO-demo.md),然后取消下面一行的注释 -->
<!-- <img src="docs/demo.gif" alt="Demo" width="280"/> -->

</div>

---

## What it is

FaceID points the front camera at a face, decides **whether that person is already in a local database**, and draws a green box + name when it recognizes them (red "stranger" otherwise). Everything — detection, alignment, embedding, matching — runs **on-device**; no network, no cloud, no data leaves the phone.

What makes this more than a wrapper around a pretrained model:

- **The recognition model is trained from scratch.** A ResNet-50 ArcFace backbone trained on Glint360K (360K identities, 17M images) on a 16-GPU H200 cluster, with a **custom augmentation pipeline specifically targeting the two failure modes that break naive face apps: glasses/occlusion and scale changes** (enroll close-up, recognize far away).
- **Clean, reusable architecture.** The UI/camera/detection layer is Swift; the matching engine is **pure C++17** behind a thin Objective-C++ bridge, so the same engine can be reused on Android or a desktop CLI without change.
- **Geometric face alignment**, not just a bounding-box crop — a closed-form similarity transform maps 5 facial landmarks onto the ArcFace template, which is what makes embeddings stable across pose and distance.

## Features

| | |
|---|---|
| 🎯 Real-time recognition | Front-camera live preview, ~7 fps recognition pipeline, green/red boxes + name + score |
| 🧬 5-point alignment | Vision landmarks → closed-form similarity transform → 112×112 ArcFace template |
| 🧠 Self-trained ArcFace R50 | 512-d embeddings, fp16 Core ML, runs on the Neural Engine |
| 💾 On-device enrollment | Tap to enroll; persisted locally, survives relaunch |
| 📸 Multi-shot enrollment | One tap captures several quality-gated frames → multiple templates per person → better recall across pose/glasses |
| 👤 Identity management | List enrolled people, rename, swipe-to-delete (no more clear-all only) |
| ✅ Quality gate + smoothing | Rejects too-far / off-angle faces at enroll; temporal vote stabilizes the on-screen label |
| 🎚️ Tunable threshold | Live cosine-similarity threshold slider for FAR/FRR trade-off |
| 🔒 Fully offline | No network calls; embeddings + database never leave the device |

## How it works

```mermaid
flowchart LR
    A[AVCaptureSession<br/>front camera] --> B[Vision<br/>VNDetectFaceLandmarks]
    B -->|5 landmarks| C[Similarity transform<br/>→ 112×112 aligned crop]
    C -->|NCHW, x-127.5/127.5| D[Core ML<br/>ArcFace R50]
    D -->|512-d embedding| E[C++ FaceEngine<br/>L2-norm + cosine]
    E -->|best match ≥ threshold| F[SwiftUI overlay<br/>name / stranger]
```

1. **Capture** — `AVCaptureSession` with the front camera; orientation handled by `AVCaptureDevice.RotationCoordinator` so preview and analysis frames agree in any device rotation.
2. **Detect** — `VNDetectFaceLandmarksRequest` returns a bounding box and facial landmarks per frame (throttled to ~7 fps).
3. **Align** — 5 points (eyes, nose, mouth corners) are mapped onto ArcFace's canonical template via a **closed-form least-squares similarity transform** (no SVD). Alignment is the single biggest lever for cross-pose / cross-distance stability.
4. **Embed** — the aligned 112×112 RGB crop is normalized `(x − 127.5)/127.5`, fed NCHW into the Core ML ArcFace R50 model, out comes a 512-d vector.
5. **Match** — the pure-C++ `FaceEngine` L2-normalizes the vector and compares it against every enrolled identity by cosine similarity; the best score ≥ threshold wins.

## Architecture

```
┌──────────────────────────── Swift / SwiftUI ────────────────────────────┐
│  ContentView         camera preview · boxes · enroll UI · threshold      │
│  CameraModel         AVFoundation + Vision detection + alignment points   │
│  FaceEmbedder        Core ML ArcFace R50  (align → preprocess → infer)    │
│  CameraPreview       AVCaptureVideoPreviewLayer bridge                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                    │  Objective-C++ bridge
┌──────────────────────────────────▼──────────────────────────────────────┐
│  FaceEngineBridge (.mm)   ObjC++ wrapper, [Float] ⇆ std::vector<float>    │
│  FaceEngine (.cpp/.hpp)   PURE C++17 — L2-norm, cosine, file persistence, │
│                           std::mutex thread-safety. No Apple dependency.  │
└───────────────────────────────────────────────────────────────────────── ┘
```

The C++ core has **zero Apple dependencies** — that's deliberate. See [`training/`](training/) for how the model was made and [`tools/`](tools/) for the export pipeline.

## Liveness / anti-spoofing (explored, not shipped)

A liveness prototype was built and then removed from the app: TrueDepth planarity (a flat photo or
replayed video is a plane; a real face isn't) plus a self-trained MobileNetV3 anti-spoof CNN
(CelebA-Spoof, **99.9%** val) — see [`training/antispoof/`](training/antispoof/). Robust RGB+depth
presentation-attack detection needs per-device threshold calibration that was out of scope, so the
shipped app focuses on doing recognition well. The trained model and recipe stay in the repo as a
documented experiment.

## Build & run

> Requires a **physical iOS device** (the recognition pipeline needs a real camera) running iOS 18.5+, **Xcode 16.4+**, and **Git LFS** (the Core ML weights are stored in LFS).

```bash
# 1. Clone WITH the model weights (Git LFS)
git lfs install
git clone <repo-url>
cd FaceID
git lfs pull                       # fetches ArcFaceR50.mlpackage weight.bin (~87 MB)

# 2. Open & sign
open FaceID.xcodeproj
#   - select the FaceID target → Signing & Capabilities → set your Team
#   - pick your connected device as the run destination

# 3. Run (⌘R)
```

First launch asks for camera permission. Then: point at a face, tap **录入当前人脸 / Enroll**, give it a name. Point at the same person again → green box. ⚠️ Because the model defines the embedding space, **if you ever swap the model you must clear the database (🗑) and re-enroll.**

## Project structure

```
FaceID/
├── FaceID/                      # iOS app
│   ├── ContentView.swift            UI: preview, overlays, enroll, threshold
│   ├── CameraModel.swift            camera + Vision detection + landmark extraction
│   ├── CameraPreview.swift          preview layer bridge
│   ├── FaceEmbedder.swift           Core ML ArcFace: alignment + preprocess + inference
│   ├── FaceEngine.{hpp,cpp}         pure C++17 matching engine
│   ├── FaceEngineBridge.{h,mm}      Objective-C++ bridge
│   ├── FaceID-Bridging-Header.h
│   └── ArcFaceR50.mlpackage         recognition model (Git LFS)
├── engine/                      # the C++ engine, OUTSIDE the app
│   ├── tests/test_face_engine.cpp   unit tests (make -C engine test)
│   ├── cli/face_cli.cpp             standalone CLI (cross-platform proof)
│   └── Makefile
├── tools/                       # ONNX → Core ML export
│   ├── onnx_to_coreml.py
│   ├── EXPORT_RUNBOOK.md
│   └── eval/                        quantitative evaluation (ROC, FAR/FRR, robustness)
└── training/                    # how the models were trained (H200)
    ├── *.py / configs / submit       Glint360K ArcFace recipe
    └── antispoof/                    CelebA-Spoof anti-spoof recipe (explored, not shipped)
```

## The model

| | |
|---|---|
| Architecture | ResNet-50 backbone, ArcFace head (512-d) |
| Training data | Glint360K — 360,232 identities, 17M images |
| Loss | CosFace margin (1.0, 0.0, 0.4), PartialFC `sample_rate=0.2` |
| Schedule | 16 epochs, effective batch 4096, lr 0.3 cosine, 1-epoch warmup |
| Hardware | 4 nodes × 4× NVIDIA H200 (16 GPUs), SLURM + `torchrun` (c10d), ~37k img/s, **~2 h wall-clock** |
| Robustness aug | horizontal flip · Gaussian blur · **downscale→upscale (small-face)** · color jitter · **random erasing (glasses/occlusion)** |
| Export | PyTorch → ONNX → Core ML fp16, output cosine vs. PyTorch reference **0.9984** |

Full recipe, augmentation rationale, and one-command reproduction in [`training/`](training/).

## Evaluation

### Standard benchmarks (10-fold accuracy)

On the canonical InsightFace verification sets (pre-aligned 112×112, flip-TTA, standard 10-fold
protocol; same exported ONNX as the app), the self-trained ResNet-50 beats the MobileFaceNet
baseline on **every** benchmark — by the widest margin on **cross-pose** and **cross-age**, exactly
where robustness matters:

| Benchmark | MobileFaceNet (baseline) | ResNet-50 (ours) |
|---|---|---|
| LFW (6000 pairs) | 99.60% ± 0.25% | **99.77% ± 0.26%** |
| CFP-FP — cross-pose (7000) | 96.01% ± 1.10% | **97.44% ± 0.95%** |
| AgeDB-30 — cross-age (6000) | 96.35% ± 0.63% | **98.10% ± 0.73%** |

<p align="center"><img src="tools/eval/results/benchmarks.png" width="62%"/></p>

Reproduce: `python tools/eval/eval_bins.py` (InsightFace `.bin` sets — pre-aligned, so no detection
needed).

### Robustness stress test (the project's thesis)

A separate detection-based pass ([`evaluate.py`](tools/eval/evaluate.py)) degrades genuine pairs to
probe the two failure modes the model was trained for. Under simulated eye-region occlusion
(glasses/mask proxy) the self-trained model's genuine-cosine advantage *grows* with severity — the
random-erasing augmentation paying off:

| genuine cosine @ occlusion | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| MobileFaceNet | 0.613 | 0.524 | 0.450 | 0.384 | 0.294 |
| **ResNet-50 (ours)** | **0.681** | **0.611** | **0.536** | **0.482** | **0.391** |

<p align="center"><img src="tools/eval/results/robustness_occlusion.png" width="48%"/></p>

> The app defaults the cosine threshold to 0.35 (with a live slider); stranger scores cluster at ≈0.

## Roadmap

Done:
- [x] Multi-shot enrollment (several quality-gated templates per identity)
- [x] Per-identity management (rename / swipe-delete a single person)
- [x] Enrollment quality gate (reject off-angle / tiny faces via Vision pose + size)
- [x] Temporal label smoothing (majority vote across frames)
- [x] C++ engine unit tests + standalone CLI (cross-platform proof)
- [~] Anti-spoofing prototype (TrueDepth + RGB CNN) — built & trained, not shipped (see above)

Next:
- [ ] Port the C++ engine to Android (NDK) with a TFLite ArcFace front-end
- [ ] Multi-face tracking with stable IDs (per-track smoothing)
- [ ] Robust on-device liveness with a proper per-device calibration flow
- [ ] Approximate nearest-neighbor index for large galleries

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push:

- **engine-tests** (Ubuntu) — compiles `engine/` against the shipped `FaceEngine.cpp` and runs the
  22 unit tests + builds the CLI. This is the project's real logic coverage, and it's
  platform-independent (no Apple toolchain), reinforcing the reusable-engine design.
- **ios-build** (macOS) — checks out with Git LFS and builds the app for the iOS Simulator
  (no signing), verifying the Swift / Core ML / Objective-C++ all compile.

<!-- After creating the GitHub remote, add the badge:
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml) -->

## Tech stack

`Swift 6` · `SwiftUI` · `AVFoundation` · `Vision` · `Core ML` · `C++17` · `Objective-C++` · `PyTorch` · `InsightFace / arcface_torch` · `coremltools`

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) — `arcface_torch` training recipe and the ArcFace/Glint360K line of work.
- Apple Vision & Core ML.

## License

[MIT](LICENSE). See the note there regarding the trained model weights and upstream dataset/recipe licenses.
