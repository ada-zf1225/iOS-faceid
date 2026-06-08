# Training the FaceID recognition model

The Core ML model shipped in this app (`FaceID/ArcFaceR50.mlpackage`) is **not** an
off-the-shelf checkpoint — it was trained from scratch for this project. This folder is
everything needed to reproduce it.

## TL;DR

| | |
|---|---|
| Backbone / head | ResNet-50 + ArcFace, 512-d embedding |
| Data | Glint360K — 360,232 identities, 17.1M images, 112×112 aligned |
| Loss | CosFace margin `(1.0, 0.0, 0.4)`, PartialFC `sample_rate=0.2` |
| Schedule | 16 epochs, per-GPU batch 256 (effective 4096), lr 0.3 cosine, 1-epoch warmup |
| Hardware | 4 nodes × 4× NVIDIA H200 (16 GPUs), SLURM + `torchrun` (c10d) |
| Throughput | ~36,800 images/sec → **~2 h wall-clock**, loss 35 → 1.2 |
| Framework | [InsightFace `arcface_torch`](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) |

## Why train at all (the thesis)

A naive face app that wraps a pretrained model tends to fail in two specific ways:

1. **Scale** — enroll a close-up face, then fail to recognize the same person from across the room.
2. **Glasses / occlusion** — putting on (or taking off) glasses tanks the similarity score.

So instead of using the stock weights, the model is trained with an **augmentation pipeline
that deliberately simulates both failure modes** ([`wds_loader.py`](wds_loader.py)):

```python
T.RandomApply([T.Resize(40), T.Resize(112)], p=0.25)   # downscale→upscale  → far/small faces
T.RandomErasing(p=0.25, scale=(0.02, 0.2))             # erase a patch       → glasses/occlusion
T.GaussianBlur(...) ; T.ColorJitter(...)               # motion blur, lighting
```

The quantitative payoff (how much less the genuine cosine drops under scale/occlusion vs. the
baseline MobileFaceNet) is measured in [`../tools/eval/`](../tools/eval/).

## Files

| File | What |
|---|---|
| [`wds_loader.py`](wds_loader.py) | WebDataset loader + the robustness augmentation |
| [`configs/glint360k_r50_robust.py`](configs/glint360k_r50_robust.py) | training config |
| [`submit_glint.sh`](submit_glint.sh) | multi-node SLURM launcher (torchrun c10d) |
| [`insightface_arcface_torch.patch`](insightface_arcface_torch.patch) | minimal diff to vanilla `arcface_torch` |

## Reproduce

### 1. Environment
```bash
conda create -n facetrain python=3.11 -y && conda activate facetrain
pip install torch torchvision webdataset easydict scikit-learn opencv-python
# (CUDA build of torch matching your driver)
git clone https://github.com/deepinsight/insightface
cd insightface/recognition/arcface_torch
```

### 2. Apply this project's changes
```bash
# from the insightface repo root:
git apply /path/to/training/insightface_arcface_torch.patch
cp /path/to/training/wds_loader.py recognition/arcface_torch/
cp /path/to/training/configs/glint360k_r50_robust.py recognition/arcface_torch/configs/
```
The patch makes `dataset.py` route to the WebDataset loader when `config.rec` is a directory of
`*.tar` shards, makes the `mxnet` imports optional (not needed for WebDataset), and fixes
`torch2onnx.py` for modern NumPy / CPU export (see Gotchas).

### 3. Data → WebDataset shards
Download the Glint360K `.rec/.idx` release (per the InsightFace dataset zoo), then convert each
record to a `{jpg, cls}` sample and pack into `*.tar` shards with the `webdataset` writer. Point
`config.rec` at that shard directory. (Why WebDataset: streamed, shardable across nodes, no mxnet
runtime dependency.)

### 4. Train
```bash
# single node, 4 GPUs (smoke / small scale)
torchrun --nproc_per_node=4 train_v2.py configs/glint360k_r50_robust

# multi-node via SLURM (what produced the shipped model)
sbatch submit_glint.sh
```
Checkpoints land in `work_dirs/glint360k_r50_robust/model.pt` (backbone state_dict, overwritten
each epoch).

### 5. Export to Core ML
See [`../tools/EXPORT_RUNBOOK.md`](../tools/EXPORT_RUNBOOK.md): `torch2onnx.py` → ONNX, then
[`../tools/onnx_to_coreml.py`](../tools/onnx_to_coreml.py) → fp16 `.mlpackage`
(output cosine vs. PyTorch reference: 0.9984).

## Gotchas we hit (so you don't)

- **Multi-node `WebDataset` ValueError "need explicit nodesplitter"** — pass
  `nodesplitter=wds.split_by_node` to the `WebDataset` constructor. Single-node is silent about it;
  multi-node hard-fails. This was the one fix that got 16-GPU training to start.
- **`np.float` removed** — vanilla `torch2onnx.py` calls `img.astype(np.float)`, gone in NumPy ≥1.24.
  Patched to `np.float32`.
- **CUDA checkpoint on a CPU login node** — `torch.load(model.pt)` fails with
  *"Attempting to deserialize on a CUDA device…"*. Patched to `map_location="cpu"`.
- **`mxnet` not installed** — several `arcface_torch` files import it at module top level even when
  using WebDataset. Made optional via `try/except`.
- **Stale `~/.local/bin/torchrun`** pointing at a deleted interpreter — use
  `python -m torch.distributed.run` instead of the `torchrun` shim.
- **SLURM "Requested node configuration is not available"** — `--cpus-per-task` exceeded a node's
  allocatable cores; drop it to fit `CfgTRES`.
- **The custom loader is not a `DataLoader` subclass** — `train_v2.py` calls
  `sampler.set_epoch(...)`; the `_CudaLoader` wrapper exposes a no-op `reset()` and yields cuda
  tensors directly to avoid that path.

## License / data

Glint360K and InsightFace are released for non-commercial research use. Respect their licenses;
this folder only contains original config/loader/launcher code plus a small diff.
