"""
WebDataset loader for InsightFace arcface_torch, with robustness augmentation.

Drop this into `recognition/arcface_torch/` and apply `insightface_arcface_torch.patch`
so `dataset.py` routes to it when `config.rec` points at a directory of `.tar` shards.

The augmentation block is the core of this project's thesis: it deliberately
simulates the two failure modes that break naive face apps —
  • downscale→upscale  → far-away / small / low-res faces
  • random erasing     → glasses / partial occlusion
so the resulting embeddings stay stable across distance and eyewear.
"""
import os, glob, torch, webdataset as wds
from torchvision import transforms as T


def _build_transform():
    # Glint360K is already 112x112 aligned; add robustness aug on top.
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),                          # blur
        T.RandomApply([T.Resize(40, antialias=True), T.Resize(112, antialias=True)], p=0.25), # scale: far/small/low-res
        T.ColorJitter(brightness=0.3, contrast=0.3),                                          # lighting
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),                                        # (x-127.5)/127.5
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),                                           # occlusion / glasses proxy
    ])


def _to_label(c):
    if isinstance(c, (bytes, bytearray)):
        c = c.decode()
    return int(c)


class _CudaLoader:
    """Not a DataLoader subclass (so train_v2 won't call sampler.set_epoch);
    yields cuda tensors; reset() is a no-op."""
    def __init__(self, loader, local_rank):
        self.loader, self.local_rank = loader, local_rank

    def __iter__(self):
        for img, label in self.loader:
            yield (img.cuda(self.local_rank, non_blocking=True),
                   label.cuda(self.local_rank, non_blocking=True))

    def reset(self):
        pass


def get_wds_dataloader(root_dir, local_rank, batch_size, num_workers=8,
                       seed=2048, num_images=17091657):
    shards = sorted(glob.glob(os.path.join(root_dir, "*.tar*")))
    assert shards, f"no .tar shards under {root_dir}"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        rank, world = 0, 1
    tf = _build_transform()
    ds = (wds.WebDataset(shards, resampled=True, shardshuffle=1000, seed=seed + rank,
                         nodesplitter=wds.split_by_node,      # REQUIRED for multi-node, else ValueError
                         handler=wds.warn_and_continue)
            .shuffle(4000)
            .decode("pil", handler=wds.warn_and_continue)
            .to_tuple("jpg", "cls", handler=wds.warn_and_continue)
            .map_tuple(tf, _to_label))
    loader = wds.WebLoader(ds, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=True, drop_last=True)
    loader = loader.with_epoch(max(1, num_images // (batch_size * world)))
    return _CudaLoader(loader, local_rank)
