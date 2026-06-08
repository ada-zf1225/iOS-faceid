from easydict import EasyDict as edict

# ResNet-50 ArcFace on Glint360K with robustness augmentation.
# Place under recognition/arcface_torch/configs/ ; train with:
#   python train_v2.py configs/glint360k_r50_robust
config = edict()
config.margin_list = (1.0, 0.0, 0.4)   # CosFace margin (Glint360K official)
config.network = "r50"
config.resume = False
config.output = "work_dirs/glint360k_r50_robust"
config.embedding_size = 512
config.sample_rate = 0.2               # PartialFC — 360K classes, sample 20% per step
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256                # per-GPU; effective = 256 * num_gpus
config.lr = 0.3                        # for effective batch ~4096 (16 GPUs)
config.verbose = 2000
config.dali = False
config.num_workers = 12

# Path to the Glint360K WebDataset shards (a directory of *.tar files).
# See training/README.md for how to build these from the .rec/.idx release.
config.rec = "/path/to/glint360k-wds"

config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 16
config.warmup_epoch = 1
config.val_targets = []                # wds has no lfw/cfp/agedb .bin; evaluate separately (see tools/eval/)
