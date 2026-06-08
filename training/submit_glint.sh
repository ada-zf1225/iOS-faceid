#!/bin/bash
# Multi-node ArcFace training launcher (SLURM + torchrun c10d rendezvous).
# Trained on 4 nodes × 4× NVIDIA H200 (16 GPUs) → ~37k img/s → ~2 h wall-clock.
# Adjust --account / --partition / --gres to your cluster.
#SBATCH --job-name=glint-r50-robust
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=4
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56            # keep ≤ allocatable cores per node
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=12399
echo "nodes=$SLURM_NNODES master=$MASTER_ADDR"

# one task per node; torchrun spawns 4 ranks (one per GPU) under each
srun --ntasks-per-node=1 bash -lc '
source ~/miniforge3/etc/profile.d/conda.sh
conda activate facetrain
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
cd ~/insightface/recognition/arcface_torch
python -m torch.distributed.run \
  --nnodes=$SLURM_NNODES --nproc-per-node=4 \
  --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_v2.py configs/glint360k_r50_robust
'
