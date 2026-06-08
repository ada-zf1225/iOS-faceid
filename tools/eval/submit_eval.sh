#!/bin/bash
# Optional: run the evaluation on a SLURM GPU node instead of a laptop CPU.
# Detection (RetinaPallet) on GPU is ~15× faster than CPU for the 6000-pair set.
# Requires onnxruntime-gpu with a working CUDAExecutionProvider in the env, plus
# scikit-learn / matplotlib / insightface / opencv-python-headless.
#SBATCH --job-name=faceid-eval
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate facetrain
export FACEID_CTX=0                              # 0 = GPU (needs onnxruntime CUDAExecutionProvider)
export FACEID_R50_ONNX=$HOME/r50_glint_robust.onnx
python evaluate.py --subset 10_folds             # full LFW 6000, 10-fold protocol
