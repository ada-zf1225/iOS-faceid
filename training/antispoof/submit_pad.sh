#!/bin/bash
# Single-node anti-spoofing trainer. CelebA-Spoof (~420K train imgs), MobileNetV3,
# 4 epochs — fast on H200 (data is the bottleneck, hence many workers).
#SBATCH --job-name=pad-mbv3
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate facetrain
cd ~/antispoof
python train_pad.py --epochs 4 --bs 256 --workers 48 --out work_dirs/pad_mbv3
