#!/bin/bash

# Kaggle T4 Multi-GPU Launch Script
# Usage: bash launch_kaggle.sh

# 1. Set environment variables for PyTorch DDP to avoid CPU contention
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 2. Launch with accelerate
# --multi_gpu: Enable multi-GPU
# --mixed_precision=fp16: Essential for T4s (speed + memory)
# --num_processes=2: For 2x T4s
# --dynamo_backend=no: T4s don't support torch.compile well
echo "🚀 Launching Training in Full Security Mode..."
accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_processes=2 \
    src/training/train.py \
    --phase full \
    --save checkpoints/model.pth
