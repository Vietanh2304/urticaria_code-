#!/bin/bash

# -- CẤU HÌNH YÊU CẦU TÀI NGUYÊN --
# Đặt tên cho tác vụ để dễ nhận biết
#SBATCH --job-name=urticaria_pretrain

# File để ghi output và lỗi
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

# Yêu cầu tài nguyên
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# Dựa trên tính toán (khoảng 29 tiếng), yêu cầu 2 ngày là an toàn
#SBATCH --time=2-00:00:00

# -- CÁC LỆNH SẼ ĐƯỢC THỰC THI --
echo "--- Bắt đầu tác vụ PRE-TRAINING trên node: $(hostname) lúc $(date) ---"

mkdir -p logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate urticaria_env
python code/pretrain.py

echo "--- Tác vụ PRE-TRAINING hoàn thành lúc $(date) ---"