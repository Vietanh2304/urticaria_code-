#!/bin/bash

# -- CẤU HÌNH YÊU CẦU TÀI NGUYÊN --
# Đặt tên cho tác vụ để dễ nhận biết
#SBATCH --job-name=urticaria_finetune

# File để ghi output và lỗi
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# Yêu cầu tài nguyên
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# Dựa trên tính toán (khoảng 7 ngày), yêu cầu 7 ngày
# CẢNH BÁO: Thời gian này có thể vượt quá giới hạn của hệ thống
#SBATCH --time=7-00:00:00

# -- CÁC LỆNH SẼ ĐƯỢC THỰC THI --
echo "--- Bắt đầu tác vụ FINE-TUNING trên node: $(hostname) lúc $(date) ---"

mkdir -p logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate urticaria_env
python code/finetune.py

echo "--- Tác vụ FINE-TUNING hoàn thành lúc $(date) ---"