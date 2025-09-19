#!/bin/bash

# ===============================================================
# CẤU HÌNH YÊU CẦU TÀI NGUYÊN CHO SLURM
# ===============================================================
# Đặt tên cho tác vụ để dễ nhận biết trong hàng đợi
#SBATCH --job-name=urticaria_training

# File để ghi output và lỗi (sử dụng %j là Job ID để mỗi lần chạy có file riêng)
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Yêu cầu tài nguyên: 1 node, 1 GPU, chạy trong 8 tiếng
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# ===============================================================
# CÁC LỆNH SẼ ĐƯỢC THỰC THI
# ===============================================================

# In ra thông tin bắt đầu để dễ theo dõi trong file log
echo "--- Bắt đầu tác vụ trên node: $(hostname) lúc $(date) ---"

# Tự động tạo thư mục logs nếu nó chưa tồn tại (an toàn)
mkdir -p logs

# 1. Kích hoạt trình quản lý môi trường Conda
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Kích hoạt môi trường Python đã được chuẩn bị sẵn
#    (Hãy chắc chắn tên này khớp với môi trường bạn đã tạo)
conda activate urticaria_env

echo "--- Đã kích hoạt môi trường, bắt đầu chạy script Python ---"

# 3. Chạy file Python huấn luyện của bạn
#    (Giả định lệnh sbatch được chạy từ thư mục gốc của dự án)
python code/main.py

echo "--- Tác vụ hoàn thành lúc $(date) ---"