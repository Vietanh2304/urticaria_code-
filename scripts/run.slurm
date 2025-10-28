#!/bin/bash
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
# Tạm thời sửa lại để chạy thử
#SBATCH --time=00:30:00

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


# (Tùy chọn) Đặt API key nếu bạn không muốn login thủ công
 export fc9f7ae8d22da414e88b406057ff4d34f2ea3d7e


# Chạy file training chính
# Thêm -u để output không bị buffer (log ra ngay lập tức)
python -u train.py

echo "Job finished."