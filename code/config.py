# config.py
import os
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    # Đường dẫn & Thư mục (Sửa đổi cho phù hợp với server của bạn)
    BASE_DIR = "./data/skin-cancer" # <-- CẬP NHẬT ĐƯỜNG DẪN NÀY
    CSV_FILE = os.path.join(BASE_DIR, "metadata.csv")
    IMG_ROOTS = [
        os.path.join(BASE_DIR, "imgs_part_1/imgs_part_1"),
        os.path.join(BASE_DIR, "imgs_part_2/imgs_part_2"),
        os.path.join(BASE_DIR, "imgs_part_3/imgs_part_3"),
    ]
    OUTPUT_DIR = "./output_skin_class_11" # <-- Thư mục output

    # Môi trường chạy
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    IMG_SIZE = 224

    # Siêu tham số của mô hình (G_MMNet)
    NUM_CLASSES = 6
    IMG_ENCODER_EMBED_DIM_S1 = 192
    IMG_ENCODER_EMBED_DIM_S2 = 768
    IMG_ENCODER_FINAL_DIM = 696
    META_DEPTH = 2
    META_HEADS = 12
    CO_ATTENTION_DEPTH = 2
    CO_ATTENTION_HEADS = 12
    DROPOUT = 0.1

    # Siêu tham số huấn luyện
    N_SPLITS = 5
    BATCH_SIZE = 16 
    EPOCHS = 80
    LR = 1e-4
    WEIGHT_DECAY = 4e-2
    PATIENCE = 10
    WARMUP_EPOCHS = 20
    LABEL_SMOOTHING = 0.15

    # Chỉ số để lưu model
    ACC_WEIGHT = 0.7
    BACC_WEIGHT = 0.3
    
    # Cấu hình W&B
    WANDB_PROJECT = "skin-class-11-slurm"

cfg = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===============================================================
# DATA AUGMENTATION
# ===============================================================
train_tf = A.Compose([
    A.RandomResizedCrop(size=(cfg.IMG_SIZE, cfg.IMG_SIZE), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        translate_percent=0.0625,
        scale=(0.9, 1.1),
        rotate=(-20, 20),
        p=0.7
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.CoarseDropout(
        num_holes=8, 
        max_h_size=cfg.IMG_SIZE // 10, 
        max_w_size=cfg.IMG_SIZE // 10, 
        p=0.5
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

valid_tf = A.Compose([
    A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])