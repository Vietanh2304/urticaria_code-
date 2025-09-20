import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

# Import tất cả các thành phần cần thiết từ file common.py
from common import (
    cfg, seed_everything, HAM10000Dataset, MMFNet, 
    train_tf, valid_tf, FocalLoss, 
    train_one_epoch, valid_one_epoch, preprocess_ham10000_metadata, preprocess_pad_metadata
)

def run_pretraining(cfg):
    print("\n" + "="*40 + "\n--- BƯỚC 1: TIỀN HUẤN LUYỆN TRÊN HAM10000 ---\n" + "="*40)
    wandb.init(project="skin-cancer-hpc", name="pretraining", config=vars(cfg))
    
    # --- LOGIC XỬ LÝ METADATA MỚI ---
    # 1. Tải cả 2 file metadata để lấy cấu trúc cột chuẩn
    pad_df_for_cols = pd.read_csv(cfg.CSV_FILE)
    _, target_meta_cols = preprocess_pad_metadata(pad_df_for_cols)
    meta_dim = len(target_meta_cols)
    print(f"Kiến trúc model sẽ được xây dựng với meta_dim = {meta_dim}")

    # 2. Tải và xử lý metadata của HAM10000
    ham_df = pd.read_csv(cfg.HAM10000_CSV)
    ham_meta_df_processed = preprocess_ham10000_metadata(ham_df, target_meta_cols)
    
    # --- Tiếp tục xử lý dữ liệu như cũ ---
    dx_to_unified = {'bcc': 'BCC', 'mel': 'MEL', 'nv': 'NEV', 'akiec': 'AK', 'bkl': 'BKL'}
    ham_df['unified_dx'] = ham_df['dx'].map(dx_to_unified)
    ham_df_filtered = ham_df.dropna(subset=['unified_dx']).copy()
    
    train_df, val_df = train_test_split(ham_df_filtered, test_size=0.2, random_state=cfg.SEED, stratify=ham_df_filtered['unified_dx'])
    
    # Lấy đúng metadata đã xử lý cho tập train và val
    train_meta_df = ham_meta_df_processed.loc[train_df.index]
    val_meta_df = ham_meta_df_processed.loc[val_df.index]
    
    # Khởi tạo Dataset với metadata
    train_ds = HAM10000Dataset(train_df, train_meta_df, cfg.HAM10000_IMG_DIRS, transform=train_tf)
    val_ds = HAM10000Dataset(val_df, val_meta_df, cfg.HAM10000_IMG_DIRS, transform=valid_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Khởi tạo model với meta_dim chính xác
    model = MMFNet(meta_dim=meta_dim, cfg=cfg, num_classes=cfg.NUM_CLASSES_PRETRAIN).to(cfg.DEVICE)
    criterion = FocalLoss(gamma=2) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_PRETRAIN, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg.EPOCHS_PRETRAIN):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_metrics = valid_one_epoch(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.EPOCHS_PRETRAIN} -> ...")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.PRETRAINED_MODEL_PATH)
            print(f"Val loss improved. Saving pre-trained model.")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"Early stopping."); break
    
    wandb.finish()
    print(f"\nPre-trained model saved to {cfg.PRETRAINED_MODEL_PATH}")

# Đoạn code để thực thi khi bạn chạy `python code/pretrain.py`
# if __name__ == "__main__":
#     seed_everything(cfg.SEED)
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     print(f"Thiết bị đang sử dụng: {cfg.DEVICE}")
    
#     if not os.path.exists(cfg.PRETRAINED_MODEL_PATH):
#         run_pretraining(cfg)
#     else:
#         print(f"Đã tìm thấy file pre-train tại: {cfg.PRETRAINED_MODEL_PATH}. Bỏ qua bước tiền huấn luyện.")
if __name__ == "__main__":
    # --- ĐOẠN CODE THÊM MỚI ĐỂ XÓA FILE CŨ ---
    if os.path.exists(cfg.PRETRAINED_MODEL_PATH):
        print(f"Phát hiện model pre-train cũ tại: {cfg.PRETRAINED_MODEL_PATH}")
        print("Đang xóa file để bắt đầu pre-train mới...")
        os.remove(cfg.PRETRAINED_MODEL_PATH)
        print("Đã xóa file cũ thành công.") 
    # ---------------------------------------------
    
    # Các lệnh khởi tạo khác
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Thiết bị đang sử dụng: {cfg.DEVICE}")
    
    # Luôn chạy pre-training vì file cũ (nếu có) đã bị xóa
    run_pretraining(cfg)
