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
    train_one_epoch, valid_one_epoch
)

def run_pretraining(cfg):
    """
    Thực thi toàn bộ quá trình tiền huấn luyện trên bộ dataset HAM10000.
    """
    print("\n" + "="*40 + "\n--- BẮT ĐẦU GIAI ĐOẠN TIỀN HUẤN LUYỆN ---\n" + "="*40)

    # Khởi tạo WandB cho quá trình pre-training
    wandb.init(project="skin-cancer-hpc", name="pretraining", config=vars(cfg))
    
    ham_df = pd.read_csv(cfg.HAM10000_CSV)
    dx_to_unified = {'bcc': 'BCC', 'mel': 'MEL', 'nv': 'NEV', 'akiec': 'AK', 'bkl': 'BKL'}
    ham_df['unified_dx'] = ham_df['dx'].map(dx_to_unified)
    ham_df_filtered = ham_df.dropna(subset=['unified_dx']).copy()
    print(f"Sử dụng {len(ham_df_filtered)}/{len(ham_df)} mẫu từ HAM10000 để pre-train trên {cfg.NUM_CLASSES_PRETRAIN} lớp chung.")

    train_df, val_df = train_test_split(ham_df_filtered, test_size=0.2, random_state=cfg.SEED, stratify=ham_df_filtered['unified_dx'])
    train_ds = HAM10000Dataset(train_df, cfg.HAM10000_IMG_DIRS, transform=train_tf)
    val_ds = HAM10000Dataset(val_df, cfg.HAM10000_IMG_DIRS, transform=valid_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # meta_dim=1 là một giá trị giữ chỗ vì HAM10000 không dùng metadata
    model = MMFNet(meta_dim=1, cfg=cfg, num_classes=cfg.NUM_CLASSES_PRETRAIN).to(cfg.DEVICE)
    criterion = FocalLoss(gamma=2) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_PRETRAIN, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg.EPOCHS_PRETRAIN):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_metrics = valid_one_epoch(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg.EPOCHS_PRETRAIN} -> Train Loss: {train_loss:.4f}, Train BAcc: {train_metrics['bacc']:.4f} | Val Loss: {val_loss:.4f}, Val BAcc: {val_metrics['bacc']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.PRETRAINED_MODEL_PATH)
            print(f"Val loss improved to {best_val_loss:.4f}. Saving pre-trained model.")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}."); break

    # Kết thúc lần chạy WandB
    wandb.finish()
    print(f"\nPre-trained model saved to {cfg.PRETRAINED_MODEL_PATH}")

# Đoạn code để thực thi khi bạn chạy `python code/pretrain.py`
if __name__ == "__main__":
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Thiết bị đang sử dụng: {cfg.DEVICE}")
    
    if not os.path.exists(cfg.PRETRAINED_MODEL_PATH):
        run_pretraining(cfg)
    else:
        print(f"Đã tìm thấy file pre-train tại: {cfg.PRETRAINED_MODEL_PATH}. Bỏ qua bước tiền huấn luyện.")