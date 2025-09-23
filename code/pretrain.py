# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# <<< THÊM MỚI: Import thư viện cho AMP
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

# Import tất cả các thành phần cần thiết từ file common.py
from common import (
    cfg, seed_everything, HAM10000Dataset, MMFNet,
    train_tf, valid_tf, FocalLoss,
    preprocess_ham10000_metadata, preprocess_pad_metadata
)

def run_pretraining(cfg):
    print("\n" + "="*40 + "\n--- BƯỚC 1: TIỀN HUẤN LUYỆN TRÊN HAM10000 ---\n" + "="*40)
    wandb.init(project="skin-cancer-hpc", name="pretraining_improved", config=vars(cfg))

    # --- LOGIC XỬ LÝ METADATA ---
    pad_df_for_cols = pd.read_csv(cfg.CSV_FILE)
    _, target_meta_cols = preprocess_pad_metadata(pad_df_for_cols)
    meta_dim = len(target_meta_cols)
    print(f"Kiến trúc model sẽ được xây dựng với meta_dim = {meta_dim}")

    ham_df = pd.read_csv(cfg.HAM10000_CSV)
    ham_meta_df_processed = preprocess_ham10000_metadata(ham_df, target_meta_cols)

    dx_to_unified = {'bcc': 'BCC', 'mel': 'MEL', 'nv': 'NEV', 'akiec': 'AK', 'bkl': 'BKL'}
    ham_df['unified_dx'] = ham_df['dx'].map(dx_to_unified)
    ham_df_filtered = ham_df.dropna(subset=['unified_dx']).copy()

    train_df, val_df = train_test_split(ham_df_filtered, test_size=0.2, random_state=cfg.SEED, stratify=ham_df_filtered['unified_dx'])

    train_meta_df = ham_meta_df_processed.loc[train_df.index]
    val_meta_df = ham_meta_df_processed.loc[val_df.index]

    # <<< THÊM MỚI: TÍNH TOÁN CLASS WEIGHTS (QUAN TRỌNG NHẤT) >>>
    print("Tính toán class weights cho tập train HAM10000...")
    train_labels = train_df['unified_dx']
    # Lấy class names theo đúng thứ tự label map trong class Dataset
    temp_ds = HAM10000Dataset(train_df.head(1), train_meta_df.head(1), cfg.HAM10000_IMG_DIRS)
    class_names = sorted(temp_ds.unified_label_map, key=temp_ds.unified_label_map.get)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(class_names),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(cfg.DEVICE)
    print(f"Class weights cho pre-training: {class_weights_tensor.cpu().numpy().round(2)}")
    
    # Khởi tạo Dataset
    train_ds = HAM10000Dataset(train_df, train_meta_df, cfg.HAM10000_IMG_DIRS, transform=train_tf)
    val_ds = HAM10000Dataset(val_df, val_meta_df, cfg.HAM10000_IMG_DIRS, transform=valid_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    # Khởi tạo model, loss, optimizer
    model = MMFNet(meta_dim=meta_dim, cfg=cfg, num_classes=cfg.NUM_CLASSES_PRETRAIN).to(cfg.DEVICE)
    # <<< THAY ĐỔI: Áp dụng class weights vào FocalLoss >>>
    criterion = FocalLoss(gamma=2, weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_PRETRAIN, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # <<< THÊM MỚI: Khởi tạo GradScaler cho AMP và Gradient Accumulation >>>
    scaler = GradScaler()
    accumulation_steps = 4  # Effective batch size = 16 * 4 = 64
    
    best_val_loss = float('inf')
    patience_counter = 0

    # Vòng lặp training chính
    for epoch in range(cfg.EPOCHS_PRETRAIN):
        # --- TRAINING EPOCH ---
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad() # Xóa grad ở đầu epoch
        
        for i, (imgs, metas, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
            imgs, metas, labels = imgs.to(cfg.DEVICE), metas.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            with autocast():
                logits = model(imgs, metas)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps # Chuẩn hóa loss cho accumulation
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps * imgs.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        wandb.log({"pretrain_train_loss": avg_train_loss, "epoch": epoch})

        # --- VALIDATION EPOCH ---
        model.eval()
        total_val_loss = 0.0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, metas, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Valid"):
                imgs, metas, labels = imgs.to(cfg.DEVICE), metas.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                with autocast():
                    logits = model(imgs, metas)
                    loss = criterion(logits, labels)
                total_val_loss += loss.item() * imgs.size(0)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        
        # Tính toán metrics
        all_labels = np.concatenate(all_labels)
        all_probs = np.vstack(all_probs)
        y_pred = np.argmax(all_probs, axis=1)
        bacc = balanced_accuracy_score(all_labels, y_pred)
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

        wandb.log({
            "pretrain_val_loss": avg_val_loss, 
            "pretrain_val_bacc": bacc,
            "pretrain_val_auc": auc,
            "epoch": epoch
        })

        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS_PRETRAIN}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Valid Loss: {avg_val_loss:.4f} | BAcc: {bacc:.4f} | AUC: {auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.PRETRAINED_MODEL_PATH)
            print(f"  >> Val loss improved. Saving pre-trained model to {cfg.PRETRAINED_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"  >> Early stopping at epoch {epoch+1}.")
                break
    
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
    if os.path.exists(cfg.PRETRAINED_MODEL_PATH):
        print(f"Phát hiện model pre-train cũ tại: {cfg.PRETRAINED_MODEL_PATH}")
        print("Đang xóa file để bắt đầu pre-train mới...")
        os.remove(cfg.PRETRAINED_MODEL_PATH)
        print("Đã xóa file cũ thành công.")
    
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Thiết bị đang sử dụng: {cfg.DEVICE}")
    
    run_pretraining(cfg)