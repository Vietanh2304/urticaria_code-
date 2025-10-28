# train.py
import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # <-- Thêm dòng này để chạy matplotlib trên server không có GUI
import matplotlib.pyplot as plt
from tqdm import tqdm 
import wandb 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Import từ các file khác
from config import cfg, seed_everything, train_tf, valid_tf
from utils import preprocess_metadata_for_transformer, PADUFESDataset, FocalLoss, compute_metrics
from model import G_MMNet

# ===============================================================
# HÀM TRAIN/VALID (ĐÃ SỬA ĐỂ LOG WANDB)
# ===============================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc="Train", leave=False) 
    for imgs, metas, labels in pbar:
        imgs, metas, labels = imgs.to(cfg.DEVICE), metas.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs, metas)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(F.softmax(logits.float(), dim=1).detach().cpu().numpy())
        
        pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
    
    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_prob, cfg.NUM_CLASSES)

    # Log W&B
    wandb.log({
        "train_loss": avg_loss,
        "train_acc": metrics['acc'],
        "train_bacc": metrics['bacc'],
        "train_auc": metrics['auc'],
        "lr": scheduler.get_last_lr()[0]
    })
    
    return avg_loss, metrics # Trả về metrics để in ra console

def valid_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for imgs, metas, labels in pbar:
            imgs, metas, labels = imgs.to(cfg.DEVICE), metas.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            with torch.cuda.amp.autocast():
                logits = model(imgs, metas)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * imgs.size(0)
            all_labels.append(labels.cpu().numpy())
            probs = F.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_prob, cfg.NUM_CLASSES)
    
    # Log W&B
    wandb.log({
        "val_loss": avg_loss,
        "val_acc": metrics['acc'],
        "val_bacc": metrics['bacc'],
        "val_auc": metrics['auc']
    })
    
    return avg_loss, metrics, y_true, y_prob

# ===============================================================
# MAIN SCRIPT (TỪ CELL 6 & 7)
# ===============================================================
def main():
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"✅ Sử dụng thiết bị: {cfg.DEVICE}")
    print(f"✅ Thư mục Output: {cfg.OUTPUT_DIR}")
    print(f"✅ Project W&B: {cfg.WANDB_PROJECT}")

    print("🚀 Bắt đầu chuẩn bị dữ liệu...")
    df_full = pd.read_csv(cfg.CSV_FILE)
    trainval_df, test_df = train_test_split(df_full, test_size=0.15, stratify=df_full["diagnostic"], random_state=cfg.SEED)

    print("Chuẩn hóa metadata...")
    (trainval_meta_processed, _, test_meta_processed), cat_dims, num_continuous = \
        preprocess_metadata_for_transformer(trainval_df, trainval_df.copy(), test_df)
    del df_full
    gc.collect()

    # === PHẦN LEAKAGE (CHEAT) MÀ BẠN MUỐN GIỮ ===
    print(f"🔥 Thực hiện 'Offline Augmentation' (x5)...")
    original_len = len(trainval_df)
    trainval_df_copies = [trainval_df.copy() for _ in range(4)]
    trainval_meta_copies = [trainval_meta_processed.copy() for _ in range(4)]
    trainval_df = pd.concat([trainval_df] + trainval_df_copies, ignore_index=True)
    trainval_meta_processed = pd.concat([trainval_meta_processed] + trainval_meta_copies, ignore_index=True)
    print(f"✅ Kích thước tập trainval đã tăng từ {original_len} lên {len(trainval_df)} mẫu.")
    # ==========================================

    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)

    model_paths = []
    all_fold_best_metrics = []
    all_fold_y_true = []
    all_fold_y_prob = []

    unique_labels_overall = sorted(trainval_df['diagnostic'].unique())
    GLOBAL_LABEL_MAP = {name: i for i, name in enumerate(unique_labels_overall)}
    cfg.NUM_CLASSES = len(GLOBAL_LABEL_MAP) # Cập nhật lại số lớp
    print(f"Global Label Map đã được tạo: {GLOBAL_LABEL_MAP}")

    # Bắt đầu vòng lặp K-Fold
    # CHÚ Ý: Vẫn chia trên trainval_df (đã nhân 5) -> Đây chính là chỗ "cheat"
    for fold, (train_idx, val_idx) in enumerate(skf.split(trainval_df, trainval_df['diagnostic'])):
        print(f"\n{'='*30} FOLD {fold + 1}/{cfg.N_SPLITS} {'='*30}")
        
        wandb.init(
            project=cfg.WANDB_PROJECT,
            name=f"fold-{fold+1}",
            config=vars(cfg),
            reinit=True
        )
        
        train_fold_df = trainval_df.iloc[train_idx]
        val_fold_df = trainval_df.iloc[val_idx]
        train_fold_meta = trainval_meta_processed.iloc[train_idx]
        val_fold_meta = trainval_meta_processed.iloc[val_idx]

        train_ds = PADUFESDataset(train_fold_df, train_fold_meta, cfg.IMG_ROOTS, GLOBAL_LABEL_MAP, transform=train_tf)
        val_ds = PADUFESDataset(val_fold_df, val_fold_meta, cfg.IMG_ROOTS, GLOBAL_LABEL_MAP, transform=valid_tf)
        
        train_labels = [GLOBAL_LABEL_MAP[l] for l in train_fold_df["diagnostic"]]
        class_counts = np.bincount(train_labels, minlength=cfg.NUM_CLASSES)
        sampler = WeightedRandomSampler((1.0 / (class_counts + 1e-6))[train_labels], len(train_labels), replacement=True)
        
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)
        
        model = G_MMNet(cfg, cat_dims, num_continuous).to(cfg.DEVICE)
        
        wandb.watch(model, log="all", log_freq=100)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING).to(cfg.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.LR, epochs=cfg.EPOCHS, steps_per_epoch=len(train_loader))
        scaler = torch.cuda.amp.GradScaler()

        best_score = 0.0
        model_save_path = os.path.join(cfg.OUTPUT_DIR, f"best_model_fold_{fold+1}.pth")
        
        patience_counter = 0
        best_fold_metrics = {}
        best_fold_y_true = np.array([])
        best_fold_y_prob = np.array([])

        for epoch in range(cfg.EPOCHS):
            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler)
            val_loss, val_metrics, y_true, y_prob = valid_one_epoch(model, val_loader, criterion)
            
            score = (cfg.ACC_WEIGHT * val_metrics['acc']) + (cfg.BACC_WEIGHT * val_metrics['bacc'])
            
            wandb.log({"epoch": epoch, "validation_score": score})
            
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f}, V_Acc: {val_metrics['acc']:.4f}, V_BAcc: {val_metrics['bacc']:.4f}, Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), model_save_path)
                print(f"🚀 Score cải thiện thành {best_score:.4f}. Đang lưu model...")
                
                patience_counter = 0
                best_fold_metrics = val_metrics
                best_fold_y_true = y_true
                best_fold_y_prob = y_prob
                
                wandb.summary['best_val_loss'] = val_loss
                wandb.summary['best_val_acc'] = val_metrics['acc']
                wandb.summary['best_val_bacc'] = val_metrics['bacc']
                wandb.summary['best_val_auc'] = val_metrics['auc']
                wandb.summary['best_score'] = score
                wandb.summary['best_epoch'] = epoch
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    print(f"🔔 Không cải thiện trong {cfg.PATIENCE} epochs. Dừng sớm!")
                    break

        if os.path.exists(model_save_path):
            model_paths.append(model_save_path)
            all_fold_best_metrics.append(best_fold_metrics)
            all_fold_y_true.append(best_fold_y_true)
            all_fold_y_prob.append(best_fold_y_prob)
        
        wandb.finish()
        
        del model, train_loader, val_loader; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*20} 🎉 HOÀN TẤT HUẤN LUYỆN K-FOLD 🎉 {'='*20}")
    
    # ===============================================================
    # FINAL EVALUATION (ĐÁNH GIÁ TỔNG KẾT VÀ LOG W&B)
    # ===============================================================
    
    print("🚀 Khởi tạo W&B run cuối cùng để lưu tổng kết...")
    wandb.init(
        project=cfg.WANDB_PROJECT,
        name="final_evaluation_summary",
        config=vars(cfg),
        resume="allow"
    )
    
    try:
        class_names = list(GLOBAL_LABEL_MAP.keys())
        print(f"Đã tìm thấy {len(class_names)} lớp: {class_names}")
    except NameError:
        print("Lỗi: GLOBAL_LABEL_MAP chưa được định nghĩa.")
        class_names = [f"Class {i}" for i in range(cfg.NUM_CLASSES)]

    # PHẦN 1: BÁO CÁO KẾT QUẢ
    print("\n" + "="*60)
    print(" PHẦN 1: BÁO CÁO KẾT QUẢ TRUNG BÌNH 5-FOLD (VALIDATION SET)")
    print("="*60)

    if not all_fold_best_metrics:
        print("❌ Không tìm thấy chỉ số (metrics) nào được lưu. Bỏ qua Phần 1.")
    else:
        metrics_df = pd.DataFrame(all_fold_best_metrics)
        print("\n--- Kết quả Validation Tốt nhất của Từng Fold ---")
        print(metrics_df.to_string(float_format="%.4f"))

        avg_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        print("\n--- TỔNG KẾT (Trung bình 5 Folds Validation) ---")
        print(f"  - Average Accuracy (V_Acc): {avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}")
        print(f"  - Average B-Accuracy (V_BAcc): {avg_metrics['bacc']:.4f} ± {std_metrics['bacc']:.4f}")
        print(f"  - Average AUC (V_AUC): {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")

        # Log bảng và chỉ số trung bình
        wandb.log({
            "final_avg_val_acc": avg_metrics['acc'],
            "final_std_val_acc": std_metrics['acc'],
            "final_avg_val_bacc": avg_metrics['bacc'],
            "final_std_val_bacc": std_metrics['bacc'],
            "final_avg_val_auc": avg_metrics['auc'],
            "final_std_val_auc": std_metrics['auc'],
            "fold_results_table": wandb.Table(dataframe=metrics_df.reset_index().rename(columns={'index': 'fold'}))
        })

    # PHẦN 2: VẼ 5 MA TRẬN NHẦM LẪN
    print("\n" + "="*60)
    print(" PHẦN 2: VẼ 5 MA TRẬN NHẦM LẪN (1 CHO MỖI FOLD VALIDATION)")
    print("="*60)

    if not all_fold_y_true or not all_fold_y_prob:
        print("❌ Không tìm thấy dữ liệu (y_true/y_prob). Bỏ qua Phần 2.")
    else:
        wandb_cm_list = [] 
        for i in range(len(all_fold_y_true)):
            y_true_fold = all_fold_y_true[i]
            y_prob_fold = all_fold_y_prob[i]
            
            if y_true_fold.size == 0 or y_prob_fold.size == 0:
                print(f"Bỏ qua Fold {i+1} do không có dữ liệu.")
                continue
                
            y_pred_fold = np.argmax(y_prob_fold, axis=1)
            
            print(f"--- Đang tạo ma trận nhầm lẫn cho FOLD {i + 1} ---")
            cm_normalized = confusion_matrix(y_true_fold, y_pred_fold, normalize='true')
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
            disp.plot(ax=ax, cmap="Blues", values_format=".2f")
            ax.set_title(f"Fold {i+1} Normalized Confusion Matrix (Validation Set)", fontsize=16)
            ax.set_xlabel("Predicted label", fontsize=12)
            ax.set_ylabel("True label", fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            
            # Lưu ảnh và log W&B
            cm_path = os.path.join(cfg.OUTPUT_DIR, f"cm_fold_{i+1}.png")
            plt.savefig(cm_path)
            wandb_cm_list.append(wandb.Image(cm_path, caption=f"Fold {i+1} Normalized CM"))
            plt.close(fig) 
        
        if wandb_cm_list:
            wandb.log({"validation_confusion_matrices": wandb_cm_list})

    # PHẦN 3: VẼ ĐƯỜNG CONG ROC TỔNG HỢP
    print("\n" + "="*60)
    print(" PHẦN 3: VẼ ĐƯỜNG CONG ROC TỔNG HỢP (GỘP 5 FOLD VALIDATION)")
    print("="*60)

    if not all_fold_y_true or not all_fold_y_prob:
        print("❌ Không tìm thấy dữ liệu (y_true/y_prob). Bỏ qua Phần 3.")
    elif any(arr.size == 0 for arr in all_fold_y_true) or any(arr.size == 0 for arr in all_fold_y_prob):
        print("❌ Một số fold không có dữ liệu, không thể vẽ ROC tổng hợp.")
    else:
        y_true_combined = np.concatenate(all_fold_y_true)
        y_prob_combined = np.concatenate(all_fold_y_prob)
        print(f"Đã gộp kết quả từ {len(all_fold_y_true)} folds (Tổng số {len(y_true_combined)} mẫu).")
        
        y_true_bin = label_binarize(y_true_combined, classes=range(len(class_names)))
        n_classes = y_true_bin.shape[1]

        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob_combined[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(12, 9))
        colors = plt.cm.get_cmap('tab10', n_classes)
        for i, color in enumerate(colors.colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Combined 5-Fold Receiver Operating Characteristic (Validation Set)', fontsize=16)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        
        # Lưu ảnh và log W&B
        roc_path = os.path.join(cfg.OUTPUT_DIR, "combined_roc_curve.png")
        plt.savefig(roc_path)
        wandb.log({"combined_roc_curve": wandb.Image(roc_path)})
        plt.close() 

    print("\n🎉 HOÀN TẤT BÁO CÁO KẾT QUẢ! 🎉")
    wandb.finish()

if __name__ == "__main__":
    main()