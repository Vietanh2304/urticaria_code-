# ===============================================================
# IMPORTS
# ===============================================================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import gc
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Import các thành phần cần thiết từ file common.py
from common import (
    cfg, seed_everything, PADUFESDataset, MMFNet, preprocess_pad_metadata,
    valid_tf, train_tf, FocalLoss, train_one_epoch, valid_one_epoch, 
    plot_training_history, compute_metrics, DATASET_MEAN, DATASET_STD
)

# ===============================================================
# HÀM CHÍNH CHO FINE-TUNING VÀ ĐÁNH GIÁ
# ===============================================================
def run_finetuning_and_evaluation(cfg):
    """
    Thực thi toàn bộ quá trình fine-tuning K-Fold và đánh giá cuối cùng trên tập test.
    """
    print("\n" + "="*40 + "\n--- BẮT ĐẦU GIAI ĐOẠN FINE-TUNING VÀ ĐÁNH GIÁ ---\n" + "="*40)
    
    # --- 1. Chuẩn bị dữ liệu và class weights ---
    df = pd.read_csv(cfg.CSV_FILE)
    meta_df, meta_cols = preprocess_pad_metadata(df)
    trainval_df, test_df = train_test_split(df, test_size=0.15, stratify=df["diagnostic"], random_state=cfg.SEED)
    trainval_meta, test_meta = meta_df.loc[trainval_df.index], meta_df.loc[test_df.index]

    scaler = StandardScaler()
    num_cols_to_scale = ["age", "diameter_1", "diameter_2", "lesion_area", "aspect_ratio"]
    trainval_meta_scaled = trainval_meta.copy()
    test_meta_scaled = test_meta.copy()
    trainval_meta_scaled[num_cols_to_scale] = scaler.fit_transform(trainval_meta[num_cols_to_scale])
    test_meta_scaled[num_cols_to_scale] = scaler.transform(test_meta[num_cols_to_scale])

    test_ds = PADUFESDataset(test_df.reset_index(drop=True), test_meta_scaled.reset_index(drop=True), cfg.IMG_ROOTS, transform=valid_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train/Validation data size: {len(trainval_df)}, Test data size: {len(test_df)}")

    # Lấy danh sách các lớp duy nhất
    class_labels = sorted(trainval_df['diagnostic'].unique())

# Chuyển đổi danh sách sang dạng numpy array trước khi đưa vào hàm
    class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.array(class_labels), # <-- THAY ĐỔI CHÍNH Ở ĐÂY
    y=trainval_df['diagnostic']
)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(cfg.DEVICE)
    print(f"Class Weights đã được tính toán: {class_weights_tensor.cpu().numpy().round(2)}")

    # --- 2. Vòng lặp K-Fold ---
    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    fold_val_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(trainval_df, trainval_df['diagnostic'])):
        print(f"\n{'='*30} FOLD {fold + 1}/{cfg.N_SPLITS} {'='*30}")
        wandb.init(project="skin-cancer-hpc", name=f"finetune-fold-{fold+1}", config=vars(cfg))

        train_fold_df, val_fold_df = trainval_df.iloc[train_idx], trainval_df.iloc[val_idx]
        train_fold_meta, val_fold_meta = trainval_meta_scaled.iloc[train_idx], trainval_meta_scaled.iloc[val_idx]
        train_ds_fold = PADUFESDataset(train_fold_df, train_fold_meta, cfg.IMG_ROOTS, transform=train_tf)
        val_ds_fold = PADUFESDataset(val_fold_df, val_fold_meta, cfg.IMG_ROOTS, transform=valid_tf)
        
        train_targets = train_fold_df['diagnostic'].map(train_ds_fold.label_map).values
        class_sample_count = np.array([np.sum(train_targets == i) for i in range(cfg.NUM_CLASSES_PAD)])
        weight = 1. / (class_sample_count + 1e-9)
        samples_weight = np.array([weight[t] for t in train_targets])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader_fold = DataLoader(train_ds_fold, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
        val_loader_fold = DataLoader(val_ds_fold, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = MMFNet(meta_dim=len(meta_cols), cfg=cfg, num_classes=cfg.NUM_CLASSES_PAD).to(cfg.DEVICE)
        print("Loading pre-trained weights...")
        pretrained_dict = torch.load(cfg.PRETRAINED_MODEL_PATH, map_location=cfg.DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and "classifier" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
        criterion = FocalLoss(gamma=2, weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_FINETUNE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = OneCycleLR(optimizer, max_lr=cfg.LR_FINETUNE, 
                               steps_per_epoch=len(train_loader_fold), epochs=cfg.EPOCHS_FINETUNE)
        
        best_val_bacc = 0.0
        patience_counter = 0
        history = defaultdict(list)
        best_metrics_for_fold = {}

        for epoch in range(cfg.EPOCHS_FINETUNE):
            train_loss, train_metrics = train_one_epoch(model, train_loader_fold, criterion, optimizer, cfg.DEVICE, scheduler)
            val_loss, val_metrics = valid_one_epoch(model, val_loader_fold, criterion, cfg.DEVICE)
            
            history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
            history['train_bacc'].append(train_metrics['bacc']); history['val_bacc'].append(val_metrics['bacc'])
            history['train_auc'].append(train_metrics['auc']); history['val_auc'].append(val_metrics['auc'])
            history['train_acc'].append(train_metrics['acc']); history['val_acc'].append(val_metrics['acc'])
        
            print(f"Epoch {epoch+1}/{cfg.EPOCHS_FINETUNE}")
            print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_metrics['acc']:.4f}, BAcc: {train_metrics['bacc']:.4f}, AUC: {train_metrics['auc']:.4f}" )
            print(f"  Valid -> Loss: {val_loss:.4f}, Acc: {val_metrics['acc']:.4f}, BAcc: {val_metrics['bacc']:.4f}, AUC: {val_metrics['auc']:.4f}")

            if val_metrics['bacc'] > best_val_bacc:
                best_val_bacc = val_metrics['bacc']
                patience_counter = 0
                print(f"  >> Val BAcc improved to {best_val_bacc:.4f}. Saving model...")
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"best_model_fold_{fold+1}.pth"))
                best_metrics_for_fold = val_metrics
                best_metrics_for_fold['loss'] = val_loss
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    print(f"  >> Early stopping at epoch {epoch+1}."); break
        
        wandb.finish()
        
        if best_metrics_for_fold:
            fold_val_metrics.append(best_metrics_for_fold)
            
        # plot_training_history(history, fold + 1) # Có thể comment dòng này đi để không bị treo khi chạy trên server
        
        del model, train_loader_fold, val_loader_fold, history, best_metrics_for_fold
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*30} K-FOLD TRAINING COMPLETE {'='*30}")

    # --- 3. Đánh giá cuối cùng trên tập Test (Ensembling + TTA) ---
    if not fold_val_metrics:
        print("Bỏ qua đánh giá cuối cùng vì training chưa hoàn tất.")
        return

    print("\n" + "="*40 + "\n--- ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST ---\n" + "="*40)

    tta_transforms = [
        A.Compose([A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE), A.Normalize(mean=DATASET_MEAN, std=DATASET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE), A.HorizontalFlip(p=1.0), A.Normalize(mean=DATASET_MEAN, std=DATASET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE), A.VerticalFlip(p=1.0), A.Normalize(mean=DATASET_MEAN, std=DATASET_STD), ToTensorV2()]),
    ]

    ensemble_models = []
    for fold in range(cfg.N_SPLITS):
        model = MMFNet(meta_dim=len(meta_cols), cfg=cfg, num_classes=cfg.NUM_CLASSES_PAD).to(cfg.DEVICE)
        model_path = os.path.join(cfg.OUTPUT_DIR, f"best_model_fold_{fold+1}.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
            model.eval()
            ensemble_models.append(model)
            print(f"Loaded model from fold {fold+1}")
        else:
            print(f"Warning: Model for fold {fold+1} not found. Skipping.")

    if not ensemble_models:
        print("No models were found to perform evaluation.")
        return

    final_probs = []
    final_labels = []
    with torch.no_grad():
        for imgs_tensor, metas, labels in tqdm(test_loader, desc="Ensembling + TTA on Test Set"):
            metas = metas.to(cfg.DEVICE)
            batch_tta_probs = []
            
            imgs_numpy = imgs_tensor.cpu().numpy().transpose(0, 2, 3, 1)

            for tta_tf in tta_transforms:
                tta_imgs_list = [tta_tf(image=img)['image'] for img in imgs_numpy]
                tta_imgs = torch.stack(tta_imgs_list).to(cfg.DEVICE)
                
                batch_ensemble_probs = [F.softmax(model(tta_imgs, metas), dim=1).cpu().numpy() for model in ensemble_models]
                avg_ensemble_prob = np.mean(batch_ensemble_probs, axis=0)
                batch_tta_probs.append(avg_ensemble_prob)

            avg_tta_prob = np.mean(batch_tta_probs, axis=0)
            final_probs.append(avg_tta_prob)
            final_labels.append(labels.numpy())

    final_probs = np.vstack(final_probs)
    final_labels = np.concatenate(final_labels)

    final_metrics = compute_metrics(final_labels, final_probs)
    print(f"\n--- ENSEMBLE + TTA RESULTS ON TEST SET ---")
    print(f"Test Accuracy:          {final_metrics['acc']:.4f}")
    print(f"Test Balanced Accuracy: {final_metrics['bacc']:.4f}")
    print(f"Test AUC:               {final_metrics['auc']:.4f}")

    y_pred_test = np.argmax(final_probs, axis=1)
    label_names = list(test_ds.label_map.keys())
    cm = confusion_matrix(final_labels, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Ensemble + TTA Confusion Matrix on Test Set")
    plt.xticks(rotation=45)
    
    cm_path = os.path.join(cfg.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    
    # Khởi tạo một lần chạy WandB cuối cùng để lưu kết quả tổng kết
    wandb.init(project="skin-cancer-hpc", name="final_evaluation", config=vars(cfg))
    wandb.log({
        "final_test_accuracy": final_metrics['acc'],
        "final_test_bacc": final_metrics['bacc'],
        "final_test_auc": final_metrics['auc'],
        "final_confusion_matrix": wandb.Image(cm_path)
    })
    wandb.finish()


# ===============================================================
# ĐOẠN CODE THỰC THI CHÍNH
# ===============================================================
if __name__ == "__main__":
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Thiết bị đang sử dụng: {cfg.DEVICE}")
    
    if not os.path.exists(cfg.PRETRAINED_MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file model đã pre-train tại {cfg.PRETRAINED_MODEL_PATH}")
        print("Vui lòng chạy pretrain.py trước.")
    else:
        run_finetuning_and_evaluation(cfg)