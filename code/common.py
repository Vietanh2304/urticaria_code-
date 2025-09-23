import argparse
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict
import gc
import cv2
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Configuration Class ---
class Config:
    # Paths
    BASE_DIR = "./data/Skin_cancer"
    CSV_FILE = os.path.join(BASE_DIR, "metadata.csv")
    IMG_ROOTS = [
        os.path.join(BASE_DIR, "imgs_part_1/imgs_part_1"),
        os.path.join(BASE_DIR, "imgs_part_2/imgs_part_2"),
        os.path.join(BASE_DIR, "imgs_part_3/imgs_part_3"),
    ]
    
    HAM10000_BASE_DIR = "./data/skin-cancer-mnist-ham10000"
    HAM10000_CSV = os.path.join(HAM10000_BASE_DIR, "HAM10000_metadata.csv")
    HAM10000_IMG_DIRS = [
        os.path.join(HAM10000_BASE_DIR, "HAM10000_images_part_1"),
        os.path.join(HAM10000_BASE_DIR, "HAM10000_images_part_2")
    ]
    
    OUTPUT_DIR = "./output"
    PRETRAINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "pretrained_ham_model_5_classes.pth")

    # Runtime
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # Model & Data
    IMG_SIZE = 224
    NUM_CLASSES_PAD = 6
    NUM_CLASSES_PRETRAIN = 5
    EMBED_DIM = 512
    
    # Training
    N_SPLITS = 5
    BATCH_SIZE = 16
    EPOCHS_PRETRAIN = 100
    EPOCHS_FINETUNE = 100
    # <<< THAY ĐỔI 1: Giảm learning rate cho giai đoạn fine-tune chính
    LR_FINETUNE = 5e-6 
    LR_PRETRAIN = 3e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 15
    # <<< THAY ĐỔI 2: Giảm Dropout Rate
    DROPOUT_RATE = 0.2

# Khởi tạo Config
cfg = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(f"Device: {cfg.DEVICE}")


# ===============================================================
# DATASET & PREPROCESSING
# ===============================================================
def remove_hair(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_image

def preprocess_ham10000_metadata(df, target_meta_cols):
    df = df.copy()
    df['age'].fillna(df['age'].median(), inplace=True)
    df['sex'].fillna('unknown', inplace=True)
    df = pd.get_dummies(df, columns=['sex', 'localization'], prefix=['sex', 'loc'])
    df_aligned = df.reindex(columns=target_meta_cols, fill_value=0)
    return df_aligned

def preprocess_pad_metadata(df):
    df = df.copy()
    df['diameter_1'] = pd.to_numeric(df['diameter_1'], errors='coerce')
    df['diameter_2'] = pd.to_numeric(df['diameter_2'], errors='coerce')
    df.fillna({'diameter_1': df['diameter_1'].median(), 'diameter_2': df['diameter_2'].median()}, inplace=True)
    df['lesion_area'] = df['diameter_1'] * df['diameter_2']
    df['aspect_ratio'] = df['diameter_1'] / (df['diameter_2'] + 1e-6)
    
    num_cols = ["age", "diameter_1", "diameter_2", "lesion_area", "aspect_ratio"]
    df['age'] = pd.to_numeric(df['age'], errors="coerce").fillna(df['age'].median())

    bool_cols = ["smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history", "itch", "grew", "hurt", "changed", "bleed", "biopsed"]
    for c in bool_cols:
        if c in df.columns: df[c] = df[c].astype(str).str.lower().map({"yes":1, "no":0}).fillna(0).astype(int)
    
    cat_cols = ["gender", "region", "fitspatrick"]
    df_meta = pd.get_dummies(df[num_cols + bool_cols + cat_cols], columns=[c for c in cat_cols if c in df.columns])
    return df_meta, list(df_meta.columns)

class PADUFESDataset(Dataset):
    def __init__(self, df, meta_df, img_roots, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta_df = meta_df.reset_index(drop=True)
        self.transform = transform
        self.img_roots = img_roots
        self.meta_arr = self.meta_df.values.astype("float32")
        self.label_map = {"ACK":0, "BCC":1, "MEL":2, "NEV":3, "SCC":4, "SEK":5}
        
    def __len__(self): return len(self.df)
    def find_image_path(self, img_id):
        for root in self.img_roots:
            path = os.path.join(root, img_id)
            if os.path.exists(path): return path
        raise FileNotFoundError(f"Image {img_id} not found.")
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.find_image_path(str(row["img_id"]))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = remove_hair(img)
        if self.transform: img = self.transform(image=img)['image']
        meta = torch.tensor(self.meta_arr[idx])
        label = torch.tensor(self.label_map[row["diagnostic"]], dtype=torch.long)
        return img, meta, label

class HAM10000Dataset(Dataset):
    def __init__(self, df, meta_df, img_dirs, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta_df = meta_df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform
        self.unified_label_map = {"BCC": 0, "MEL": 1, "NEV": 2, "AK": 3, "BKL": 4}
        self.meta_arr = self.meta_df.values.astype("float32")

    def __len__(self): return len(self.df)
    def find_image_path(self, img_id):
        for img_dir in self.img_dirs:
            path = os.path.join(img_dir, img_id + ".jpg")
            if os.path.exists(path): return path
        raise FileNotFoundError(f"Image {img_id}.jpg not found.")
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.find_image_path(row['image_id'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = remove_hair(img)
        if self.transform: img = self.transform(image=img)['image']
        meta = torch.tensor(self.meta_arr[idx])
        label = torch.tensor(self.unified_label_map[row['unified_dx']], dtype=torch.long)
        return img, meta, label
    
# ===============================================================
# DATA AUGMENTATION
# ===============================================================
DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD  = [0.229, 0.224, 0.225]

train_tf = A.Compose([
    A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE), 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7),
    A.RandomBrightnessContrast(p=0.7),
    # <<< THAY ĐỔI 3: Vô hiệu hóa CoarseDropout để tránh xóa mất vùng tổn thương
    # A.CoarseDropout(max_holes=8, max_height=24, max_width=24, fill_value=0, p=0.5),
    A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ToTensorV2(),
])

valid_tf = A.Compose([
    A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
    A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ToTensorV2(),
])

# ===============================================================
# MODEL ARCHITECTURE (No changes needed here)
# ===============================================================
class EnhancedChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        self.conv3x3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv_final = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        out_3x3 = self.conv3x3(pooled)
        out_5x5 = self.conv5x5(pooled)
        out_7x7 = self.conv7x7(pooled)
        multi_scale = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)
        return x * self.sigmoid(self.conv_final(multi_scale))

class EnhancedCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(EnhancedCBAM, self).__init__()
        self.channel_attention = EnhancedChannelAttention(in_planes, ratio)
        self.spatial_attention = MultiScaleSpatialAttention()
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = EnhancedCBAM(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CustomImageEncoder(nn.Module):
    def __init__(self, block, num_blocks, out_dim=512):
        super(CustomImageEncoder, self).__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, out_dim)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.stem(x); out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out); out = self.avgpool(out)
        out = torch.flatten(out, 1); out = self.projection(out)
        return out

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(1024, out_dim), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class MMFNet(nn.Module):
    def __init__(self, meta_dim, cfg, num_classes):
        super().__init__()
        self.img_enc = CustomImageEncoder(ResidualCBAMBlock, [2, 2, 2, 2], out_dim=cfg.EMBED_DIM)
        self.meta_enc = MLPEncoder(in_dim=meta_dim, out_dim=cfg.EMBED_DIM, dropout=cfg.DROPOUT_RATE)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.EMBED_DIM * 2),
            nn.Linear(cfg.EMBED_DIM * 2, cfg.EMBED_DIM), nn.GELU(),
            nn.Dropout(cfg.DROPOUT_RATE), nn.Linear(cfg.EMBED_DIM, num_classes))
    def forward(self, img, meta):
        img_f = self.img_enc(img)
        if meta.numel() == 0: meta_f = torch.zeros_like(img_f)
        else: meta_f = self.meta_enc(meta)
        fused = torch.cat([img_f, meta_f], dim=1)
        logits = self.classifier(fused)
        return logits

# ===============================================================
# LOSS FUNCTIONS & TRAINING LOOPS (No changes needed here)
# ===============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def compute_metrics(y_true, y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    except ValueError:
        auc = 0.5
    return {"acc": acc, "bacc": bacc, "auc": auc}

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for imgs, metas, labels in tqdm(loader, desc="Train", leave=False):
        imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, metas)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    
    epoch_loss = total_loss / len(loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_probs = np.vstack(all_probs)
    epoch_metrics = compute_metrics(all_labels, all_probs)
    
    wandb.log({
        "train_loss": epoch_loss, "train_acc": epoch_metrics['acc'],
        "train_bacc": epoch_metrics['bacc'], "train_auc": epoch_metrics['auc'],
        "lr": optimizer.param_groups[0]['lr']
    })
    return epoch_loss, epoch_metrics

def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, metas, labels in tqdm(loader, desc="Val", leave=False):
            imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
            logits = model(imgs, metas)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    
    epoch_loss = total_loss / len(loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_probs = np.vstack(all_probs)
    epoch_metrics = compute_metrics(all_labels, all_probs)
    
    wandb.log({
        "val_loss": epoch_loss, "val_acc": epoch_metrics['acc'],
        "val_bacc": epoch_metrics['bacc'], "val_auc": epoch_metrics['auc']
    })
    return epoch_loss, epoch_metrics

def plot_training_history(history, fold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Training History for Fold {fold}', fontsize=16)
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history['val_bacc'], label='Validation BAcc')
    ax2.plot(history['val_auc'], label='Validation AUC')
    ax2.set_title('Metrics vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    plt.show()

print("Common components loaded.")