"""
GeoShift — Fast ConvNeXt-Tiny + Vision Transformer Hybrid
===========================================================

Hybrid architecture:
    4-channel B,G,R,NIR
        -> ImageNet-pretrained ConvNeXt-Tiny
        -> 4x4 feature map (768 channels)
        -> 16 spatial tokens + learnable CLS token
        -> lightweight 2-layer Transformer encoder
        -> 6-class classifier

This version intentionally uses a GENERAL stratified 80:20 split, not
region-aware splitting, because the requested experiment is a general
train/validation benchmark.

The competition test set is NEVER used for model selection.

Important:
    A script cannot mathematically guarantee validation Macro-F1 will
    monotonically increase with training Macro-F1. This pipeline instead:
      * computes clean, non-MixUp training Macro-F1 after each epoch;
      * uses validation Macro-F1 as the model-selection metric;
      * reduces the learning rate when validation Macro-F1 plateaus;
      * saves the best checkpoint only by validation Macro-F1.

Speed:
    The Transformer operates on only 17 tokens (16 ConvNeXt spatial tokens
    + CLS), so its overhead is small compared with the CNN. The script
    measures epoch time and warns if the requested 30 s/epoch budget is
    exceeded. Exact time depends on GPU, disk speed, PyTorch/CUDA versions,
    and dataset size; no code can guarantee <=30 s on unknown hardware.
"""

from pathlib import Path
import copy
import math
import random
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(r"C:\Users\badri\OneDrive\Desktop\GIS-intra-iit\GeoShift\data")

TRAIN_IMAGES_PATH = DATA_DIR / "train_images.npy"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.npy"
TRAIN_CSV_PATH = DATA_DIR / "train.csv"
TEST_IMAGES_PATH = DATA_DIR / "test_images.npy"
TEST_CSV_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

BEST_MODEL_PATH = DATA_DIR / "best_geoshift_convnext_vit_fast.pth"
HISTORY_PATH = DATA_DIR / "convnext_vit_fast_history.csv"
REPORT_PATH = DATA_DIR / "convnext_vit_fast_best_report.txt"
CONFUSION_PATH = DATA_DIR / "convnext_vit_fast_best_confusion_matrix.csv"
SUBMISSION_PATH = DATA_DIR / "submission_convnext_vit_fast.csv"

NUM_CLASSES = 6
VAL_SIZE = 0.20
RANDOM_SEED = 42

# Speed-oriented settings.
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = True
USE_AMP = True
USE_CHANNELS_LAST = True

# ConvNeXt fine-tuning.
LR_BACKBONE = 2.0e-5
LR_STEM = 5.0e-5
LR_TRANSFORMER = 5.0e-5
LR_CLASSIFIER = 1.5e-4
WEIGHT_DECAY = 1.0e-4

# Lightweight ViT head.
VIT_DIM = 384
VIT_HEADS = 6
VIT_LAYERS = 2
VIT_FF_DIM = 768
VIT_DROPOUT = 0.10

# Regularization / Macro-F1 orientation.
LABEL_SMOOTHING = 0.05
USE_CLASS_BALANCED_LOSS = True
CLASS_WEIGHT_POWER = 0.50       # moderated inverse-frequency weighting
USE_MIXUP = True
MIXUP_PROBABILITY = 0.20        # reduced to preserve clean fine-tuning
MIXUP_ALPHA = 0.20
GRAD_CLIP_NORM = 1.0

# EMA.
USE_EMA = True
EMA_DECAY = 0.9995

# TTA costs time. Keep it OFF during routine validation for speed.
USE_VALIDATION_TTA = False
USE_TEST_TTA = True

# Warmup + cosine.
WARMUP_EPOCHS = 5
MIN_LR_FACTOR = 0.05

# If validation Macro-F1 does not improve, reduce all LRs.
PLATEAU_PATIENCE = 5
PLATEAU_FACTOR = 0.5

MAX_EPOCH_SECONDS = 30.0

CLASS_NAMES = [
    "Forest",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Water/Wetland",
]

# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

print("=" * 80)
print("GeoShift — ConvNeXt-Tiny + Lightweight Vision Transformer")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Train/Validation split: 80% / 20%")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Workers: {NUM_WORKERS}")
print()

# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")
train_images = np.load(TRAIN_IMAGES_PATH)
train_labels = np.load(TRAIN_LABELS_PATH).astype(np.int64)
test_images = np.load(TEST_IMAGES_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

if train_images.ndim != 4 or train_images.shape[1:] != (4, 64, 64):
    raise ValueError(f"Expected train_images shape (N,4,64,64), got {train_images.shape}")
if test_images.ndim != 4 or test_images.shape[1:] != (4, 64, 64):
    raise ValueError(f"Expected test_images shape (N,4,64,64), got {test_images.shape}")
if len(train_images) != len(train_labels) or len(train_images) != len(train_df):
    raise ValueError("Training image/label/csv lengths do not match.")
if len(test_images) != len(test_df):
    raise ValueError("Test image/csv lengths do not match.")

print(f"train_images: {train_images.shape} {train_images.dtype}")
print(f"test_images : {test_images.shape} {test_images.dtype}")
print()

# ============================================================
# GENERAL STRATIFIED 80:20 SPLIT
# ============================================================

all_idx = np.arange(len(train_images))
train_idx, val_idx = train_test_split(
    all_idx,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True,
    stratify=train_labels,
)

train_idx = np.sort(train_idx)
val_idx = np.sort(val_idx)

train_x = train_images[train_idx]
train_y = train_labels[train_idx]
val_x = train_images[val_idx]
val_y = train_labels[val_idx]

print("=" * 80)
print("80:20 STRATIFIED SPLIT")
print("=" * 80)
print(f"Train: {len(train_idx):,} ({100*len(train_idx)/len(all_idx):.2f}%)")
print(f"Val  : {len(val_idx):,} ({100*len(val_idx)/len(all_idx):.2f}%)")
print()

# ============================================================
# TRAIN-ONLY NORMALIZATION
# ============================================================

channel_mean = train_x.astype(np.float32).mean(axis=(0, 2, 3)) / 255.0
channel_std = train_x.astype(np.float32).std(axis=(0, 2, 3)) / 255.0
channel_std = np.maximum(channel_std, 1e-6)

print("Train-only channel mean:", channel_mean)
print("Train-only channel std :", channel_std)
print()

# ============================================================
# DATASET
# ============================================================

class SatelliteDataset(Dataset):
    def __init__(self, images, labels, mean, std, augment=False):
        self.images = images
        self.labels = labels
        self.mean = np.asarray(mean, dtype=np.float32).reshape(4, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(4, 1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].astype(np.float32) / 255.0

        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()

        x = (x - self.mean) / self.std
        x = torch.from_numpy(x.copy()).float()

        if self.labels is None:
            return x
        return x, torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = SatelliteDataset(train_x, train_y, channel_mean, channel_std, augment=True)
val_dataset = SatelliteDataset(val_x, val_y, channel_mean, channel_std, augment=False)
test_dataset = SatelliteDataset(test_images, None, channel_mean, channel_std, augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    persistent_workers=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available(),
)

# ============================================================
# HYBRID MODEL
# ============================================================

class ConvNeXtViT(nn.Module):
    """ImageNet ConvNeXt-Tiny feature extractor + small ViT token head."""

    def __init__(self, num_classes=6):
        super().__init__()

        # FULL ImageNet-pretrained ConvNeXt-Tiny.
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        # Convert RGB stem -> B,G,R,NIR while retaining pretrained filters.
        old_stem = backbone.features[0][0]
        new_stem = nn.Conv2d(
            4,
            old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
        )

        with torch.no_grad():
            # GeoShift order is B,G,R,NIR; ImageNet order is R,G,B.
            new_stem.weight[:, 0].copy_(old_stem.weight[:, 2])
            new_stem.weight[:, 1].copy_(old_stem.weight[:, 1])
            new_stem.weight[:, 2].copy_(old_stem.weight[:, 0])
            new_stem.weight[:, 3].copy_(old_stem.weight.mean(dim=1))
            if new_stem.bias is not None and old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        backbone.features[0][0] = new_stem

        # Keep ConvNeXt features only; its original classifier is replaced.
        self.features = backbone.features
        self.feature_dim = 768

        # ConvNeXt on 64x64 produces 4x4 spatial features.
        self.token_projection = nn.Linear(self.feature_dim, VIT_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, VIT_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 17, VIT_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=VIT_DIM,
            nhead=VIT_HEADS,
            dim_feedforward=VIT_FF_DIM,
            dropout=VIT_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=VIT_LAYERS,
        )
        self.norm = nn.LayerNorm(VIT_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(VIT_DIM, VIT_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(VIT_DIM // 2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)               # B,768,4,4
        x = x.flatten(2).transpose(1, 2)   # B,16,768
        x = self.token_projection(x)       # B,16,384

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)      # B,17,384
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])             # CLS token
        return self.classifier(x)


print("=" * 80)
print("BUILDING IMAGE-NET PRETRAINED CONVNEXT-TINY + VIT")
print("=" * 80)

model = ConvNeXtViT(NUM_CLASSES).to(DEVICE)
if USE_CHANNELS_LAST and DEVICE.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params:,}")
print(f"ViT tokens: 17 (CLS + 4x4 ConvNeXt tokens)")
print()

# ============================================================
# EMA
# ============================================================

class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        es = self.ema.state_dict()
        ms = model.state_dict()
        for k in es:
            if es[k].is_floating_point():
                es[k].mul_(self.decay).add_(ms[k], alpha=1.0 - self.decay)
            else:
                es[k].copy_(ms[k])

ema = ModelEMA(model, EMA_DECAY) if USE_EMA else None

# ============================================================
# OPTIMIZER: DIFFERENTIAL LR
# ============================================================

stem_ids = {id(p) for p in model.features[0][0].parameters()}
vit_ids = set()
for module in [model.token_projection, model.transformer, model.norm, model.cls_token, model.pos_embed]:
    if isinstance(module, nn.Parameter):
        vit_ids.add(id(module))
    else:
        vit_ids.update(id(p) for p in module.parameters())
classifier_ids = {id(p) for p in model.classifier.parameters()}

param_groups = {
    "backbone_decay": [], "backbone_no_decay": [],
    "stem_decay": [], "stem_no_decay": [],
    "vit_decay": [], "vit_no_decay": [],
    "classifier_decay": [], "classifier_no_decay": [],
}

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    no_decay = p.ndim == 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name

    if id(p) in stem_ids:
        prefix = "stem"
    elif id(p) in vit_ids:
        prefix = "vit"
    elif id(p) in classifier_ids:
        prefix = "classifier"
    else:
        prefix = "backbone"

    key = f"{prefix}_{'no_decay' if no_decay else 'decay'}"
    param_groups[key].append(p)

optimizer = torch.optim.AdamW([
    {"params": param_groups["backbone_decay"], "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
    {"params": param_groups["backbone_no_decay"], "lr": LR_BACKBONE, "weight_decay": 0.0},
    {"params": param_groups["stem_decay"], "lr": LR_STEM, "weight_decay": WEIGHT_DECAY},
    {"params": param_groups["stem_no_decay"], "lr": LR_STEM, "weight_decay": 0.0},
    {"params": param_groups["vit_decay"], "lr": LR_TRANSFORMER, "weight_decay": WEIGHT_DECAY},
    {"params": param_groups["vit_no_decay"], "lr": LR_TRANSFORMER, "weight_decay": 0.0},
    {"params": param_groups["classifier_decay"], "lr": LR_CLASSIFIER, "weight_decay": WEIGHT_DECAY},
    {"params": param_groups["classifier_no_decay"], "lr": LR_CLASSIFIER, "weight_decay": 0.0},
])

# ============================================================
# CLASS-BALANCED LOSS
# ============================================================

counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(np.float64)
weights = (counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))) ** CLASS_WEIGHT_POWER
weights = weights / weights.mean()
class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

print("Class weights:", np.round(weights, 4))

criterion = nn.CrossEntropyLoss(
    weight=class_weights if USE_CLASS_BALANCED_LOSS else None,
    label_smoothing=LABEL_SMOOTHING,
)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")

# ============================================================
# LR SCHEDULER
# ============================================================

base_lrs = [g["lr"] for g in optimizer.param_groups]

# Validation-F1 plateau reduction is applied manually. A cosine factor
# is also used during the first planned schedule; plateau reduction can
# further reduce the current LR when F1 stops improving.

def cosine_factor(epoch):
    if epoch <= WARMUP_EPOCHS:
        return epoch / max(1, WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
    return MIN_LR_FACTOR + 0.5 * (1.0 - MIN_LR_FACTOR) * (1.0 + math.cos(math.pi * progress))

# ============================================================
# MIXUP
# ============================================================

def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

# ============================================================
# TTA
# ============================================================

def tta_transforms(x):
    yield x
    yield torch.flip(x, [3])
    yield torch.flip(x, [2])
    yield torch.flip(x, [2, 3])
    yield torch.rot90(x, 1, [2, 3])
    yield torch.rot90(x, 2, [2, 3])
    yield torch.rot90(x, 3, [2, 3])
    yield x.transpose(2, 3)

# ============================================================
# CLEAN TRAINING METRIC
# ============================================================

@torch.no_grad()
def evaluate_loader(eval_model, loader, use_tta=False):
    eval_model.eval()
    targets, predictions = [], []
    total_loss = 0.0

    for x, y in loader:
        if DEVICE.type == "cuda" and USE_CHANNELS_LAST:
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if use_tta:
            probs = None
            for tx in tta_transforms(x):
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                    p = torch.softmax(eval_model(tx), dim=1)
                probs = p if probs is None else probs + p
            probs = probs / 8.0
            logits_for_loss = torch.log(probs.clamp_min(1e-8))
            loss = torch.nn.functional.nll_loss(logits_for_loss, y)
            pred = probs.argmax(1)
        else:
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                logits = eval_model(x)
                loss = criterion(logits, y)
            pred = logits.argmax(1)

        total_loss += loss.item() * x.size(0)
        targets.extend(y.cpu().numpy())
        predictions.extend(pred.cpu().numpy())

    f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    acc = accuracy_score(targets, predictions)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, f1, acc, np.asarray(targets), np.asarray(predictions)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():
    model.train()
    total_loss = 0.0

    # First pass: optimize with augmentation/MixUp.
    for x, y in train_loader:
        if DEVICE.type == "cuda" and USE_CHANNELS_LAST:
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        do_mix = USE_MIXUP and np.random.rand() < MIXUP_PROBABILITY and x.size(0) > 1
        if do_mix:
            x_mix, ya, yb, lam = mixup_batch(x, y, MIXUP_ALPHA)
        else:
            x_mix, ya, yb, lam = x, y, y, 1.0

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
            logits = model(x_mix)
            if do_mix:
                loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
            else:
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * x.size(0)

    # IMPORTANT: clean training F1 is evaluated without augmentation or MixUp,
    # so it is directly comparable to validation F1.
    clean_loss, clean_f1, clean_acc, _, _ = evaluate_loader(model, train_eval_loader, use_tta=False)
    return total_loss / len(train_loader.dataset), clean_f1, clean_acc, clean_loss

# Separate clean training-evaluation loader.
train_eval_dataset = SatelliteDataset(train_x, train_y, channel_mean, channel_std, augment=False)
train_eval_loader = DataLoader(
    train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available(),
)

# ============================================================
# TRAINING LOOP
# ============================================================

history = []
best_val_f1 = -1.0
best_epoch = -1
best_state = None
best_ema_state = None
plateau_count = 0

training_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # Warmup/cosine base schedule.
    factor = cosine_factor(epoch)
    for i, group in enumerate(optimizer.param_groups):
        group["lr"] = base_lrs[i] * factor

    train_loss, train_f1, train_acc, clean_train_loss = train_one_epoch()

    eval_model = ema.ema if ema is not None else model
    val_loss, val_f1, val_acc, val_targets, val_predictions = evaluate_loader(
        eval_model, val_loader, use_tta=USE_VALIDATION_TTA
    )

    # Validation-F1 plateau logic.
    if val_f1 > best_val_f1 + 1e-9:
        plateau_count = 0
    else:
        plateau_count += 1
        if plateau_count >= PLATEAU_PATIENCE:
            for group in optimizer.param_groups:
                group["lr"] *= PLATEAU_FACTOR
            plateau_count = 0
            print(f"  ↳ Validation F1 plateau: LR × {PLATEAU_FACTOR}")

    epoch_time = time.time() - epoch_start

    row = {
        "epoch": epoch,
        "train_loss_augmented": train_loss,
        "train_loss_clean": clean_train_loss,
        "train_f1": train_f1,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_accuracy": val_acc,
        "epoch_time_sec": epoch_time,
        "lr_backbone": optimizer.param_groups[0]["lr"],
        "lr_stem": optimizer.param_groups[2]["lr"],
        "lr_transformer": optimizer.param_groups[4]["lr"],
        "lr_classifier": optimizer.param_groups[6]["lr"],
    }
    history.append(row)
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
        f"Train F1 {train_f1:.5f} | Val F1 {val_f1:.5f} | "
        f"Train Acc {train_acc:.5f} | Val Acc {val_acc:.5f} | "
        f"{epoch_time:.1f}s"
    )

    if epoch_time > MAX_EPOCH_SECONDS:
        print(f"  ⚠ Epoch exceeded requested {MAX_EPOCH_SECONDS:.0f}s budget.")

    # Best model is selected ONLY by validation Macro-F1.
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        best_ema_state = copy.deepcopy(ema.ema.state_dict()) if ema is not None else None

        checkpoint = {
            "epoch": epoch,
            "val_f1": float(val_f1),
            "val_accuracy": float(val_acc),
            "model_state_dict": best_state,
            "ema_state_dict": best_ema_state,
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "model_type": "ConvNeXt-Tiny + 2-layer ViT",
            "vit_dim": VIT_DIM,
            "vit_heads": VIT_HEADS,
            "vit_layers": VIT_LAYERS,
        }
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"  ★ NEW BEST VAL MACRO-F1: {best_val_f1:.6f}")

print()
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best validation Macro-F1: {best_val_f1:.6f}")
print(f"Best epoch: {best_epoch}")
print(f"Total time: {(time.time() - training_start) / 60:.1f} min")

# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)

best_model = ConvNeXtViT(NUM_CLASSES).to(DEVICE)
best_model.load_state_dict(
    checkpoint["ema_state_dict"] if checkpoint.get("ema_state_dict") is not None else checkpoint["model_state_dict"]
)
best_model.eval()

# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

_, final_val_f1, final_val_acc, y_true, y_pred = evaluate_loader(
    best_model, val_loader, use_tta=USE_VALIDATION_TTA
)

report = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, digits=6, zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Validation Macro-F1: {final_val_f1:.8f}\n")
    f.write(f"Validation Accuracy: {final_val_acc:.8f}\n\n")
    f.write(report)

pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(CONFUSION_PATH)

print("\nFINAL VALIDATION REPORT")
print(report)

# ============================================================
# COMPETITION TEST PREDICTION
# ============================================================

@torch.no_grad()
def predict_test(eval_model):
    eval_model.eval()
    preds = []

    for x in test_loader:
        if DEVICE.type == "cuda" and USE_CHANNELS_LAST:
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(DEVICE, non_blocking=True)

        if USE_TEST_TTA:
            probs = None
            for tx in tta_transforms(x):
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                    p = torch.softmax(eval_model(tx), dim=1)
                probs = p if probs is None else probs + p
            pred = (probs / 8.0).argmax(1)
        else:
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                pred = eval_model(x).argmax(1)

        preds.extend(pred.cpu().numpy().tolist())

    return np.asarray(preds, dtype=np.int64)

print("Generating competition-test predictions...")
test_predictions = predict_test(best_model)

submission = sample_submission.copy()
if "Id" not in submission.columns or "label" not in submission.columns:
    submission = pd.DataFrame({"Id": test_df["Id"].values})
submission["label"] = test_predictions
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Saved submission: {SUBMISSION_PATH}")
print("Done.")
