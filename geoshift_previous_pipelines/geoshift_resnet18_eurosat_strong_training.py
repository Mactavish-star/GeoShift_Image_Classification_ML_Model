"""
GeoShift — STRONG EuroSAT-Pretrained ResNet-18 DEVELOPMENT PIPELINE
====================================================

Goal:
    Train the same ResNet-18-Tiny architecture that successfully
    powered the previous GeoShift / EuroSAT predictor, but with a
    stronger, longer training pipeline.

Key properties:
    - 4-channel B,G,R,NIR ResNet-18
    - ImageNet pretrained backbone
    - Region-aware StratifiedGroupKFold 70:15:15 train/validation/internal-test split
    - No early stopping
    - Train for the full NUM_EPOCHS
    - Save a checkpoint whenever validation Macro-F1 reaches a new high
    - Train-only normalization
    - Geometric augmentation
    - Optional MixUp
    - Label smoothing
    - Differential learning rates
    - Warmup + cosine decay
    - AdamW
    - AMP
    - Gradient clipping
    - EMA model for validation/checkpoint selection
    - Validation-time 8-way dihedral TTA
    - Internal-test evaluation from train_images.npy
    - Full training history
    - Per-class F1 + confusion matrix
    - Final competition-test prediction + submission

The backbone is a ResNet-18 already fine-tuned on EuroSAT (Sentinel-2
RGB land-cover imagery). The published model is loaded from the Hugging Face
Hub through timm, then its first convolution is adapted from 3 RGB channels
to 4 B,G,R,NIR channels and its 10-class EuroSAT head is replaced by 6 GeoShift classes.

No DataLoader workers are used.
"""

# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\badri\OneDrive\Desktop\GIS-intra-iit\GeoShift\data"
)

TRAIN_IMAGES_PATH = DATA_DIR / "train_images.npy"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.npy"
TRAIN_CSV_PATH = DATA_DIR / "train.csv"

TEST_IMAGES_PATH = DATA_DIR / "test_images.npy"
TEST_CSV_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

BEST_MODEL_PATH = DATA_DIR / "best_geoshift_resnet18_eurosat_strong.pth"
BEST_EMA_MODEL_PATH = DATA_DIR / "best_geoshift_resnet18_eurosat_strong_ema.pth"

HISTORY_PATH = DATA_DIR / "strong_training_history.csv"
REPORT_PATH = DATA_DIR / "strong_best_classification_report.txt"
CONFUSION_PATH = DATA_DIR / "strong_best_confusion_matrix.csv"
SUBMISSION_PATH = DATA_DIR / "submission_resnet18_eurosat_strong.csv"

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

NUM_CLASSES = 6
# The labeled train_images.npy pool is split into:
#   70% model-training data
#   15% validation data (used for model selection)
#   15% internal test data (held out until the best model is ready)
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
INTERNAL_TEST_SIZE = 0.15
RANDOM_SEED = 42

# IMPORTANT:
# There is NO early stopping.
# Training always continues through all epochs.
NUM_EPOCHS = 200
WARMUP_EPOCHS = 10

BATCH_SIZE = 64
NUM_WORKERS = 0

# Differential LR.
LEARNING_RATE_BACKBONE = 3e-5
LEARNING_RATE_STEM = 1e-4
LEARNING_RATE_CLASSIFIER = 3e-4

WEIGHT_DECAY = 1e-4

# Label smoothing.
LABEL_SMOOTHING = 0.05

# MixUp.
USE_MIXUP = True
MIXUP_PROBABILITY = 0.35
MIXUP_ALPHA = 0.20

# EMA.
USE_EMA = True
EMA_DECAY = 0.9997

# Gradient clipping.
GRAD_CLIP_NORM = 1.0

# Mixed precision.
USE_AMP = True

# Validation TTA:
# identity + H + V + HV + rotations 90/180/270 + transpose.
USE_VALIDATION_TTA = True

# Test TTA.
USE_TEST_TTA = True

# Print detailed information every N epochs.
MONITOR_EVERY = 5

# Minimum improvement required to save a new best.
# Set to 0.0 if every tiny floating-point improvement should save.
MIN_DELTA = 0.0

CLASS_NAMES = [
    "Forest",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Water/Wetland",
]


# ============================================================
# IMPORTS
# ============================================================

import copy
import math
import random
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import timm

warnings.filterwarnings("ignore")


# ============================================================
# REPRODUCIBILITY / PERFORMANCE
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("GeoShift — STRONG ResNet-18-Tiny PIPELINE")
print("=" * 80)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

print(f"Epochs: {NUM_EPOCHS}")
print("Early stopping: DISABLED")
print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

train_images = np.load(TRAIN_IMAGES_PATH)
train_labels = np.load(TRAIN_LABELS_PATH)

test_images = np.load(TEST_IMAGES_PATH)

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print(f"train_images: {train_images.shape} {train_images.dtype}")
print(f"train_labels: {train_labels.shape} {train_labels.dtype}")
print(f"test_images:  {test_images.shape} {test_images.dtype}")
print(f"train.csv:    {train_df.shape}")
print(f"test.csv:     {test_df.shape}")
print()


# ============================================================
# DATA CHECKS
# ============================================================

def check_images(images, name):

    if images.ndim != 4:
        raise ValueError(
            f"{name} must have shape (N,4,64,64). "
            f"Got {images.shape}"
        )

    if images.shape[1] != 4:
        raise ValueError(
            f"{name} must have 4 channels B,G,R,NIR. "
            f"Got {images.shape[1]}"
        )

    if images.shape[2:] != (64, 64):
        raise ValueError(
            f"{name} must contain 64x64 tiles. "
            f"Got {images.shape[2:]}"
        )


check_images(train_images, "train_images")
check_images(test_images, "test_images")

train_labels = train_labels.astype(np.int64)

if len(train_images) != len(train_labels):
    raise ValueError(
        "train_images and train_labels have different lengths."
    )

if len(train_df) != len(train_images):
    raise ValueError(
        "train.csv rows do not match train_images."
    )

if len(test_df) != len(test_images):
    raise ValueError(
        "test.csv rows do not match test_images."
    )

if train_labels.min() < 0 or train_labels.max() >= NUM_CLASSES:
    raise ValueError("Labels must be integers 0..5.")

if "Id" not in train_df.columns:
    raise ValueError("train.csv must contain Id.")

if "Id" not in test_df.columns:
    raise ValueError("test.csv must contain Id.")

# Extract region from Id exactly in the GeoShift convention.
train_df["Region"] = (
    train_df["Id"]
    .astype(str)
    .str.extract(r"(R\d{2})")[0]
)

if train_df["Region"].isna().any():
    bad = int(train_df["Region"].isna().sum())
    raise ValueError(
        f"Could not extract Region from {bad} train IDs."
    )

regions = train_df["Region"].astype(str).to_numpy()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("=" * 80)
print("FULL TRAINING CLASS DISTRIBUTION")
print("=" * 80)

full_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

for i, name in enumerate(CLASS_NAMES):

    pct = (
        100.0 * full_counts[i] / len(train_labels)
    )

    print(
        f"{i} | {name:15s} | "
        f"{full_counts[i]:7d} | {pct:6.2f}%"
    )

print()


# ============================================================
# REGION-AWARE 70:15:15 TRAIN / VALIDATION / INTERNAL-TEST SPLIT
# ============================================================
#
# Important distinction:
#   * internal_test_* comes from train_images.npy and has labels.
#     It is NOT the competition test set.
#   * test_images.npy remains completely untouched until the final
#     prediction stage after the best checkpoint has been selected.
#
# We use StratifiedGroupKFold twice:
#   1. 10 folds over the full labeled training pool. Three folds are
#      selected as a ~30% holdout, leaving ~70% for training.
#   2. The 30% holdout is split into two group-disjoint halves using
#      StratifiedGroupKFold(n_splits=2), giving ~15% validation and
#      ~15% internal test.
#
# Because groups are whole regions, exact 70/15/15 counts are not
# guaranteed when region sizes differ. The code chooses folds that
# minimize the deviation from the requested fractions while also
# keeping class proportions close to the full labeled dataset.
# ============================================================

print("=" * 80)
print("REGION-AWARE 70:15:15 SPLIT")
print("=" * 80)

full_class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES,
).astype(np.float64)
full_class_distribution = (
    full_class_counts / full_class_counts.sum()
)


def split_quality(indices, target_fraction):
    """Score a candidate split by size and class-distribution deviation."""

    fraction = len(indices) / len(train_images)
    counts = np.bincount(
        train_labels[indices],
        minlength=NUM_CLASSES,
    ).astype(np.float64)

    if counts.sum() > 0:
        distribution = counts / counts.sum()
    else:
        distribution = np.zeros(NUM_CLASSES, dtype=np.float64)

    size_error = abs(fraction - target_fraction)
    class_error = np.mean(
        np.abs(distribution - full_class_distribution)
    )

    # Size is the primary objective; class balance is the tie-breaker.
    return size_error + 0.50 * class_error


# ------------------------------------------------------------
# Stage 1: choose a group-disjoint ~30% holdout from 10 SGKF folds.
# ------------------------------------------------------------

outer_sgkf = StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_SEED,
)

outer_folds = []

for fold, (_, fold_idx) in enumerate(
    outer_sgkf.split(
        train_images,
        train_labels,
        groups=regions,
    )
):
    outer_folds.append(fold_idx)

if len(outer_folds) != 10:
    raise RuntimeError("Expected 10 outer SGKF folds.")

# Three 10%-ish folds make the desired ~30% holdout.
# Enumerating 10 choose 3 = 120 combinations is tiny and lets us
# choose a combination with both the requested size and class balance.
best_outer = None

for a in range(10):
    for b in range(a + 1, 10):
        for c in range(b + 1, 10):
            candidate = np.concatenate(
                [outer_folds[a], outer_folds[b], outer_folds[c]]
            )

            score = split_quality(
                candidate,
                (1.0 - TRAIN_SIZE),
            )

            if best_outer is None or score < best_outer["score"]:
                best_outer = {
                    "folds": (a, b, c),
                    "indices": candidate,
                    "score": score,
                }

holdout_idx = np.sort(best_outer["indices"])

all_indices = np.arange(len(train_images))
train_idx = np.setdiff1d(
    all_indices,
    holdout_idx,
    assume_unique=False,
)

# ------------------------------------------------------------
# Stage 2: split the ~30% holdout into validation/internal-test.
# ------------------------------------------------------------

holdout_labels = train_labels[holdout_idx]
holdout_regions = regions[holdout_idx]
holdout_images = train_images[holdout_idx]

inner_sgkf = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=RANDOM_SEED + 1,
)

inner_candidates = []

for fold, (_, half_idx) in enumerate(
    inner_sgkf.split(
        holdout_images,
        holdout_labels,
        groups=holdout_regions,
    )
):
    global_idx = holdout_idx[half_idx]
    inner_candidates.append(
        {
            "fold": fold,
            "indices": global_idx,
            "fraction": len(global_idx) / len(train_images),
        }
    )

selected_inner = min(
    inner_candidates,
    key=lambda item: split_quality(
        item["indices"],
        INTERNAL_TEST_SIZE,
    ),
)

internal_test_idx = np.sort(
    selected_inner["indices"]
)

val_idx = np.sort(
    np.setdiff1d(
        holdout_idx,
        internal_test_idx,
        assume_unique=False,
    )
)

train_x = train_images[train_idx]
train_y = train_labels[train_idx]

val_x = train_images[val_idx]
val_y = train_labels[val_idx]

internal_test_x = train_images[internal_test_idx]
internal_test_y = train_labels[internal_test_idx]

train_regions = set(regions[train_idx])
val_regions = set(regions[val_idx])
internal_test_regions = set(regions[internal_test_idx])

if train_regions & val_regions:
    raise RuntimeError(
        f"REGION LEAKAGE DETECTED between train/validation: "
        f"{train_regions & val_regions}"
    )

if train_regions & internal_test_regions:
    raise RuntimeError(
        f"REGION LEAKAGE DETECTED between train/internal-test: "
        f"{train_regions & internal_test_regions}"
    )

if val_regions & internal_test_regions:
    raise RuntimeError(
        f"REGION LEAKAGE DETECTED between validation/internal-test: "
        f"{val_regions & internal_test_regions}"
    )

print(f"Outer SGKF holdout folds: {best_outer['folds']}")
print(f"Inner SGKF selected fold:  {selected_inner['fold']}")
print()
print(f"Train samples:              {len(train_x)}")
print(f"Validation samples:         {len(val_x)}")
print(f"Internal-test samples:      {len(internal_test_x)}")
print(f"Train fraction:             {len(train_x) / len(train_images):.4f}")
print(f"Validation fraction:        {len(val_x) / len(train_images):.4f}")
print(f"Internal-test fraction:     {len(internal_test_x) / len(train_images):.4f}")
print()
print(f"Train regions:              {len(train_regions)}")
print(f"Validation regions:         {len(val_regions)}")
print(f"Internal-test regions:      {len(internal_test_regions)}")
print(f"Train/Val overlap:          {len(train_regions & val_regions)}")
print(f"Train/Internal-test overlap:{len(train_regions & internal_test_regions)}")
print(f"Val/Internal-test overlap:  {len(val_regions & internal_test_regions)}")

print("\nTrain class counts:")
print(np.bincount(train_y, minlength=NUM_CLASSES))

print("\nValidation class counts:")
print(np.bincount(val_y, minlength=NUM_CLASSES))

print("\nInternal-test class counts:")
print(np.bincount(internal_test_y, minlength=NUM_CLASSES))

# ============================================================
# TRAIN-ONLY CHANNEL NORMALIZATION
# ============================================================

train_float = (
    train_x.astype(np.float32) / 255.0
)

channel_mean = train_float.mean(
    axis=(0, 2, 3)
)

channel_std = train_float.std(
    axis=(0, 2, 3)
)

channel_std = np.maximum(
    channel_std,
    1e-6
)

del train_float

print("=" * 80)
print("CHANNEL NORMALIZATION")
print("=" * 80)

for i, name in enumerate(
    ["Blue", "Green", "Red", "NIR"]
):

    print(
        f"{name:5s}: "
        f"mean={channel_mean[i]:.7f} "
        f"std={channel_std[i]:.7f}"
    )

print()


# ============================================================
# DATASET
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(
        self,
        images,
        labels,
        mean,
        std,
        augment=False,
    ):

        self.images = images
        self.labels = labels

        self.mean = np.asarray(
            mean,
            dtype=np.float32
        ).reshape(4, 1, 1)

        self.std = np.asarray(
            std,
            dtype=np.float32
        ).reshape(4, 1, 1)

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        x = (
            self.images[idx]
            .astype(np.float32)
            / 255.0
        )

        if self.augment:

            # Random horizontal flip.
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()

            # Random vertical flip.
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()

            # Random 90-degree rotation.
            k = np.random.randint(0, 4)

            if k:
                x = np.rot90(
                    x,
                    k=k,
                    axes=(1, 2)
                ).copy()

        x = (
            x - self.mean
        ) / self.std

        x = torch.from_numpy(
            x.copy()
        ).float()

        if self.labels is None:
            return x

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return x, y


train_dataset = SatelliteDataset(
    train_x,
    train_y,
    channel_mean,
    channel_std,
    augment=True,
)

val_dataset = SatelliteDataset(
    val_x,
    val_y,
    channel_mean,
    channel_std,
    augment=False,
)

internal_test_dataset = SatelliteDataset(
    internal_test_x,
    internal_test_y,
    channel_mean,
    channel_std,
    augment=False,
)

competition_test_dataset = SatelliteDataset(
    test_images,
    None,
    channel_mean,
    channel_std,
    augment=False,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

internal_test_loader = DataLoader(
    internal_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

competition_test_loader = DataLoader(
    competition_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# MODEL
# EURO-SAT-PRETRAINED RESNET-18
# ============================================================

print("=" * 80)
print("BUILDING EURO-SAT-PRETRAINED RESNET-18")
print("=" * 80)

EUROSAT_MODEL_ID = "hf_hub:cm93/resnet18-eurosat"


def build_resnet18_eurosat(pretrained=True):
    """
    Build the exact ResNet-18 architecture used by the published
    EuroSAT fine-tuned checkpoint, then adapt it for GeoShift's
    4-channel B,G,R,NIR input and 6 classes.

    The published EuroSAT model is RGB. GeoShift is B,G,R,NIR, so
    the RGB filters are remapped semantically:
        GeoShift B -> pretrained B filter (RGB index 2)
        GeoShift G -> pretrained G filter (RGB index 1)
        GeoShift R -> pretrained R filter (RGB index 0)
        GeoShift NIR -> mean of pretrained RGB filters
    """
    model = timm.create_model(
        EUROSAT_MODEL_ID,
        pretrained=pretrained,
    )

    old_conv = model.conv1

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    if pretrained:
        with torch.no_grad():
            # Original EuroSAT checkpoint expects RGB.
            # GeoShift input is B,G,R,NIR.
            new_conv.weight[:, 0] = old_conv.weight[:, 2]
            new_conv.weight[:, 1] = old_conv.weight[:, 1]
            new_conv.weight[:, 2] = old_conv.weight[:, 0]
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)

            if new_conv.bias is not None and old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

    model.conv1 = new_conv

    # Replace the EuroSAT 10-class head with GeoShift's 6 classes.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model


model = build_resnet18_eurosat(pretrained=True)
model = model.to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Pretrained source: {EUROSAT_MODEL_ID}")
print(f"Parameters: {total_params:,}")
print("Input:  [batch, 4, 64, 64]")
print("Output: [batch, 6]")
print("GeoShift class head: Forest / Shrubland / Grassland / Cropland / Built-up / Water-Wetland")
print()

# ============================================================
# EMA MODEL
# ============================================================

class ModelEMA:

    def __init__(
        self,
        model,
        decay=0.9997,
    ):

        self.decay = decay

        self.ema = copy.deepcopy(
            model
        ).eval()

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):

        ema_state = (
            self.ema.state_dict()
        )

        model_state = (
            model.state_dict()
        )

        for key in ema_state.keys():

            ema_value = ema_state[key]
            model_value = model_state[key]

            if not ema_value.is_floating_point():

                ema_value.copy_(
                    model_value
                )

            else:

                ema_value.mul_(
                    self.decay
                ).add_(
                    model_value,
                    alpha=1.0 - self.decay
                )


ema = (
    ModelEMA(
        model,
        EMA_DECAY
    )
    if USE_EMA
    else None
)


# ============================================================
# DIFFERENTIAL LEARNING RATE OPTIMIZER
# ============================================================

stem = model.conv1
classifier = model.fc

stem_ids = {
    id(p)
    for p in stem.parameters()
}

classifier_ids = {
    id(p)
    for p in classifier.parameters()
}

parameter_groups = {
    "backbone_decay": [],
    "backbone_no_decay": [],
    "stem_decay": [],
    "stem_no_decay": [],
    "classifier_decay": [],
    "classifier_no_decay": [],
}

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    no_decay = (
        param.ndim == 1
        or name.endswith(".bias")
    )

    if id(param) in stem_ids:

        key = (
            "stem_no_decay"
            if no_decay
            else "stem_decay"
        )

    elif id(param) in classifier_ids:

        key = (
            "classifier_no_decay"
            if no_decay
            else "classifier_decay"
        )

    else:

        key = (
            "backbone_no_decay"
            if no_decay
            else "backbone_decay"
        )

    parameter_groups[key].append(
        param
    )


optimizer = torch.optim.AdamW(
    [
        {
            "params":
                parameter_groups[
                    "backbone_decay"
                ],
            "lr":
                LEARNING_RATE_BACKBONE,
            "weight_decay":
                WEIGHT_DECAY,
        },
        {
            "params":
                parameter_groups[
                    "backbone_no_decay"
                ],
            "lr":
                LEARNING_RATE_BACKBONE,
            "weight_decay": 0.0,
        },
        {
            "params":
                parameter_groups[
                    "stem_decay"
                ],
            "lr":
                LEARNING_RATE_STEM,
            "weight_decay":
                WEIGHT_DECAY,
        },
        {
            "params":
                parameter_groups[
                    "stem_no_decay"
                ],
            "lr":
                LEARNING_RATE_STEM,
            "weight_decay": 0.0,
        },
        {
            "params":
                parameter_groups[
                    "classifier_decay"
                ],
            "lr":
                LEARNING_RATE_CLASSIFIER,
            "weight_decay":
                WEIGHT_DECAY,
        },
        {
            "params":
                parameter_groups[
                    "classifier_no_decay"
                ],
            "lr":
                LEARNING_RATE_CLASSIFIER,
            "weight_decay": 0.0,
        },
    ]
)


# ============================================================
# WARMUP + COSINE SCHEDULER
# ============================================================

def lr_multiplier(epoch):

    if epoch < WARMUP_EPOCHS:

        return (
            0.1
            + 0.9
            * (
                epoch
                / WARMUP_EPOCHS
            )
        )

    progress = (
        epoch - WARMUP_EPOCHS
    ) / max(
        1,
        NUM_EPOCHS - WARMUP_EPOCHS,
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    return (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


scheduler = (
    torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier
    )
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        USE_AMP
        and torch.cuda.is_available()
    )
)


# ============================================================
# MIXUP
# ============================================================

def mixup_batch(
    x,
    y,
    alpha=0.2,
):

    if alpha <= 0:

        return (
            x,
            y,
            y,
            1.0
        )

    lam = np.random.beta(
        alpha,
        alpha
    )

    index = torch.randperm(
        x.size(0),
        device=x.device
    )

    mixed_x = (
        lam * x
        + (1.0 - lam)
        * x[index]
    )

    y_a = y
    y_b = y[index]

    return (
        mixed_x,
        y_a,
        y_b,
        lam
    )


# ============================================================
# TTA TRANSFORMS
# ============================================================

def tta_transforms(x):

    # x: B,C,H,W
    #
    # Eight elements of the square dihedral group:
    # identity, H, V, HV, rot90, rot180, rot270, transpose.

    yield x

    yield torch.flip(
        x,
        dims=[3]
    )

    yield torch.flip(
        x,
        dims=[2]
    )

    yield torch.flip(
        x,
        dims=[2, 3]
    )

    yield torch.rot90(
        x,
        1,
        dims=[2, 3]
    )

    yield torch.rot90(
        x,
        2,
        dims=[2, 3]
    )

    yield torch.rot90(
        x,
        3,
        dims=[2, 3]
    )

    yield x.transpose(
        2,
        3
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    targets = []
    predictions = []

    for x, y in train_loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        do_mixup = (
            USE_MIXUP
            and np.random.rand()
            < MIXUP_PROBABILITY
            and x.size(0) > 1
        )

        if do_mixup:

            (
                x,
                y_a,
                y_b,
                lam,
            ) = mixup_batch(
                x,
                y,
                MIXUP_ALPHA
            )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=(
                USE_AMP
                and torch.cuda.is_available()
            )
        ):

            logits = model(x)

            if do_mixup:

                loss = (
                    lam
                    * criterion(
                        logits,
                        y_a
                    )
                    + (
                        1.0 - lam
                    )
                    * criterion(
                        logits,
                        y_b
                    )
                )

            else:

                loss = criterion(
                    logits,
                    y
                )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP_NORM
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += (
            loss.item()
            * x.size(0)
        )

        # Training metric uses the actual batch
        # target labels, not mixed labels.
        pred = logits.argmax(
            dim=1
        )

        targets.extend(
            y.detach()
            .cpu()
            .numpy()
        )

        predictions.extend(
            pred.detach()
            .cpu()
            .numpy()
        )

    avg_loss = (
        total_loss
        / len(train_dataset)
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    acc = accuracy_score(
        targets,
        predictions
    )

    return (
        avg_loss,
        f1,
        acc
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    eval_model,
    use_tta=True,
):

    eval_model.eval()

    total_loss = 0.0

    targets = []
    predictions = []

    for x, y in val_loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        if use_tta:

            probability_sum = None

            for augmented_x in tta_transforms(x):

                with torch.autocast(
                    device_type=DEVICE.type,
                    enabled=(
                        USE_AMP
                        and torch.cuda.is_available()
                    )
                ):

                    logits = eval_model(
                        augmented_x
                    )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                if probability_sum is None:

                    probability_sum = (
                        probabilities
                    )

                else:

                    probability_sum = (
                        probability_sum
                        + probabilities
                    )

            probabilities = (
                probability_sum
                / 8.0
            )

            logits_for_loss = torch.log(
                probabilities.clamp_min(1e-8)
            )

            loss = F.nll_loss(
                logits_for_loss,
                y
            )

            pred = probabilities.argmax(
                dim=1
            )

        else:

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=(
                    USE_AMP
                    and torch.cuda.is_available()
                )
            ):

                logits = eval_model(x)
                loss = criterion(
                    logits,
                    y
                )

            pred = logits.argmax(
                dim=1
            )

        total_loss += (
            loss.item()
            * x.size(0)
        )

        targets.extend(
            y.cpu().numpy()
        )

        predictions.extend(
            pred.cpu().numpy()
        )

    avg_loss = (
        total_loss
        / len(val_dataset)
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    acc = accuracy_score(
        targets,
        predictions
    )

    return (
        avg_loss,
        f1,
        acc,
        np.asarray(targets),
        np.asarray(predictions)
    )


# ============================================================
# CHECKPOINT / MONITOR
# ============================================================

history = []

best_f1 = -float("inf")
best_epoch = -1

training_start = time.time()


def current_lrs():

    return (
        optimizer.param_groups[0]["lr"],
        optimizer.param_groups[2]["lr"],
        optimizer.param_groups[4]["lr"],
    )


def save_best_checkpoint(
    epoch,
    val_f1,
    val_acc,
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "val_macro_f1":
            val_f1,

        "val_accuracy":
            val_acc,

        "channel_mean":
            channel_mean,

        "channel_std":
            channel_std,

        "class_names":
            CLASS_NAMES,

        "random_seed":
            RANDOM_SEED,

        "train_size":
            TRAIN_SIZE,

        "val_size":
            VAL_SIZE,

        "internal_test_size":
            INTERNAL_TEST_SIZE,

        "outer_sgkf_folds":
            best_outer["folds"],

        "inner_sgkf_fold":
            selected_inner["fold"],

        "config": {
            "num_epochs":
                NUM_EPOCHS,

            "warmup_epochs":
                WARMUP_EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "lr_backbone":
                LEARNING_RATE_BACKBONE,

            "lr_stem":
                LEARNING_RATE_STEM,

            "lr_classifier":
                LEARNING_RATE_CLASSIFIER,

            "weight_decay":
                WEIGHT_DECAY,

            "label_smoothing":
                LABEL_SMOOTHING,

            "use_mixup":
                USE_MIXUP,

            "mixup_probability":
                MIXUP_PROBABILITY,

            "mixup_alpha":
                MIXUP_ALPHA,

            "ema_decay":
                EMA_DECAY,

            "validation_tta":
                USE_VALIDATION_TTA,
        }
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH
    )

    if ema is not None:

        torch.save(
            {
                **checkpoint,
                "model_state_dict":
                    ema.ema.state_dict(),
                "is_ema":
                    True,
            },
            BEST_EMA_MODEL_PATH
        )


def print_monitor(
    epoch,
    train_loss,
    train_f1,
    train_acc,
    val_loss,
    val_f1,
    val_acc,
    epoch_time,
):

    lr_b, lr_s, lr_c = current_lrs()

    elapsed = (
        time.time()
        - training_start
    )

    print()
    print("-" * 80)
    print(
        f"MONITOR — Epoch "
        f"{epoch:03d}/{NUM_EPOCHS}"
    )
    print("-" * 80)

    print(
        f"Train F1:       {train_f1:.5f}"
    )

    print(
        f"Val F1:         {val_f1:.5f}"
    )

    print(
        f"Best Val F1:    {best_f1:.5f}"
    )

    print(
        f"Train Acc:      {train_acc:.5f}"
    )

    print(
        f"Val Acc:        {val_acc:.5f}"
    )

    print(
        f"Train Loss:     {train_loss:.5f}"
    )

    print(
        f"Val Loss:       {val_loss:.5f}"
    )

    print(
        f"F1 gap:         "
        f"{train_f1 - val_f1:+.5f}"
    )

    print(
        f"LR backbone:    {lr_b:.3e}"
    )

    print(
        f"LR stem:        {lr_s:.3e}"
    )

    print(
        f"LR classifier:  {lr_c:.3e}"
    )

    print(
        f"Epoch time:     {epoch_time:.1f}s"
    )

    print(
        f"Elapsed:        "
        f"{elapsed / 60:.1f} min"
    )

    print("-" * 80)


# ============================================================
# TRAIN — NO EARLY STOPPING
# ============================================================

print("=" * 80)
print("STARTING FULL TRAINING")
print("=" * 80)
print()
print(
    f"Training will run for ALL "
    f"{NUM_EPOCHS} epochs."
)
print(
    "Early stopping is DISABLED."
)
print(
    "Every new validation Macro-F1 high "
    "will replace the best checkpoint."
)
print()

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    epoch_start = time.time()

    (
        train_loss,
        train_f1,
        train_acc
    ) = train_one_epoch()

    # First evaluate the EMA model with TTA.
    # EMA is generally smoother than the raw model.
    evaluation_model = (
        ema.ema
        if ema is not None
        else model
    )

    (
        val_loss,
        val_f1,
        val_acc,
        val_targets,
        val_predictions
    ) = validate(
        evaluation_model,
        use_tta=USE_VALIDATION_TTA
    )

    scheduler.step()

    (
        lr_backbone,
        lr_stem,
        lr_classifier
    ) = current_lrs()

    epoch_time = (
        time.time()
        - epoch_start
    )

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_f1": train_f1,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_accuracy": val_acc,
        "lr_backbone": lr_backbone,
        "lr_stem": lr_stem,
        "lr_classifier": lr_classifier,
        "epoch_time_sec": epoch_time,
    }

    history.append(row)

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False
    )

    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
        f"Train F1 {train_f1:.5f} | "
        f"Val F1 {val_f1:.5f} | "
        f"Val Acc {val_acc:.5f} | "
        f"{epoch_time:.1f}s"
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_f1 > (
        best_f1 + MIN_DELTA
    ):

        best_f1 = val_f1
        best_epoch = epoch

        save_best_checkpoint(
            epoch,
            val_f1,
            val_acc
        )

        print(
            f"  ★ NEW BEST VAL MACRO-F1: "
            f"{best_f1:.6f}"
        )

    # --------------------------------------------------------
    # PERIODIC MONITOR
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % MONITOR_EVERY == 0
        or val_f1 >= best_f1
    ):

        print_monitor(
            epoch,
            train_loss,
            train_f1,
            train_acc,
            val_loss,
            val_f1,
            val_acc,
            epoch_time
        )


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print()
print("=" * 80)
print("FULL TRAINING COMPLETE")
print("=" * 80)

print(
    f"Best validation Macro-F1: "
    f"{best_f1:.6f}"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Total time: "
    f"{(time.time() - training_start) / 60:.1f} min"
)

print()
print("Loading best EMA checkpoint...")


if (
    USE_EMA
    and BEST_EMA_MODEL_PATH.exists()
):

    best_checkpoint = torch.load(
        BEST_EMA_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

else:

    best_checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )


best_model = build_resnet18_eurosat(pretrained=False)

best_model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

best_model = best_model.to(DEVICE)
best_model.eval()


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("FINAL BEST-MODEL VALIDATION")
print("=" * 80)

(
    final_val_loss,
    final_val_f1,
    final_val_acc,
    final_targets,
    final_predictions
) = validate(
    best_model,
    use_tta=USE_VALIDATION_TTA
)

print(
    f"Macro-F1: {final_val_f1:.6f}"
)

print(
    f"Accuracy: {final_val_acc:.6f}"
)

print()
print("Classification report:")

report = classification_report(
    final_targets,
    final_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    digits=5,
    zero_division=0,
)

print(report)

Path(
    REPORT_PATH
).write_text(
    report,
    encoding="utf-8"
)

cm = confusion_matrix(
    final_targets,
    final_predictions,
    labels=np.arange(NUM_CLASSES)
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(
    CONFUSION_PATH
)


# ============================================================
# FINAL INTERNAL-TEST EVALUATION
# ============================================================
# This is the held-out 15% that came from train_images.npy.
# It was never used for gradient updates or model selection.

print()
print("=" * 80)
print("FINAL INTERNAL-TEST EVALUATION")
print("=" * 80)


def evaluate_internal_test(
    eval_model,
    use_tta=True,
):
    """Evaluate the frozen best model on the labeled 15% holdout."""

    eval_model.eval()

    targets = []
    predictions = []
    probabilities_all = []

    for x, y in internal_test_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        if use_tta:
            probability_sum = None

            for augmented_x in tta_transforms(x):
                with torch.autocast(
                    device_type=DEVICE.type,
                    enabled=(
                        USE_AMP
                        and torch.cuda.is_available()
                    ),
                ):
                    logits = eval_model(augmented_x)

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                if probability_sum is None:
                    probability_sum = probabilities
                else:
                    probability_sum += probabilities

            probabilities = probability_sum / 8.0

        else:
            with torch.autocast(
                device_type=DEVICE.type,
                enabled=(
                    USE_AMP
                    and torch.cuda.is_available()
                ),
            ):
                logits = eval_model(x)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        pred = probabilities.argmax(dim=1)

        targets.extend(y.numpy())
        predictions.extend(pred.cpu().numpy())
        probabilities_all.append(
            probabilities.cpu().numpy()
        )

    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities_all = np.concatenate(
        probabilities_all,
        axis=0,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    confidence = probabilities_all.max(axis=1)

    return (
        macro_f1,
        accuracy,
        targets,
        predictions,
        confidence,
    )


(
    internal_test_f1,
    internal_test_acc,
    internal_test_targets,
    internal_test_predictions,
    internal_test_confidences,
) = evaluate_internal_test(
    best_model,
    use_tta=USE_TEST_TTA,
)

print(
    f"Internal-test Macro-F1: {internal_test_f1:.6f}"
)
print(
    f"Internal-test Accuracy: {internal_test_acc:.6f}"
)
print(
    f"Mean confidence:        {internal_test_confidences.mean():.6f}"
)

internal_report = classification_report(
    internal_test_targets,
    internal_test_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    digits=5,
    zero_division=0,
)

print()
print("Internal-test classification report:")
print(internal_report)

INTERNAL_REPORT_PATH = DATA_DIR / "strong_internal_test_classification_report.txt"
INTERNAL_CONFUSION_PATH = DATA_DIR / "strong_internal_test_confusion_matrix.csv"

INTERNAL_REPORT_PATH.write_text(
    internal_report,
    encoding="utf-8",
)

internal_cm = confusion_matrix(
    internal_test_targets,
    internal_test_predictions,
    labels=np.arange(NUM_CLASSES),
)

pd.DataFrame(
    internal_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(
    INTERNAL_CONFUSION_PATH
)

# Save the exact split indices so the experiment is reproducible.
np.savez(
    DATA_DIR / "strong_701515_split_indices.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    internal_test_idx=internal_test_idx,
)


# ============================================================
# FINAL COMPETITION TEST PREDICTION
# ============================================================
# This is the actual unlabeled test_images.npy supplied by the
# competition. It is used only after the best model has been selected
# and after the internal labeled test has been evaluated.

print()
print("=" * 80)
print("PREDICTING COMPETITION TEST SET")
print("=" * 80)


def predict_test(
    eval_model,
    use_tta=True,
):

    eval_model.eval()

    predictions = []
    confidences = []

    for x in competition_test_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        if use_tta:

            probability_sum = None

            for augmented_x in tta_transforms(x):

                with torch.autocast(
                    device_type=DEVICE.type,
                    enabled=(
                        USE_AMP
                        and torch.cuda.is_available()
                    ),
                ):

                    logits = eval_model(
                        augmented_x
                    )

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                if probability_sum is None:
                    probability_sum = probabilities
                else:
                    probability_sum += probabilities

            probabilities = (
                probability_sum / 8.0
            )

        else:

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=(
                    USE_AMP
                    and torch.cuda.is_available()
                ),
            ):

                logits = eval_model(x)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        confidence, pred = probabilities.max(
            dim=1
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        confidences.extend(
            confidence.cpu().numpy()
        )

    return (
        np.asarray(
            predictions,
            dtype=np.int64,
        ),
        np.asarray(
            confidences,
            dtype=np.float32,
        ),
    )


test_predictions, test_confidences = predict_test(
    best_model,
    use_tta=USE_TEST_TTA,
)

print(
    f"Competition-test predictions: {len(test_predictions)}"
)
print(
    f"Mean confidence:              {test_confidences.mean():.6f}"
)


# ============================================================
# COMPETITION TEST DISTRIBUTION
# ============================================================

print()
print("Competition-test prediction distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    count = int(
        np.sum(
            test_predictions == class_id
        )
    )

    percentage = (
        100.0
        * count
        / len(test_predictions)
    )

    print(
        f"{class_id} | "
        f"{class_name:15s} | "
        f"{count:7d} | "
        f"{percentage:6.2f}%"
    )


# ============================================================
# CREATE SUBMISSION
# ============================================================

if len(test_df) != len(
    test_predictions
):

    raise ValueError(
        "test.csv and predictions have "
        "different row counts."
    )

sample_ids = (
    sample_submission["Id"].to_numpy()
    if "Id" in sample_submission.columns
    else None
)

test_ids = test_df[
    "Id"
].to_numpy()

if sample_ids is not None:

    if not np.array_equal(
        sample_ids,
        test_ids
    ):

        raise ValueError(
            "test.csv Id order does not match "
            "sample_submission.csv."
        )

submission = pd.DataFrame(
    {
        "Id": test_ids,
        "label": test_predictions,
    }
)

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)

print(
    f"Best validation F1: "
    f"{best_f1:.6f}"
)

print(
    f"Internal-test Macro-F1: "
    f"{internal_test_f1:.6f}"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best model: {BEST_MODEL_PATH}"
)

if USE_EMA:
    print(
        f"Best EMA model: "
        f"{BEST_EMA_MODEL_PATH}"
    )

print(
    f"History: {HISTORY_PATH}"
)

print(
    f"Report: {REPORT_PATH}"
)

print(
    f"Confusion matrix: {CONFUSION_PATH}"
)

print(
    f"Submission: {SUBMISSION_PATH}"
)

print()
print(
    "Training was completed for the full "
    f"{NUM_EPOCHS} epochs with NO early stopping."
)
