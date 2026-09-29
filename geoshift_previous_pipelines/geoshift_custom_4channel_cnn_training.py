# ============================================================
# GeoShift - Custom 4-Channel CNN Training Pipeline
#
# Model:
#   Stage 1: Local spatial extraction
#   Stage 2: Fourier + spectral channel attention
#   Stage 3: Class-conditioned dynamic convolution
#   Stage 4: Global average pooling + classifier
#
# Training protocol:
#   - 4-channel B, G, R, NIR input
#   - Region-aware StratifiedGroupKFold split
#   - ~20% validation fold
#   - Train-only channel normalization
#   - Random H/V flips + 90-degree rotations
#   - CrossEntropyLoss (optionally class-weighted)
#   - Differential learning rates
#   - Warmup + cosine decay
#   - Macro-F1 model selection
#   - Early stopping
#   - Mixed precision for NVIDIA T4
#   - Best checkpoint + validation report + confusion matrix
#   - Test prediction + Kaggle submission
# ============================================================

import math
import time
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

# CHANGE THIS PATH TO YOUR DATASET LOCATION.
# Example for Google Drive:
# DATA_DIR = Path("/content/drive/MyDrive/GeoShift/data")
#
# Example for files copied into Colab:
# DATA_DIR = Path("/content/GeoShift/data")
DATA_DIR = Path(r"C:\Users\badri\OneDrive\Desktop\GIS-intra-iit\GeoShift\data")

TRAIN_CSV_PATH = DATA_DIR / "train.csv"
TEST_CSV_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

TRAIN_IMAGES_PATH = DATA_DIR / "train_images.npy"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.npy"
TEST_IMAGES_PATH = DATA_DIR / "test_images.npy"

BEST_MODEL_PATH = DATA_DIR / "best_geoshift_custom_cnn.pth"
HISTORY_PATH = DATA_DIR / "custom_cnn_training_history.csv"
REPORT_PATH = DATA_DIR / "custom_cnn_classification_report.txt"
CONFUSION_PATH = DATA_DIR / "custom_cnn_confusion_matrix.csv"
SUBMISSION_PATH = DATA_DIR / "submission_custom_cnn.csv"


# Competition
NUM_CLASSES = 6

CLASS_NAMES = [
    "Forest",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Water/Wetland",
]

VAL_SIZE = 0.20
RANDOM_SEED = 42


# Training
NUM_EPOCHS = 150
WARMUP_EPOCHS = 10
PATIENCE = 25

# T4-friendly settings
BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = True
PERSISTENT_WORKERS = False

# Differential learning rates
# These are based on the optimizer configuration supplied
# with the custom model.
BASE_LR = 1e-3
DECAY_FACTOR = 0.75

LEARNING_RATE_STAGE1 = BASE_LR * (DECAY_FACTOR ** 3)
LEARNING_RATE_STAGE2 = BASE_LR * (DECAY_FACTOR ** 2)
LEARNING_RATE_STAGE3 = BASE_LR * DECAY_FACTOR
LEARNING_RATE_CLASSIFIER = BASE_LR

WEIGHT_DECAY = 1e-4

# Keep False if you want the exact standard CrossEntropyLoss.
# Set True to use inverse-frequency class-weighted CE.
USE_CLASS_WEIGHTED_LOSS = False

# Mixed precision
USE_AMP = True

# Monitoring
MIN_DELTA = 1e-4
MONITOR_EVERY = 5
TREND_WINDOW = 5


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_SEED)


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 80)
print("GeoShift - Custom 4-Channel CNN")
print("=" * 80)
print(f"Device       : {DEVICE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Num workers  : {NUM_WORKERS}")
print(f"AMP enabled  : {AMP_ENABLED}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"CUDA         : {torch.version.cuda}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Good setting for CUDA matrix operations.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

train_images = np.load(TRAIN_IMAGES_PATH)
train_labels = np.load(TRAIN_LABELS_PATH)
test_images = np.load(TEST_IMAGES_PATH)

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

# sample_submission is not required for construction because
# the competition submission format is simply Id + label.
if SAMPLE_SUBMISSION_PATH.exists():
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
else:
    sample_submission = None

print(f"Train images : {train_images.shape}")
print(f"Train labels : {train_labels.shape}")
print(f"Test images  : {test_images.shape}")

if train_images.ndim != 4:
    raise ValueError("Expected train_images with shape (N, 4, H, W).")

if train_images.shape[1] != 4:
    raise ValueError("Expected exactly 4 channels: B, G, R, NIR.")

if len(train_images) != len(train_labels):
    raise ValueError("Number of training images and labels differs.")

if len(train_df) != len(train_images):
    raise ValueError("train.csv rows != number of training images.")

if "Id" not in train_df.columns:
    raise ValueError("train.csv must contain an 'Id' column.")

if "Id" not in test_df.columns:
    raise ValueError("test.csv must contain an 'Id' column.")

train_labels = train_labels.astype(np.int64)

# Extract region from IDs exactly as in the reference pipeline.
train_df["Region"] = train_df["Id"].astype(str).str.extract(r"(R\d{2})")

if train_df["Region"].isna().any():
    raise ValueError(
        "Could not extract Region from one or more training IDs. "
        "Expected an ID pattern containing R##."
    )


# ============================================================
# REGION-AWARE TRAIN / VALIDATION SPLIT
# ============================================================

regions = train_df["Region"].astype(str).to_numpy()

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED,
)

candidate_splits = []

for fold, (tr_idx, va_idx) in enumerate(
    sgkf.split(
        train_images,
        train_labels,
        groups=regions,
    )
):
    candidate_splits.append(
        {
            "fold": fold,
            "train_idx": tr_idx,
            "val_idx": va_idx,
            "val_fraction": len(va_idx) / len(train_images),
        }
    )

# Select the fold closest to 20% validation.
selected = min(
    candidate_splits,
    key=lambda x: abs(x["val_fraction"] - VAL_SIZE),
)

train_idx = selected["train_idx"]
val_idx = selected["val_idx"]

train_x = train_images[train_idx]
train_y = train_labels[train_idx]

val_x = train_images[val_idx]
val_y = train_labels[val_idx]

train_regions = set(regions[train_idx])
val_regions = set(regions[val_idx])

overlap = train_regions & val_regions

if overlap:
    raise RuntimeError(
        f"REGION LEAKAGE DETECTED: {overlap}"
    )

print("\n" + "=" * 80)
print("REGION-AWARE SPLIT")
print("=" * 80)
print(f"Selected fold       : {selected['fold']}")
print(f"Train samples       : {len(train_x)}")
print(f"Validation samples  : {len(val_x)}")
print(f"Train regions       : {len(train_regions)}")
print(f"Validation regions  : {len(val_regions)}")
print(f"Region overlap      : {len(overlap)}")
print(f"Validation fraction : {len(val_x) / len(train_images):.3f}")

print("\nTrain class counts:")
print(np.bincount(train_y, minlength=NUM_CLASSES))

print("\nValidation class counts:")
print(np.bincount(val_y, minlength=NUM_CLASSES))


# ============================================================
# NORMALIZATION
# ============================================================

# IMPORTANT:
# Statistics are calculated ONLY from the training split.
channel_mean = (
    train_x.astype(np.float32).mean(axis=(0, 2, 3)) / 255.0
)

channel_std = (
    train_x.astype(np.float32).std(axis=(0, 2, 3)) / 255.0
)

channel_std = np.maximum(channel_std, 1e-6)

print("\nChannel normalization:")
for i, name in enumerate(["B", "G", "R", "NIR"]):
    print(
        f"  {name:>3s}: "
        f"mean={channel_mean[i]:.6f}, "
        f"std={channel_std[i]:.6f}"
    )


# ============================================================
# DATASET
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(
        self,
        images,
        labels=None,
        mean=None,
        std=None,
        augment=False,
    ):
        self.images = images
        self.labels = labels
        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        ).reshape(4, 1, 1)

        self.std = np.asarray(
            std,
            dtype=np.float32,
        ).reshape(4, 1, 1)

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        x = self.images[idx].astype(
            np.float32
        ) / 255.0

        if self.augment:

            # Horizontal flip
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]

            # Vertical flip
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]

            # Random 90-degree rotation
            k = np.random.randint(0, 4)

            if k:
                x = np.rot90(
                    x,
                    k=k,
                    axes=(1, 2),
                ).copy()

        x = (x - self.mean) / self.std

        x = torch.from_numpy(
            x.copy()
        ).float()

        if self.labels is None:
            return x

        return (
            x,
            torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        )


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

test_dataset = SatelliteDataset(
    test_images,
    None,
    channel_mean,
    channel_std,
    augment=False,
)


# persistent_workers requires num_workers > 0.
persistent = (
    PERSISTENT_WORKERS and NUM_WORKERS > 0
)

loader_kwargs = {
    "num_workers": NUM_WORKERS,
    "pin_memory": (
        PIN_MEMORY and torch.cuda.is_available()
    ),
}

if persistent:
    loader_kwargs["persistent_workers"] = True


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    **loader_kwargs,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **loader_kwargs,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **loader_kwargs,
)


# ============================================================
# MODEL
# ============================================================

class FourierSpectralAttention(nn.Module):
    """
    Frequency-domain filtering combined with
    inter-band spectral channel attention.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels

        # Band/channel spectral attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                channels,
                channels // 2,
            ),
            nn.GELU(),
            nn.Linear(
                channels // 2,
                channels,
            ),
            nn.Sigmoid(),
        )

        # Complex frequency-domain weights.
        # Shape is broadcast over batch, frequency height,
        # and frequency width.
        self.complex_weight = nn.Parameter(
            torch.randn(
                channels,
                1,
                1,
                2,
                dtype=torch.float32,
            ) * 0.02
        )

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape

        # ----------------------------------------------------
        # 1. Spectral/channel attention
        # ----------------------------------------------------
        channel_weights = self.se(x).view(
            B,
            C,
            1,
            1,
        )

        x_att = x * channel_weights

        # ----------------------------------------------------
        # 2. Fourier transformation
        # ----------------------------------------------------
        x_fft = torch.fft.rfft2(
            x_att,
            norm="ortho",
        )

        x_fft_real_imag = torch.view_as_real(
            x_fft
        )

        # ----------------------------------------------------
        # 3. Complex learnable filtering
        # ----------------------------------------------------
        weight = self.complex_weight

        real = x_fft_real_imag[..., 0]
        imag = x_fft_real_imag[..., 1]

        filtered_real = (
            real * weight[..., 0]
            - imag * weight[..., 1]
        )

        filtered_imag = (
            real * weight[..., 1]
            + imag * weight[..., 0]
        )

        x_fft_filtered = torch.stack(
            [
                filtered_real,
                filtered_imag,
            ],
            dim=-1,
        )

        x_out_fft = torch.view_as_complex(
            x_fft_filtered
        )

        x_spatial = torch.fft.irfft2(
            x_out_fft,
            s=(H, W),
            norm="ortho",
        )

        # Residual connection
        return x_spatial + x


class DynamicClassRouter(nn.Module):
    """
    Class-conditioned dynamic convolution.

    A six-way router predicts a mixture over six class-specific
    convolution kernels for each sample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int = 6,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                in_channels,
                num_classes,
            ),
            nn.Softmax(dim=-1),
        )

        self.class_weights = nn.Parameter(
            torch.randn(
                num_classes,
                out_channels,
                in_channels,
                3,
                3,
            ) * 0.02
        )

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape

        routing_weights = self.router(x)

        # Blend the class-specific kernels for every sample.
        dynamic_w = torch.einsum(
            "bk,koihw->boihw",
            routing_weights,
            self.class_weights,
        )

        # Each sample has its own convolution kernel.
        # Grouped convolution lets us perform the complete
        # batch without a Python loop over samples.
        x_grouped = x.reshape(
            1,
            B * C,
            H,
            W,
        )

        w_grouped = dynamic_w.reshape(
            B * self.out_channels,
            C,
            3,
            3,
        )

        out = F.conv2d(
            x_grouped,
            w_grouped,
            padding=1,
            groups=B,
        )

        out = out.reshape(
            B,
            self.out_channels,
            H,
            W,
        )

        return out


class Custom4ChannelCNN(nn.Module):
    """
    Four-stage multispectral CNN.

    Input:
        [B, 4, 64, 64]

    Output:
        [B, 6]
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 6,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Stage 1: local spatial extraction
        # ----------------------------------------------------
        self.stage1_conv = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            padding=1,
        )

        self.stage1_bn = nn.BatchNorm2d(32)
        self.stage1_act = nn.ReLU()
        self.stage1_pool = nn.MaxPool2d(2, 2)

        # 64x64 -> 32x32

        # ----------------------------------------------------
        # Stage 2: Fourier + spectral attention
        # ----------------------------------------------------
        self.stage2_fourier = FourierSpectralAttention(32)

        self.stage2_conv = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1,
        )

        self.stage2_bn = nn.BatchNorm2d(64)
        self.stage2_act = nn.GELU()
        self.stage2_pool = nn.MaxPool2d(2, 2)

        # 32x32 -> 16x16

        # ----------------------------------------------------
        # Stage 3: class-aware dynamic abstraction
        # ----------------------------------------------------
        self.stage3_router = DynamicClassRouter(
            64,
            128,
            num_classes=num_classes,
        )

        self.stage3_bn = nn.BatchNorm2d(128)
        self.stage3_act = nn.SiLU()
        self.stage3_pool = nn.MaxPool2d(2, 2)

        # 16x16 -> 8x8

        # ----------------------------------------------------
        # Stage 4: classifier
        # ----------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            128,
            num_classes,
        )

    def forward(self, x: torch.Tensor):

        # Stage 1
        x = self.stage1_conv(x)
        x = self.stage1_bn(x)
        x = self.stage1_act(x)
        x = self.stage1_pool(x)

        # Stage 2
        x = self.stage2_fourier(x)
        x = self.stage2_conv(x)
        x = self.stage2_bn(x)
        x = self.stage2_act(x)
        x = self.stage2_pool(x)

        # Stage 3
        x = self.stage3_router(x)
        x = self.stage3_bn(x)
        x = self.stage3_act(x)
        x = self.stage3_pool(x)

        # Stage 4
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        logits = self.classifier(x)

        return logits


model = Custom4ChannelCNN(
    in_channels=4,
    num_classes=NUM_CLASSES,
).to(DEVICE)


# ============================================================
# MODEL SANITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("MODEL")
print("=" * 80)

sample_input = torch.randn(
    2,
    4,
    64,
    64,
    device=DEVICE,
)

with torch.no_grad():
    sample_output = model(sample_input)

print(f"Input shape  : {tuple(sample_input.shape)}")
print(f"Output shape : {tuple(sample_output.shape)}")
print(
    f"Parameters   : "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

if tuple(sample_output.shape) != (2, NUM_CLASSES):
    raise RuntimeError(
        "Model sanity check failed."
    )

del sample_input, sample_output

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# DIFFERENTIAL LEARNING RATE OPTIMIZER
# ============================================================

param_groups = [
    {
        "params": [],
        "lr": LEARNING_RATE_STAGE1,
        "weight_decay": WEIGHT_DECAY,
        "name": "stage1",
    },
    {
        "params": [],
        "lr": LEARNING_RATE_STAGE2,
        "weight_decay": WEIGHT_DECAY,
        "name": "stage2",
    },
    {
        "params": [],
        "lr": LEARNING_RATE_STAGE3,
        "weight_decay": WEIGHT_DECAY,
        "name": "stage3",
    },
    {
        "params": [],
        "lr": LEARNING_RATE_CLASSIFIER,
        "weight_decay": WEIGHT_DECAY,
        "name": "classifier",
    },
]

for name, param in model.named_parameters():

    if name.startswith("stage1"):
        param_groups[0]["params"].append(param)

    elif name.startswith("stage2"):
        param_groups[1]["params"].append(param)

    elif name.startswith("stage3"):
        param_groups[2]["params"].append(param)

    elif (
        name.startswith("classifier")
        or name.startswith("global_pool")
    ):
        param_groups[3]["params"].append(param)

    else:
        raise RuntimeError(
            f"Unassigned model parameter: {name}"
        )


optimizer = torch.optim.AdamW(
    param_groups,
    weight_decay=0.0,
)


# ============================================================
# WARMUP + COSINE LR SCHEDULER
# ============================================================

def lr_lambda(epoch):

    if epoch < WARMUP_EPOCHS:
        return 0.1 + 0.9 * (
            epoch / WARMUP_EPOCHS
        )

    progress = (
        epoch - WARMUP_EPOCHS
    ) / max(
        1,
        NUM_EPOCHS - WARMUP_EPOCHS,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return 0.5 * (
        1.0 + math.cos(
            math.pi * progress
        )
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda,
)


# ============================================================
# LOSS
# ============================================================

if USE_CLASS_WEIGHTED_LOSS:

    class_counts = np.bincount(
        train_y,
        minlength=NUM_CLASSES,
    ).astype(np.float32)

    if np.any(class_counts <= 0):
        raise ValueError(
            "Cannot calculate class weights because "
            "at least one class has zero training samples."
        )

    # Inverse-frequency weighting:
    # weight_k = N / (K * count_k)
    class_weights = (
        len(train_y)
        / (
            NUM_CLASSES
            * class_counts
        )
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    print("\nLoss: CLASS-WEIGHTED CrossEntropyLoss")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:15s}: "
            f"count={int(class_counts[i]):6d}, "
            f"weight={class_weights[i]:.4f}"
        )

else:

    criterion = nn.CrossEntropyLoss()

    print("\nLoss: standard CrossEntropyLoss")


# ============================================================
# AMP SCALER
# ============================================================

if AMP_ENABLED:
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )
else:
    scaler = None


# ============================================================
# TRAINING / VALIDATION FUNCTIONS
# ============================================================

def run_train_epoch():

    model.train()

    running_loss = 0.0

    targets = []
    predictions = []

    for x, y in train_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):

            logits = model(x)
            loss = criterion(
                logits,
                y,
            )

        if AMP_ENABLED:

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()
            optimizer.step()

        running_loss += (
            loss.item()
            * x.size(0)
        )

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

    epoch_loss = (
        running_loss
        / len(train_dataset)
    )

    epoch_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    epoch_acc = accuracy_score(
        targets,
        predictions,
    )

    return (
        epoch_loss,
        epoch_f1,
        epoch_acc,
    )


@torch.no_grad()
def run_validation():

    model.eval()

    running_loss = 0.0

    targets = []
    predictions = []

    for x, y in val_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):

            logits = model(x)
            loss = criterion(
                logits,
                y,
            )

        running_loss += (
            loss.item()
            * x.size(0)
        )

        pred = logits.argmax(
            dim=1
        )

        targets.extend(
            y.cpu().numpy()
        )

        predictions.extend(
            pred.cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(val_dataset)
    )

    epoch_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    epoch_acc = accuracy_score(
        targets,
        predictions,
    )

    return (
        epoch_loss,
        epoch_f1,
        epoch_acc,
        np.asarray(targets),
        np.asarray(predictions),
    )


# ============================================================
# MONITORING
# ============================================================

history = []

best_f1 = -float("inf")
best_epoch = -1
epochs_without_improvement = 0

training_start = time.time()


def get_lrs():

    return {
        "stage1": optimizer.param_groups[0]["lr"],
        "stage2": optimizer.param_groups[1]["lr"],
        "stage3": optimizer.param_groups[2]["lr"],
        "classifier": optimizer.param_groups[3]["lr"],
    }


def trend_report():

    if len(history) < 2:
        return "WARMING UP"

    recent = history[
        -min(
            TREND_WINDOW,
            len(history),
        ):
    ]

    current = recent[-1]["val_f1"]

    if len(recent) >= 3:

        xs = np.arange(
            len(recent)
        )

        ys = np.array(
            [
                row["val_f1"]
                for row in recent
            ]
        )

        slope = np.polyfit(
            xs,
            ys,
            1,
        )[0]

    else:
        slope = 0.0

    train_f1 = recent[-1]["train_f1"]
    gap = train_f1 - current

    if (
        gap > 0.15
        and slope < MIN_DELTA
    ):
        return "OVERFITTING"

    if slope > MIN_DELTA:
        return "IMPROVING"

    if abs(slope) <= MIN_DELTA:
        return "PLATEAU"

    return "SLOWING"


def print_monitor(epoch):

    current = history[-1]
    lrs = get_lrs()

    elapsed = (
        time.time()
        - training_start
    )

    print("\n" + "=" * 80)
    print(
        f"TRAINING MONITOR — "
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )
    print("=" * 80)

    print(
        f"Best Epoch            : "
        f"{best_epoch}"
    )

    print(
        f"Best Validation F1    : "
        f"{best_f1:.4f}"
    )

    print(
        f"Current Validation F1 : "
        f"{current['val_f1']:.4f}"
    )

    print(
        f"Current Train F1      : "
        f"{current['train_f1']:.4f}"
    )

    print(
        f"Generalization Gap    : "
        f"{current['train_f1'] - current['val_f1']:.4f}"
    )

    print(
        f"\nTrain Loss            : "
        f"{current['train_loss']:.4f}"
    )

    print(
        f"Validation Loss       : "
        f"{current['val_loss']:.4f}"
    )

    print(
        f"Train Accuracy        : "
        f"{current['train_acc']:.4f}"
    )

    print(
        f"Validation Accuracy   : "
        f"{current['val_acc']:.4f}"
    )

    print("\nLearning rates:")
    print(
        f"  Stage 1             : "
        f"{lrs['stage1']:.3e}"
    )
    print(
        f"  Stage 2             : "
        f"{lrs['stage2']:.3e}"
    )
    print(
        f"  Stage 3             : "
        f"{lrs['stage3']:.3e}"
    )
    print(
        f"  Classifier          : "
        f"{lrs['classifier']:.3e}"
    )

    print(
        f"\nStatus                : "
        f"{trend_report()}"
    )

    print(
        f"Elapsed time          : "
        f"{elapsed / 60:.1f} min"
    )

    print("=" * 80)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    epoch_start = time.time()

    (
        train_loss,
        train_f1,
        train_acc,
    ) = run_train_epoch()

    (
        val_loss,
        val_f1,
        val_acc,
        val_targets,
        val_predictions,
    ) = run_validation()

    # Update LR after completing this epoch.
    scheduler.step()

    lrs = get_lrs()

    epoch_time = (
        time.time()
        - epoch_start
    )

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_f1": train_f1,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_acc": val_acc,
        "lr_stage1": lrs["stage1"],
        "lr_stage2": lrs["stage2"],
        "lr_stage3": lrs["stage3"],
        "lr_classifier": lrs["classifier"],
        "epoch_time_sec": epoch_time,
    }

    history.append(row)

    # Continuously save history.
    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False,
    )

    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
        f"Train F1: {train_f1:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"{epoch_time:.1f}s"
    )

    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    if val_f1 > best_f1 + MIN_DELTA:

        best_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "channel_mean": channel_mean,
                "channel_std": channel_std,
                "class_names": CLASS_NAMES,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "use_amp": USE_AMP,
                    "use_class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
                    "random_seed": RANDOM_SEED,
                },
            },
            BEST_MODEL_PATH,
        )

        print(
            f"  ✓ NEW BEST Macro-F1: "
            f"{best_f1:.4f}"
        )

    else:

        epochs_without_improvement += 1

    # Periodic monitor
    if (
        epoch == 1
        or epoch % MONITOR_EVERY == 0
        or val_f1 >= best_f1
    ):
        print_monitor(epoch)

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print("\n" + "=" * 80)
        print("EARLY STOPPING")
        print("=" * 80)
        print(
            f"No meaningful validation Macro-F1 "
            f"improvement for {PATIENCE} epochs."
        )
        print(
            f"Best Macro-F1 : {best_f1:.4f}"
        )
        print(
            f"Best Epoch    : {best_epoch}"
        )
        print("=" * 80)

        break


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_f1 = checkpoint["best_f1"]

print(
    f"Best validation Macro-F1: "
    f"{best_f1:.4f}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

(
    val_loss,
    val_f1,
    val_acc,
    val_targets,
    val_predictions,
) = run_validation()

print("\n" + "=" * 80)
print("BEST MODEL VALIDATION")
print("=" * 80)

print(
    f"Macro-F1 : {val_f1:.4f}"
)

print(
    f"Accuracy : {val_acc:.4f}"
)

report = classification_report(
    val_targets,
    val_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\nClassification Report:")
print(report)

Path(REPORT_PATH).write_text(
    report,
    encoding="utf-8",
)

cm = confusion_matrix(
    val_targets,
    val_predictions,
    labels=np.arange(NUM_CLASSES),
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

cm_df.to_csv(
    CONFUSION_PATH
)

print("\nConfusion Matrix:")
print(cm_df)


# ============================================================
# TEST PREDICTION
# ============================================================

print("\nGenerating test predictions...")

model.eval()

test_predictions = []

with torch.no_grad():

    for x in test_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):

            logits = model(x)

        pred = logits.argmax(
            dim=1
        )

        test_predictions.extend(
            pred.cpu()
            .numpy()
            .tolist()
        )


test_predictions = np.asarray(
    test_predictions,
    dtype=np.int64,
)

if len(test_predictions) != len(test_df):
    raise ValueError(
        "Number of predictions != test.csv rows."
    )


# ============================================================
# SUBMISSION
# ============================================================

submission = pd.DataFrame(
    {
        "Id": test_df["Id"],
        "label": test_predictions,
    }
)

submission.to_csv(
    SUBMISSION_PATH,
    index=False,
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    f"Best Macro-F1       : "
    f"{best_f1:.4f}"
)

print(
    f"Best Epoch          : "
    f"{best_epoch}"
)

print(
    f"Best model          : "
    f"{BEST_MODEL_PATH}"
)

print(
    f"Training history    : "
    f"{HISTORY_PATH}"
)

print(
    f"Classification      : "
    f"{REPORT_PATH}"
)

print(
    f"Confusion matrix    : "
    f"{CONFUSION_PATH}"
)

print(
    f"Submission          : "
    f"{SUBMISSION_PATH}"
)

print("=" * 80)
