# ============================================================
# GeoShift - Prediction / Submission Script
#
# Loads the trained Custom4ChannelCNN checkpoint and generates
# a Kaggle submission from test.csv + test.npy/test_images.npy.
#
# Expected submission:
#     Id,label
#
# The checkpoint produced by:
#     geoshift_custom_4channel_cnn_training.py
# contains:
#     - model_state_dict
#     - channel_mean
#     - channel_std
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG / ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="GeoShift Custom CNN test prediction"
)

parser.add_argument(
    "--data-dir",
    type=str,
    default=r"C:\Users\badri\OneDrive\Desktop\GIS-intra-iit\GeoShift\data",
    help="Directory containing test.csv, test.npy/test_images.npy "
         "and sample_submission.csv",
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to the trained .pth checkpoint",
)

parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output submission CSV path",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Test inference batch size",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    help="DataLoader workers",
)

args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
MODEL_PATH = Path(args.model)

TEST_CSV_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Support both names.
TEST_NPY_CANDIDATES = [
    DATA_DIR / "test.npy",
    DATA_DIR / "test_images.npy",
]

if args.output is None:
    OUTPUT_PATH = DATA_DIR / "submission_custom_cnn.csv"
else:
    OUTPUT_PATH = Path(args.output)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 6

CLASS_NAMES = [
    "Forest",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Water/Wetland",
]


# ============================================================
# MODEL
# ============================================================

class FourierSpectralAttention(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels

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

        self.complex_weight = nn.Parameter(
            torch.randn(
                channels,
                1,
                1,
                2,
                dtype=torch.float32,
            ) * 0.02
        )

    def forward(self, x):

        B, C, H, W = x.shape

        channel_weights = self.se(x).view(
            B,
            C,
            1,
            1,
        )

        x_att = x * channel_weights

        x_fft = torch.fft.rfft2(
            x_att,
            norm="ortho",
        )

        x_fft_real_imag = torch.view_as_real(
            x_fft
        )

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

        return x_spatial + x


class DynamicClassRouter(nn.Module):

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

    def forward(self, x):

        B, C, H, W = x.shape

        routing_weights = self.router(x)

        dynamic_w = torch.einsum(
            "bk,koihw->boihw",
            routing_weights,
            self.class_weights,
        )

        # Batched dynamic convolution.
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

        return out.reshape(
            B,
            self.out_channels,
            H,
            W,
        )


class Custom4ChannelCNN(nn.Module):

    def __init__(
        self,
        in_channels=4,
        num_classes=6,
    ):
        super().__init__()

        self.stage1_conv = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            padding=1,
        )
        self.stage1_bn = nn.BatchNorm2d(32)
        self.stage1_act = nn.ReLU()
        self.stage1_pool = nn.MaxPool2d(2, 2)

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

        self.stage3_router = DynamicClassRouter(
            64,
            128,
            num_classes=num_classes,
        )
        self.stage3_bn = nn.BatchNorm2d(128)
        self.stage3_act = nn.SiLU()
        self.stage3_pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            128,
            num_classes,
        )

    def forward(self, x):

        x = self.stage1_conv(x)
        x = self.stage1_bn(x)
        x = self.stage1_act(x)
        x = self.stage1_pool(x)

        x = self.stage2_fourier(x)
        x = self.stage2_conv(x)
        x = self.stage2_bn(x)
        x = self.stage2_act(x)
        x = self.stage2_pool(x)

        x = self.stage3_router(x)
        x = self.stage3_bn(x)
        x = self.stage3_act(x)
        x = self.stage3_pool(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)


# ============================================================
# TEST DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(
        self,
        images,
        mean,
        std,
    ):
        self.images = images

        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        ).reshape(4, 1, 1)

        self.std = np.asarray(
            std,
            dtype=np.float32,
        ).reshape(4, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        x = (
            self.images[idx]
            .astype(np.float32)
            / 255.0
        )

        x = (x - self.mean) / self.std

        return torch.from_numpy(
            x.copy()
        ).float()


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("GeoShift Custom CNN Prediction")
print("=" * 80)

print(f"Device : {DEVICE}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )

if not TEST_CSV_PATH.exists():
    raise FileNotFoundError(
        f"test.csv not found:\n{TEST_CSV_PATH}"
    )

# ------------------------------------------------------------
# Find test NPY
# ------------------------------------------------------------

TEST_NPY_PATH = None

for candidate in TEST_NPY_CANDIDATES:
    if candidate.exists():
        TEST_NPY_PATH = candidate
        break

if TEST_NPY_PATH is None:
    raise FileNotFoundError(
        "Could not find test.npy or test_images.npy in:\n"
        f"{DATA_DIR}"
    )

print(f"Test CSV : {TEST_CSV_PATH}")
print(f"Test NPY : {TEST_NPY_PATH}")
print(f"Model    : {MODEL_PATH}")

# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

test_df = pd.read_csv(
    TEST_CSV_PATH
)

test_images = np.load(
    TEST_NPY_PATH
)

print(f"\nTest images shape: {test_images.shape}")
print(f"Test CSV rows     : {len(test_df)}")

if test_images.ndim != 4:
    raise ValueError(
        "Expected test images with shape "
        "(N, 4, H, W)."
    )

if test_images.shape[1] != 4:
    raise ValueError(
        "Expected exactly 4 channels: "
        "B, G, R, NIR."
    )

if len(test_images) != len(test_df):
    raise ValueError(
        f"Mismatch: test.npy has {len(test_images)} "
        f"images but test.csv has {len(test_df)} rows."
    )

if "Id" not in test_df.columns:
    raise ValueError(
        "test.csv must contain an 'Id' column."
    )


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

print("\nLoading model checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )

if "channel_mean" not in checkpoint:
    raise KeyError(
        "Checkpoint does not contain "
        "'channel_mean'."
    )

if "channel_std" not in checkpoint:
    raise KeyError(
        "Checkpoint does not contain "
        "'channel_std'."
    )

channel_mean = np.asarray(
    checkpoint["channel_mean"],
    dtype=np.float32,
)

channel_std = np.asarray(
    checkpoint["channel_std"],
    dtype=np.float32,
)

if channel_mean.shape != (4,):
    raise ValueError(
        f"Expected channel_mean shape (4,), "
        f"got {channel_mean.shape}"
    )

if channel_std.shape != (4,):
    raise ValueError(
        f"Expected channel_std shape (4,), "
        f"got {channel_std.shape}"
    )

print("\nUsing training normalization:")
for i, name in enumerate(
    ["B", "G", "R", "NIR"]
):
    print(
        f"  {name:>3s}: "
        f"mean={channel_mean[i]:.6f}, "
        f"std={channel_std[i]:.6f}"
    )


# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------

model = Custom4ChannelCNN(
    in_channels=4,
    num_classes=NUM_CLASSES,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print(
    f"\nCheckpoint best Macro-F1: "
    f"{checkpoint.get('best_f1', 'N/A')}"
)


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

dataset = TestDataset(
    test_images,
    channel_mean,
    channel_std,
)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(
        args.num_workers > 0
    ),
)


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning inference...")

predictions = []

use_amp = DEVICE.type == "cuda"

with torch.no_grad():

    for batch_idx, x in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(x)

        pred = logits.argmax(
            dim=1
        )

        predictions.extend(
            pred.cpu()
            .numpy()
            .tolist()
        )

        if (
            batch_idx + 1
        ) % 20 == 0:

            print(
                f"  Processed "
                f"{min((batch_idx + 1) * args.batch_size, len(dataset))}"
                f"/{len(dataset)}"
            )

predictions = np.asarray(
    predictions,
    dtype=np.int64,
)

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Prediction count mismatch: "
        f"{len(predictions)} predictions for "
        f"{len(test_df)} test rows."
    )

if predictions.min() < 0 or predictions.max() >= NUM_CLASSES:
    raise RuntimeError(
        "Invalid predicted class detected."
    )


# ============================================================
# BUILD SUBMISSION
# ============================================================

submission = pd.DataFrame(
    {
        "Id": test_df["Id"],
        "label": predictions,
    }
)


# ------------------------------------------------------------
# Validate against sample_submission when available
# ------------------------------------------------------------

if SAMPLE_SUBMISSION_PATH.exists():

    sample = pd.read_csv(
        SAMPLE_SUBMISSION_PATH
    )

    print(
        f"\nSample submission columns: "
        f"{list(sample.columns)}"
    )

    # If sample has Id, make sure our IDs follow
    # the same row ordering.
    if "Id" in sample.columns:

        if len(sample) != len(submission):
            raise ValueError(
                "sample_submission.csv and test.csv "
                "have different numbers of rows."
            )

        if not (
            sample["Id"].astype(str).to_numpy()
            == submission["Id"].astype(str).to_numpy()
        ).all():

            raise ValueError(
                "test.csv Id ordering does not match "
                "sample_submission.csv."
            )

    print(
        "✓ Submission row count/order validated "
        "against sample_submission.csv"
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

submission.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 80)
print("PREDICTION COMPLETE")
print("=" * 80)

print(f"Rows       : {len(submission)}")
print(f"Output     : {OUTPUT_PATH}")

print("\nPrediction distribution:")
print(
    submission["label"]
    .value_counts()
    .sort_index()
)

print("\nFirst 10 rows:")
print(submission.head(10).to_string(index=False))

print("\nSubmission columns:")
print(list(submission.columns))

print("=" * 80)
