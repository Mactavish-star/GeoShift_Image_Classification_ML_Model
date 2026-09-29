"""
GeoShift — ConvNeXt-Tiny + ViT head, v2 (region-aware, shift-robust)
=====================================================================

Single file, three commands:

    python geoshift_convnext_vit_v2.py smoke     # 2-minute dry run on synthetic data
    python geoshift_convnext_vit_v2.py train     # StratifiedGroupKFold training + ensemble + submission
    python geoshift_convnext_vit_v2.py predict   # re-generate the submission from saved fold checkpoints

Optional for `train`:   --folds 0 1 2     (run only these folds)      --epochs 30

What changed vs. geoshift_convnext_vit_fast80_20.py
---------------------------------------------------
FOLDS = SUBSET ENSEMBLE (new)
  * n_folds region-disjoint folds (default 8). Fold k validates on its own held-out regions
    (out-of-fold: every sample is validated exactly once, never by a model that saw its region).
  * Each fold trains on a SMALLER subset drawn from the other regions, identical in size and in
    per-class counts across all folds (subset_mode "balanced" or "proportional"). Different folds see
    different subsets -> diverse ensemble members, faster folds, every class always well represented.
  * CLI: --n_folds 10 --subset_mode proportional | balanced | none

VALIDATION
  * Random 80/20 split -> StratifiedGroupKFold by REGION (R##). Validation regions are never
    seen in training, so the score reflects generalisation to new regions (the GeoShift task).
  * Normalisation statistics are computed on each fold's TRAIN part only.
  * Out-of-fold (OOF) predictions are collected for every training sample.

MODEL
  * ConvNeXt is tapped after stage 3 (4x4x384 = 16 tokens on a 64x64 input). Before, the full
    backbone produced a 2x2 map (only 4 tokens) so the ViT head had almost nothing to attend over,
    and stage 4 (the heaviest) added capacity that can memorise region signatures.
  * Optional stem stride 2 (8x8 = 64 tokens, ~4x compute): cfg["stem_stride"] = 2.
  * 2 extra input channels: NDVI and NDWI (illumination-robust; help water / cropland / forest).
  * Stem init: B,G,R copied from pretrained R,G,B (swapped), NIR = channel mean, indices small.
  * Stochastic depth 0.2, LayerNorm before token projection, stronger head dropout.

LOSS
  * Logit-adjusted cross-entropy (+ label smoothing) instead of class-weighted CE. Removes the
    class-prior bias in the decision boundary without noisy loss re-weighting; good for Macro-F1.
    (Set loss_mode="weighted_ce" to get the old behaviour.)

OPTIMISER / SCHEDULE
  * AdamW, weight decay 0.05 (was 1e-4), no decay on norms/biases/pos-embed/layer-scale.
  * Head LR 1e-3, backbone 1e-4 with layer-wise LR decay 0.85, stem 2e-4 (old: 2e-5 backbone).
  * Warm-up + cosine applied PER STEP.
  * BUG FIX: the old plateau LR reduction was overwritten at the start of the next epoch
    (LR was recomputed from base_lr * cosine), so it never took effect. Now it is a persistent
    multiplier.
  * EMA with warm-up of the decay (early-training EMA no longer dominated by random init).
  * Early stopping on validation Macro-F1.

AUGMENTATION (all on GPU, vectorised, per sample)
  * Full dihedral group (8 flips/rotations), random zoom/translate crop,
    per-channel gain + bias jitter and noise (simulates region-to-region radiometric shift),
    MixUp / CutMix at low probability.
  * Optional per-image band standardisation: cfg["norm_mode"] = "per_image".
  * Optional region-balanced sampling: cfg["region_sampling_power"] > 0.

POST-PROCESSING
  * Fold ensembling (mean of fold probabilities, 8-way TTA each).
  * Per-class log-prob offsets tuned on OOF predictions to maximise Macro-F1.
    Two submissions are written (with / without offsets) so you can compare on the leaderboard.

NOTE: all fold outputs are read back from out_subdir. When you change the config, change out_subdir too
(or delete the folder), otherwise old fold files get mixed into the ensemble.

Speed: the dataset lives on the GPU as uint8 (~0.55 GB) and augmentation is done there, so there
are no CPU dataloader workers. Epoch time depends on your GPU; measure it with `smoke`/1 fold.
"""

import argparse
import copy
import json
import math
import random
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

CFG = dict(
    # ---- paths -------------------------------------------------------------
    data_dir=r"C:\Users\badri\OneDrive\Desktop\GIS-intra-iit\GeoShift\data",
    out_subdir="v2_subset_outputs_version_2",
    region_col=None,               # None = auto (region column, else R## extracted from the Id)

    # ---- data / cv ---------------------------------------------------------
    num_classes=6,
    seed=42,
    n_folds=8,                     # number of folds = number of ensemble members
    run_folds=None,                # None = all folds; or e.g. [0] for a quick single-fold run

    # ---- per-fold TRAINING SUBSET (bagging) ---------------------------------
    # Each fold holds out its own region-disjoint validation set (out-of-fold, covers every sample
    # exactly once) and trains on a SMALLER subset of the remaining regions. All folds get the SAME
    # number of samples and the SAME per-class counts.
    subset_mode="balanced",        # "balanced" = equal samples per class | "proportional" = keep class ratios
                                   # | None = old behaviour (train on the whole remaining pool)
    subset_fill=0.75,              # max fraction of the scarcest class (in any fold) a subset may use;
                                   # lower = more different subsets between folds = more ensemble diversity
    samples_per_class=None,        # balanced mode: fixed count per class (None = auto from subset_fill)
    subset_total=None,             # proportional mode: fixed subset size (None = auto)
    min_per_class=500,             # warn if any class gets fewer training samples than this
    subset_region_power=0.5,       # inside a class, favour rare regions: weight ~ 1/region_count**power

    # ---- training ----------------------------------------------------------
    epochs=40,
    batch_size=128,
    warmup_epochs=2,
    min_lr_factor=0.02,
    lr_backbone=1e-4,              # LR of the deepest ConvNeXt stage kept
    llrd=0.85,                     # layer-wise LR decay towards the input
    lr_stem=2e-4,
    lr_head=1e-3,
    weight_decay=0.05,
    grad_clip=1.0,
    use_amp=True,
    ema_decay=0.999,
    plateau_patience=6,            # epochs without val-F1 gain before LR *= plateau_factor
    plateau_factor=0.5,
    early_stop_patience=14,
    train_eval_samples=4000,       # size of the clean train subset used for the train-F1 monitor
    region_sampling_power=0.0,     # 0 = off; 0.5 = sample regions ~ 1/sqrt(size)

    # ---- loss --------------------------------------------------------------
    loss_mode="logit_adjusted",    # "logit_adjusted" | "weighted_ce"
    logit_adj_tau=1.0,
    class_weight_power=0.5,        # only for weighted_ce
    label_smoothing=0.05,

    # ---- augmentation ------------------------------------------------------
    radiometric_jitter=0.15,       # global gain +-15 %, per-channel gain +-7.5 %
    bias_jitter=0.03,              # additive per-channel offset in [0,1] reflectance units
    noise_std=0.01,
    crop_prob=0.5,
    crop_min_scale=0.75,
    mixup_prob=0.10,
    cutmix_prob=0.10,
    mixup_alpha=0.2,

    # ---- model -------------------------------------------------------------
    pretrained=True,
    drop_path=0.2,
    stem_stride=4,                 # 4 = original (4x4 map after stage 3); 2 = 8x8 map, ~4x compute
    backbone_stages=3,             # 3 = stop after stage 3 (recommended); 4 = full ConvNeXt-Tiny
    head_type="vit",               # "vit" | "gap"
    vit_dim=384,
    vit_heads=6,
    vit_layers=2,
    vit_ff_dim=768,
    vit_dropout=0.10,
    head_dropout=0.25,

    # ---- input features ----------------------------------------------------
    use_indices=True,              # append NDVI, NDWI -> 6 channels
    norm_mode="global",            # "global" | "per_image" (bands only)

    # ---- inference ---------------------------------------------------------
    test_tta=True,
    bias_grid_limit=1.0,
)

CLASS_NAMES = ["Forest", "Shrubland", "Grassland", "Cropland", "Built-up", "Water/Wetland"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_enabled(cfg):
    return bool(cfg["use_amp"]) and DEVICE.type == "cuda"


def autocast(cfg):
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled(cfg))


def n_channels(cfg):
    return 6 if cfg["use_indices"] else 4


def out_dir(cfg):
    p = Path(cfg["data_dir"]) / cfg["out_subdir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# DATA LOADING
# ============================================================

# Matches the region token between underscores, keeping letter prefixes: GEO_NR01_0000_0032 -> NR01, GEO_R05_.. -> R05
REGION_ID_PATTERN = r"_([A-Za-z]*R\d+)_"


def extract_regions(df, forced=None, id_pattern=REGION_ID_PATTERN):
    """Return (regions ndarray, description). Looks for a region column first, then falls back to
    extracting R## from the Id column (e.g. 'R07_000123' -> 'R07')."""
    if forced is not None:
        if forced not in df.columns:
            raise KeyError(f"region_col='{forced}' not in train.csv columns: {list(df.columns)}")
        return df[forced].astype(str).values, f"column '{forced}'"
    for c in df.columns:
        if "region" in c.lower():
            return df[c].astype(str).values, f"column '{c}'"
    for c in df.columns:
        if df[c].astype(str).str.match(r"^R\d+$").all():
            return df[c].astype(str).values, f"column '{c}'"
    if "Id" in df.columns:
        ext = df["Id"].astype(str).str.extract(id_pattern, expand=False)
        if ext.notna().all():
            return ext.values.astype(str), f"regex {id_pattern} applied to 'Id'"
        bad = df["Id"][ext.isna()].astype(str).head(5).tolist()
        raise ValueError(
            f"Found no region column, and the R## pattern did not match {int(ext.isna().sum())} Ids "
            f"(examples: {bad}). Edit REGION_ID_PATTERN or set CFG['region_col'].")
    raise KeyError(f"No region information found. Columns: {list(df.columns)}")


def load_labels_and_regions(cfg):
    d = Path(cfg["data_dir"])
    y = np.load(d / "train_labels.npy").astype(np.int64)
    df = pd.read_csv(d / "train.csv")
    if len(df) != len(y):
        raise ValueError(f"train.csv ({len(df)}) and train_labels.npy ({len(y)}) differ in length.")
    regions, col = extract_regions(df, cfg["region_col"])
    print(f"regions taken from {col}; first Ids -> regions: "
          f"{list(zip(df['Id'].astype(str).head(3), regions[:3]))}")
    uniq, cnt = np.unique(regions, return_counts=True)
    print(f"{len(uniq)} distinct regions: " + ", ".join(f"{u}({c})" for u, c in zip(uniq[:60], cnt[:60])))
    for lab_col in ("label", "Label"):
        if lab_col in df.columns and not np.array_equal(df[lab_col].values.astype(np.int64), y):
            print(f"WARNING: train.csv['{lab_col}'] disagrees with train_labels.npy — check row alignment!")
    return y, regions, col


def load_train_images(cfg):
    x = np.load(Path(cfg["data_dir"]) / "train_images.npy")
    if x.ndim != 4 or x.shape[1:] != (4, 64, 64):
        raise ValueError(f"Expected (N,4,64,64), got {x.shape}")
    return x


def compute_subset_quotas(y, folds, cfg):
    """One per-class quota vector shared by ALL folds -> identical subset size and class counts."""
    K = cfg["num_classes"]
    avail = np.array([np.bincount(y[tr], minlength=K) for tr, _ in folds])     # folds x classes
    min_avail = avail.min(axis=0)
    if (min_avail == 0).any():
        raise ValueError(f"A class is completely absent from the training pool of some fold "
                         f"(min availability per class {min_avail.tolist()}). Use fewer folds.")
    bound = np.floor(min_avail * cfg["subset_fill"]).astype(int)
    if cfg["subset_mode"] == "balanced":
        per = cfg["samples_per_class"] or int(bound.min())
        if per > bound.min():
            print(f"WARNING: samples_per_class={per} exceeds what the scarcest class allows "
                  f"({int(bound.min())} at subset_fill={cfg['subset_fill']}); using {int(bound.min())}.")
            per = int(bound.min())
        q = np.full(K, per, dtype=int)
    elif cfg["subset_mode"] == "proportional":
        prior = np.bincount(y, minlength=K) / len(y)
        total = cfg["subset_total"] or float(np.min(bound / prior))
        q = np.floor(prior * total).astype(int)
        q = np.minimum(np.maximum(q, cfg["min_per_class"]), bound)
    else:
        raise ValueError(f"unknown subset_mode {cfg['subset_mode']}")
    print(f"scarcest availability per class over all folds: {min_avail.tolist()}")
    print(f"per-fold training quota per class            : {q.tolist()}  (total {int(q.sum())})")
    if (q < cfg["min_per_class"]).any():
        print(f"WARNING: some classes have fewer than min_per_class={cfg['min_per_class']} samples per fold. "
              f"Use fewer folds, a larger subset_fill, or lower min_per_class.")
    return q


def draw_class_subset(pool_idx, y, regions, quotas, cfg, seed):
    """Class-stratified draw without replacement; inside a class, rarer regions get higher weight."""
    rng = np.random.RandomState(seed)
    chosen = []
    for c, q in enumerate(quotas):
        cand = pool_idx[y[pool_idx] == c]
        if q >= len(cand):
            chosen.append(cand)
            continue
        _, inv, cnt = np.unique(regions[cand], return_inverse=True, return_counts=True)
        w = (1.0 / cnt[inv]) ** cfg["subset_region_power"]
        chosen.append(rng.choice(cand, size=int(q), replace=False, p=w / w.sum()))
    return np.sort(np.concatenate(chosen))


def make_group_folds(y, regions, cfg):
    K, n = cfg["num_classes"], cfg["n_folds"]
    n_reg = len(np.unique(regions))
    if n > n_reg:
        raise ValueError(f"n_folds={n} > number of regions ({n_reg}); region-disjoint folds are impossible.")
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=cfg["seed"])
    folds = list(sgkf.split(np.zeros(len(y)), y, groups=regions))

    print("=" * 80)
    print(f"StratifiedGroupKFold: {n} folds over {n_reg} regions "
          f"| subset_mode = {cfg['subset_mode']}")
    print("=" * 80)
    if cfg["subset_mode"] is not None:
        quotas = compute_subset_quotas(y, folds, cfg)
        folds = [(draw_class_subset(tr, y, regions, quotas, cfg, cfg["seed"] + 1000 * k + 7), va)
                 for k, (tr, va) in enumerate(folds)]
        used = np.zeros(len(y), dtype=int)
        for tr, _ in folds:
            used[tr] += 1
        print(f"samples used by at least one fold: {(used > 0).mean() * 100:.1f}% of the dataset "
              f"| average reuse {used[used > 0].mean():.2f}x")

    for k, (tr, va) in enumerate(folds):
        assert not (set(regions[tr]) & set(regions[va])), f"fold {k}: region leak between train and val!"
        cnt_tr = np.bincount(y[tr], minlength=K)
        cnt_va = np.bincount(y[va], minlength=K)
        print(f"fold {k}: train {len(tr):6d} ({len(np.unique(regions[tr])):3d} regions) {cnt_tr.tolist()} "
              f"| val {len(va):6d} ({len(np.unique(regions[va])):3d} regions) {cnt_va.tolist()}")
        if (cnt_va == 0).any():
            print(f"  WARNING: fold {k} has a class missing from validation -> its Macro-F1 is unreliable.")
    print()
    return folds


# ============================================================
# FEATURES / NORMALISATION / AUGMENTATION  (all tensor ops, GPU friendly)
# ============================================================

def make_features(x01, use_indices, eps=1e-4):
    """x01: B,4,H,W in [0,1], channel order B,G,R,NIR -> B,4(+2),H,W."""
    if not use_indices:
        return x01
    g, r, n = x01[:, 1], x01[:, 2], x01[:, 3]
    ndvi = (n - r) / (n + r + eps)
    ndwi = (g - n) / (g + n + eps)
    return torch.cat([x01, ndvi.unsqueeze(1), ndwi.unsqueeze(1)], dim=1)


def normalize(feat, mean, std, norm_mode):
    if norm_mode == "per_image":
        bands = feat[:, :4]
        m = bands.mean(dim=(2, 3), keepdim=True)
        s = bands.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        bands = (bands - m) / s
        if feat.size(1) > 4:
            rest = (feat[:, 4:] - mean[:, 4:]) / std[:, 4:]
            return torch.cat([bands, rest], dim=1)
        return bands
    return (feat - mean) / std


def dihedral(x, k):
    """k in 0..7 : rot90 * (k%4), optionally after a horizontal flip (k>=4). Full symmetry group."""
    if k >= 4:
        x = torch.flip(x, [3])
    r = k % 4
    return torch.rot90(x, r, [2, 3]) if r else x


@torch.no_grad()
def compute_channel_stats(X, idx, cfg, chunk=2048):
    c = n_channels(cfg)
    s = torch.zeros(c, dtype=torch.float64, device=DEVICE)
    ss = torch.zeros(c, dtype=torch.float64, device=DEVICE)
    n = 0
    for i in range(0, len(idx), chunk):
        xb = X[idx[i:i + chunk]].to(DEVICE).float() / 255.0
        f = make_features(xb, cfg["use_indices"]).double()
        s += f.sum(dim=(0, 2, 3))
        ss += (f * f).sum(dim=(0, 2, 3))
        n += f.size(0) * f.size(2) * f.size(3)
    mean = s / n
    std = (ss / n - mean * mean).clamp_min(1e-12).sqrt()
    return mean.float().cpu().numpy(), std.float().cpu().numpy()


def stats_to_tensors(mean, std):
    m = torch.as_tensor(np.asarray(mean, dtype=np.float32), device=DEVICE).view(1, -1, 1, 1)
    s = torch.as_tensor(np.asarray(std, dtype=np.float32), device=DEVICE).view(1, -1, 1, 1)
    return m, s


@torch.no_grad()
def augment_bands(x, cfg):
    """x: B,4,H,W in [0,1] (raw bands). Per-sample radiometric + geometric augmentation."""
    B = x.size(0)
    dev = x.device

    j = cfg["radiometric_jitter"]
    if j > 0:
        gain_global = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * j
        gain_channel = 1.0 + (torch.rand(B, 4, 1, 1, device=dev) * 2 - 1) * (j * 0.5)
        bias = (torch.rand(B, 4, 1, 1, device=dev) * 2 - 1) * cfg["bias_jitter"]
        x = x * gain_global * gain_channel + bias
    if cfg["noise_std"] > 0:
        x = x + torch.randn_like(x) * cfg["noise_std"]
    x = x.clamp_(0.0, 1.0)

    if cfg["crop_prob"] > 0:
        apply = torch.rand(B, device=dev) < cfg["crop_prob"]
        s = torch.empty(B, device=dev).uniform_(cfg["crop_min_scale"], 1.0)
        s = torch.where(apply, s, torch.ones_like(s))
        tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
        ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = s
        theta[:, 1, 1] = s
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    op = torch.randint(0, 8, (B,), device=dev)
    out = torch.empty_like(x)
    for k in range(8):
        m = op == k
        if m.any():
            out[m] = dihedral(x[m], k)
    return out


def mix_batch(x, y, cfg):
    """MixUp / CutMix (probabilities from cfg). Returns x, y_a, y_b, lam, mixed."""
    B = x.size(0)
    u = random.random()
    p_cut, p_mix = cfg["cutmix_prob"], cfg["mixup_prob"]
    if B < 2 or u >= p_cut + p_mix:
        return x, y, y, 1.0, False
    perm = torch.randperm(B, device=x.device)
    xs = x[perm]
    if u < p_cut:
        lam = np.random.beta(1.0, 1.0)
        H, W = x.shape[2], x.shape[3]
        cut = math.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = xs[:, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / float(H * W)
    else:
        lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
        x = lam * x + (1.0 - lam) * xs
    return x, y, y[perm], lam, True


def to_model_input(x):
    if DEVICE.type == "cuda":
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()


# ============================================================
# MODEL
# ============================================================

class ConvNeXtViT(nn.Module):
    def __init__(self, cfg, pretrained):
        super().__init__()
        in_ch = n_channels(cfg)
        kwargs = dict(stochastic_depth_prob=cfg["drop_path"])

        backbone, loaded = None, False
        if pretrained:
            try:
                backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, **kwargs)
                loaded = True
            except Exception as e:  # offline machine etc.
                print(f"WARNING: could not load ImageNet weights ({e}). Training from scratch!")
        if backbone is None:
            backbone = convnext_tiny(weights=None, **kwargs)

        old = backbone.features[0][0]
        stride = int(cfg["stem_stride"])
        padding = old.padding if stride == 4 else (1, 1)
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=stride, padding=padding, bias=old.bias is not None)
        if loaded:
            with torch.no_grad():
                w = old.weight                         # [96, 3, 4, 4] in R,G,B order
                new.weight.zero_()
                new.weight[:, 0] = w[:, 2] * 0.75      # B  <- pretrained B
                new.weight[:, 1] = w[:, 1] * 0.75      # G
                new.weight[:, 2] = w[:, 0] * 0.75      # R
                new.weight[:, 3] = w.mean(dim=1) * 0.75  # NIR <- mean filter
                if in_ch > 4:
                    new.weight[:, 4:] = w.mean(dim=1, keepdim=True) * 0.10  # NDVI, NDWI: small start
                if new.bias is not None and old.bias is not None:
                    new.bias.copy_(old.bias)
        backbone.features[0][0] = new

        self.features = backbone.features[: 2 * int(cfg["backbone_stages"])]

        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            fm = self.features(torch.zeros(1, in_ch, 64, 64))
        self.features.train(was_training)
        self.feature_dim, self.feature_h, self.feature_w = fm.shape[1], fm.shape[2], fm.shape[3]
        self.num_spatial = self.feature_h * self.feature_w
        self.head_type = cfg["head_type"]
        d = cfg["vit_dim"]

        if self.head_type == "vit":
            self.pre_norm = nn.LayerNorm(self.feature_dim)
            self.token_projection = nn.Linear(self.feature_dim, d)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial + 1, d))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg["vit_heads"], dim_feedforward=cfg["vit_ff_dim"],
                dropout=cfg["vit_dropout"], activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=cfg["vit_layers"])
            self.norm = nn.LayerNorm(d)
            self.classifier = nn.Sequential(
                nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(cfg["head_dropout"]),
                nn.Linear(d // 2, cfg["num_classes"]),
            )
        elif self.head_type == "gap":
            self.norm = nn.LayerNorm(self.feature_dim)
            self.classifier = nn.Sequential(
                nn.Dropout(cfg["head_dropout"]), nn.Linear(self.feature_dim, cfg["num_classes"])
            )
        else:
            raise ValueError(f"unknown head_type {self.head_type}")

    def forward(self, x):
        x = self.features(x).flatten(2).transpose(1, 2)        # B,N,C
        if self.head_type == "gap":
            return self.classifier(self.norm(x.mean(dim=1)))
        x = self.token_projection(self.pre_norm(x))
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0]))


def build_model(cfg, pretrained):
    m = ConvNeXtViT(cfg, pretrained).to(DEVICE)
    if DEVICE.type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    return m


# ============================================================
# LOSS
# ============================================================

class Objective:
    def __init__(self, train_counts, cfg):
        counts = np.maximum(np.asarray(train_counts, dtype=np.float64), 1.0)
        prior = counts / counts.sum()
        self.mode = cfg["loss_mode"]
        self.tau = cfg["logit_adj_tau"]
        self.ls = cfg["label_smoothing"]
        self.log_prior = torch.tensor(np.log(prior), dtype=torch.float32, device=DEVICE)
        w = (counts.sum() / (len(counts) * counts)) ** cfg["class_weight_power"]
        self.weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)

    def train_loss(self, logits, y):
        logits = logits.float()
        if self.mode == "logit_adjusted":
            return F.cross_entropy(logits + self.tau * self.log_prior, y, label_smoothing=self.ls)
        return F.cross_entropy(logits, y, weight=self.weight, label_smoothing=self.ls)


# ============================================================
# EMA
# ============================================================

class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.e = list(self.ema.state_dict().values())
        self.m = list(model.state_dict().values())

    @torch.no_grad()
    def update(self):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for e, m in zip(self.e, self.m):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                e.copy_(m)


# ============================================================
# OPTIMISER
# ============================================================

def build_optimizer(model, cfg):
    n_feat = len(model.features)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = (p.ndim == 1 or name.endswith(".bias") or "pos_embed" in name
                 or "cls_token" in name or "layer_scale" in name)
        if name.startswith("features.0.0."):
            tag, lr = "stem", cfg["lr_stem"]
        elif name.startswith("features."):
            i = int(name.split(".")[1])
            tag, lr = f"f{i}", cfg["lr_backbone"] * (cfg["llrd"] ** (n_feat - 1 - i))
        else:
            tag, lr = "head", cfg["lr_head"]
        key = (tag, no_wd)
        if key not in groups:
            groups[key] = dict(params=[], lr=lr, base_lr=lr,
                               weight_decay=0.0 if no_wd else cfg["weight_decay"])
        groups[key]["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999))


def lr_factor(step, warmup_steps, total_steps, min_factor):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_probs(model, X, idx, mean_t, std_t, cfg, tta, batch_size=256):
    model.eval()
    out = []
    views = range(8) if tta else [0]
    for s in range(0, len(idx), batch_size):
        xb = X[idx[s:s + batch_size]].to(DEVICE, non_blocking=True).float() / 255.0
        xin = normalize(make_features(xb, cfg["use_indices"]), mean_t, std_t, cfg["norm_mode"])
        acc = 0
        for k in views:
            with autocast(cfg):
                logits = model(to_model_input(dihedral(xin, k)))
            acc = acc + torch.softmax(logits.float(), dim=1)
        out.append((acc / len(views)).cpu())
    return torch.cat(out).numpy()


def macro_f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(np.float64)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1.mean())


def load_checkpoint_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    cfg = ckpt["cfg"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["ema_state_dict"], strict=True)
    model.eval()
    mean_t, std_t = stats_to_tensors(ckpt["channel_mean"], ckpt["channel_std"])
    return model, mean_t, std_t, cfg, ckpt


def predict_test_with_checkpoint(path, test_x_u8, tta):
    model, mean_t, std_t, cfg, _ = load_checkpoint_model(path)
    idx = torch.arange(len(test_x_u8))
    probs = predict_probs(model, test_x_u8, idx, mean_t, std_t, cfg, tta=tta)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return probs


# ============================================================
# ONE FOLD
# ============================================================

def train_fold(fold, tr_idx, va_idx, X, y_np, cfg, odir):
    seed_everything(cfg["seed"] + fold)
    K = cfg["num_classes"]
    y_t = torch.from_numpy(y_np).to(DEVICE)
    tr_t = torch.as_tensor(tr_idx, device=X.device)
    va_t = torch.as_tensor(va_idx, device=X.device)

    mean, std = compute_channel_stats(X, tr_t, cfg)
    mean_t, std_t = stats_to_tensors(mean, std)
    print(f"[fold {fold}] channel mean {np.round(mean, 4)} std {np.round(std, 4)}")

    model = build_model(cfg, cfg["pretrained"])
    print(f"[fold {fold}] params {sum(p.numel() for p in model.parameters()):,} | "
          f"feature map {model.feature_h}x{model.feature_w}x{model.feature_dim} "
          f"({model.num_spatial} tokens) | head={cfg['head_type']}")

    ema = ModelEMA(model, cfg["ema_decay"])
    optimizer = build_optimizer(model, cfg)
    objective = Objective(np.bincount(y_np[tr_idx], minlength=K), cfg)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=amp_enabled(cfg))

    bs = cfg["batch_size"]
    steps_per_epoch = max(1, len(tr_idx) // bs)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup_epochs"]

    # Optional region-balanced sampling weights
    sampler_w = None
    if cfg["region_sampling_power"] > 0:
        regs = REGIONS[tr_idx]
        uniq, inv, cnt = np.unique(regs, return_inverse=True, return_counts=True)
        w = (1.0 / cnt[inv]) ** cfg["region_sampling_power"]
        sampler_w = torch.tensor(w / w.sum(), dtype=torch.float32, device=DEVICE)

    rng = np.random.RandomState(cfg["seed"] + fold)
    sub = rng.choice(len(tr_idx), size=min(cfg["train_eval_samples"], len(tr_idx)), replace=False)
    train_sub_t = torch.as_tensor(np.asarray(tr_idx)[sub], device=X.device)

    history, best = [], dict(f1=-1.0, epoch=-1, state=None)
    step, lr_scale, stale = 0, 1.0, 0
    ckpt_path = odir / f"fold{fold}.pth"
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        t_ep = time.time()
        model.train()
        if sampler_w is not None:
            order = torch.multinomial(sampler_w, len(tr_idx), replacement=True)
        else:
            order = torch.randperm(len(tr_idx), device=DEVICE)
        tr_dev = torch.as_tensor(tr_idx, device=DEVICE)
        run_loss = torch.zeros((), device=DEVICE)

        for s in range(steps_per_epoch):
            bi = tr_dev[order[s * bs:(s + 1) * bs]]
            xb = X[bi.to(X.device)].to(DEVICE, non_blocking=True).float() / 255.0
            yb = y_t[bi]

            f = lr_factor(step, warmup_steps, total_steps, cfg["min_lr_factor"]) * lr_scale
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * f

            xb = augment_bands(xb, cfg)
            xin = normalize(make_features(xb, cfg["use_indices"]), mean_t, std_t, cfg["norm_mode"])
            xin, ya, yb2, lam, mixed = mix_batch(xin, yb, cfg)

            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):
                logits = model(to_model_input(xin))
            if mixed:
                loss = lam * objective.train_loss(logits, ya) + (1 - lam) * objective.train_loss(logits, yb2)
            else:
                loss = objective.train_loss(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            run_loss += loss.detach()
            step += 1

        run_loss = run_loss.item()

        # ---- evaluation (EMA weights on both, so the train/val gap is comparable) ----
        p_tr = predict_probs(ema.ema, X, train_sub_t, mean_t, std_t, cfg, tta=False)
        y_tr = y_np[train_sub_t.cpu().numpy()]
        tr_f1 = macro_f1_fast(y_tr, p_tr.argmax(1), K)

        p_va = predict_probs(ema.ema, X, va_t, mean_t, std_t, cfg, tta=False)
        y_va = y_np[va_idx]
        va_f1 = macro_f1_fast(y_va, p_va.argmax(1), K)
        va_acc = float((p_va.argmax(1) == y_va).mean())
        va_loss = float(-np.log(np.clip(p_va[np.arange(len(y_va)), y_va], 1e-8, 1)).mean())

        history.append(dict(epoch=epoch, train_loss=run_loss / steps_per_epoch, train_f1_clean=tr_f1,
                            val_loss=va_loss, val_f1=va_f1, val_acc=va_acc, lr_scale=lr_scale,
                            seconds=time.time() - t_ep))
        pd.DataFrame(history).to_csv(odir / f"history_fold{fold}.csv", index=False)

        improved = va_f1 > best["f1"] + 1e-9
        print(f"[fold {fold}] ep {epoch:03d}/{cfg['epochs']} | loss {run_loss / steps_per_epoch:.4f} | "
              f"train F1 {tr_f1:.4f} | val F1 {va_f1:.4f} | val acc {va_acc:.4f} | "
              f"{time.time() - t_ep:.1f}s{'  *' if improved else ''}")

        if improved:
            best.update(f1=va_f1, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale % cfg["plateau_patience"] == 0:
                lr_scale *= cfg["plateau_factor"]
                print(f"[fold {fold}]   plateau -> LR scale {lr_scale:.3f}")
            if stale >= cfg["early_stop_patience"]:
                print(f"[fold {fold}] early stop (no gain for {stale} epochs)")
                break

    print(f"[fold {fold}] best val Macro-F1 (no TTA, EMA) {best['f1']:.5f} at epoch {best['epoch']} "
          f"| {(time.time() - t0) / 60:.1f} min")

    torch.save(dict(
        cfg=cfg, fold=fold, epoch=best["epoch"], val_f1=float(best["f1"]),
        ema_state_dict=best["state"], channel_mean=[float(v) for v in mean],
        channel_std=[float(v) for v in std],
    ), ckpt_path)

    # ---- reload from disk (verifies the checkpoint round-trip) and produce OOF + test probs ----
    model2, m_t, s_t, cfg2, _ = load_checkpoint_model(ckpt_path)
    oof = predict_probs(model2, X, va_t, m_t, s_t, cfg2, tta=cfg["test_tta"])
    np.savez(odir / f"oof_fold{fold}.npz", idx=np.asarray(va_idx), probs=oof)
    f1_tta = macro_f1_fast(y_np[va_idx], oof.argmax(1), K)
    print(f"[fold {fold}] val Macro-F1 with {'8x TTA' if cfg['test_tta'] else 'no TTA'}: {f1_tta:.5f}")
    del model2, model, ema
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best["f1"], f1_tta


# ============================================================
# BIAS TUNING + SUBMISSION
# ============================================================

def tune_class_bias(logp, y, K, limit=1.0, steps=41, rounds=3):
    grid = np.linspace(-limit, limit, steps)
    b = np.zeros(K)
    best = macro_f1_fast(y, (logp + b).argmax(1), K)
    for _ in range(rounds):
        improved = False
        for c in range(K):
            for v in grid:
                trial = b.copy()
                trial[c] = v
                f = macro_f1_fast(y, (logp + trial).argmax(1), K)
                if f > best + 1e-5:
                    best, b, improved = f, trial, True
        if not improved:
            break
    return b, best


def write_submission(cfg):
    K = cfg["num_classes"]
    odir = out_dir(cfg)
    d = Path(cfg["data_dir"])
    y, regions, _ = load_labels_and_regions(cfg)

    oof = np.zeros((len(y), K), dtype=np.float64)
    mask = np.zeros(len(y), dtype=bool)
    fold_f1 = []
    for f in sorted(odir.glob("oof_fold*.npz")):
        z = np.load(f)
        oof[z["idx"]] = z["probs"]
        mask[z["idx"]] = True
        fold_f1.append(macro_f1_fast(y[z["idx"]], z["probs"].argmax(1), K))
    test_files = sorted(odir.glob("test_probs_fold*.npy"))
    if not test_files or not mask.any():
        raise RuntimeError(f"No fold outputs found in {odir}. Run `train` first.")

    print("=" * 80)
    print("ENSEMBLE / OOF SUMMARY")
    print("=" * 80)
    print(f"folds with OOF: {len(fold_f1)} | per-fold Macro-F1 (TTA): {np.round(fold_f1, 4).tolist()} "
          f"| mean {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")

    yo, po = y[mask], oof[mask]
    raw_f1 = macro_f1_fast(yo, po.argmax(1), K)
    print(f"OOF Macro-F1 (unseen-region estimate): {raw_f1:.5f}")
    print(classification_report(yo, po.argmax(1), target_names=CLASS_NAMES, digits=4, zero_division=0))
    cm = confusion_matrix(yo, po.argmax(1), labels=np.arange(K))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(odir / "oof_confusion_matrix.csv")

    reg_df = pd.DataFrame(dict(region=regions[mask], correct=(po.argmax(1) == yo)))
    reg_acc = reg_df.groupby("region")["correct"].agg(["mean", "size"]).sort_values("mean")
    reg_acc.to_csv(odir / "oof_region_accuracy.csv")
    print("hardest regions (OOF accuracy):")
    print(reg_acc.head(5).to_string())

    logp = np.log(np.clip(po, 1e-8, 1.0))
    bias, tuned_f1 = tune_class_bias(logp, yo, K, limit=cfg["bias_grid_limit"])
    print(f"\nclass log-prob offsets {np.round(bias, 3).tolist()} -> OOF Macro-F1 {raw_f1:.5f} -> {tuned_f1:.5f} "
          "(in-sample on OOF: expect the real gain to be smaller)")

    test_probs = np.mean([np.load(f) for f in test_files], axis=0)
    test_df = pd.read_csv(d / "test.csv")
    sample = pd.read_csv(d / "sample_submission.csv")
    if len(test_probs) != len(test_df):
        raise RuntimeError("test probabilities and test.csv differ in length")

    def save(pred, name):
        if len(sample) == len(test_df) and {"Id", "label"} <= set(sample.columns):
            sub = sample.copy()
            if not sub["Id"].equals(test_df["Id"]):
                sub["Id"] = test_df["Id"].values
        else:
            sub = pd.DataFrame({"Id": test_df["Id"].values})
        sub["label"] = pred.astype(np.int64)
        sub = sub[["Id", "label"]]
        sub.to_csv(odir / name, index=False)
        dist = (np.bincount(pred, minlength=K) / len(pred) * 100).round(2).tolist()
        print(f"saved {odir / name} | predicted class % {dist}")

    lt = np.log(np.clip(test_probs, 1e-8, 1.0))
    save(lt.argmax(1), "submission_v2_nobias.csv")
    save((lt + bias).argmax(1), "submission_v2_bias.csv")
    np.save(odir / "test_probs_ensemble.npy", test_probs)
    with open(odir / "ensemble_meta.json", "w") as fh:
        json.dump(dict(bias=bias.tolist(), oof_f1_raw=raw_f1, oof_f1_bias=tuned_f1,
                       n_test_models=len(test_files), fold_f1=fold_f1), fh, indent=2)
    print("Done.")


# ============================================================
# COMMANDS
# ============================================================

REGIONS = None   # set in run_training (used by optional region-balanced sampling)


def run_training(cfg):
    global REGIONS
    seed_everything(cfg["seed"])
    odir = out_dir(cfg)
    d = Path(cfg["data_dir"])

    print("=" * 80)
    print("GeoShift v2 — ConvNeXt-Tiny + ViT, region-aware CV")
    print("=" * 80)
    print(f"device {DEVICE}" + (f" | {torch.cuda.get_device_name(0)}" if DEVICE.type == 'cuda' else ""))
    keys = ["epochs", "batch_size", "lr_backbone", "lr_head", "weight_decay", "loss_mode", "stem_stride",
            "backbone_stages", "head_type", "use_indices", "norm_mode", "drop_path", "region_sampling_power"]
    print({k: cfg[k] for k in keys})

    y, regions, col = load_labels_and_regions(cfg)
    REGIONS = regions
    print(f"region column: '{col}' | {len(np.unique(regions))} regions | class counts {np.bincount(y).tolist()}")

    x_np = load_train_images(cfg)
    X = torch.from_numpy(x_np)
    if DEVICE.type == "cuda":
        X = X.to(DEVICE)                                    # uint8 ~0.55 GB, stays resident
    test_x = torch.from_numpy(np.load(d / "test_images.npy"))
    if test_x.ndim != 4 or tuple(test_x.shape[1:]) != (4, 64, 64):
        raise ValueError(f"Bad test_images shape {tuple(test_x.shape)}")

    folds = make_group_folds(y, regions, cfg)
    run = cfg["run_folds"] if cfg["run_folds"] is not None else list(range(len(folds)))

    summary = []
    for k in run:
        tr_idx, va_idx = folds[k]
        f1_best, f1_tta = train_fold(k, tr_idx, va_idx, X, y, cfg, odir)
        summary.append((k, f1_best, f1_tta))
        np.save(odir / f"test_probs_fold{k}.npy",
                predict_test_with_checkpoint(odir / f"fold{k}.pth", test_x, cfg["test_tta"]))

    print("\nfold summary (fold, best val F1 no-TTA, val F1 TTA):")
    for k, a, b in summary:
        print(f"  fold {k}: {a:.5f}  {b:.5f}")
    write_submission(cfg)


def run_predict(cfg):
    odir = out_dir(cfg)
    ckpts = sorted(odir.glob("fold*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No fold*.pth in {odir}")
    test_x = torch.from_numpy(np.load(Path(cfg["data_dir"]) / "test_images.npy"))
    for p in ckpts:
        k = int(re.findall(r"fold(\d+)", p.name)[0])
        print(f"predicting with {p.name}")
        np.save(odir / f"test_probs_fold{k}.npy", predict_test_with_checkpoint(p, test_x, cfg["test_tta"]))
    write_submission(cfg)


def make_synthetic_dataset(root, n=384, n_test=96, n_regions=12, seed=0):
    """Tiny fake dataset in the real file formats, used by `smoke`."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 6, size=n)
    x = rng.randint(0, 256, size=(n, 4, 64, 64)).astype(np.uint8)
    for c in range(6):                       # make classes weakly separable so F1 is not pure noise
        x[y == c, c % 4] = np.clip(x[y == c, c % 4].astype(int) + 40 * (c + 1), 0, 255).astype(np.uint8)
    regions = np.array([f"R{r:02d}" for r in rng.randint(1, n_regions + 1, size=n)])
    np.save(root / "train_images.npy", x)
    np.save(root / "train_labels.npy", y)
    pd.DataFrame(dict(Id=np.arange(n), region=regions, label=y)).to_csv(root / "train.csv", index=False)
    np.save(root / "test_images.npy", rng.randint(0, 256, size=(n_test, 4, 64, 64)).astype(np.uint8))
    pd.DataFrame(dict(Id=np.arange(n_test))).to_csv(root / "test.csv", index=False)
    pd.DataFrame(dict(Id=np.arange(n_test), label=0)).to_csv(root / "sample_submission.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict", "smoke"])
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_folds", type=int, default=None)
    ap.add_argument("--subset_mode", choices=["balanced", "proportional", "none"], default=None)
    args = ap.parse_args()

    cfg = copy.deepcopy(CFG)
    if args.folds is not None and len(args.folds) > 0:
        cfg["run_folds"] = args.folds
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.n_folds is not None:
        cfg["n_folds"] = args.n_folds
    if args.subset_mode is not None:
        cfg["subset_mode"] = None if args.subset_mode == "none" else args.subset_mode

    if args.command == "smoke":
        tmp = Path(tempfile.mkdtemp(prefix="geoshift_smoke_"))
        make_synthetic_dataset(tmp)
        cfg.update(data_dir=str(tmp), n_folds=3, run_folds=[0, 1], epochs=2, batch_size=32,
                   pretrained=False, warmup_epochs=1, train_eval_samples=64,
                   min_per_class=5)
        print(f"SMOKE TEST on synthetic data in {tmp}")
        run_training(cfg)
        print("\nSMOKE TEST PASSED — pipeline runs end-to-end.")
    elif args.command == "train":
        run_training(cfg)
    else:
        run_predict(cfg)


if __name__ == "__main__":
    main()
