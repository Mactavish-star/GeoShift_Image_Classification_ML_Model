# GeoShift — Region-Aware Multispectral Image Classification

A region-shift-robust image classification pipeline for the GeoShift competition using a **ConvNeXt-Tiny backbone with a lightweight Vision Transformer (ViT) head**.

The model classifies 64×64 multispectral image patches into six land-cover classes while explicitly accounting for **geographic distribution shift** between training and validation regions.

---

## Overview

GeoShift is a six-class image classification problem with small 64×64 multispectral image patches.

The main challenge is not only image classification, but **generalisation to previously unseen geographic regions**. A model can obtain an optimistic validation score if visually similar samples from the same region appear in both training and validation.

Our solution therefore treats region as a grouping variable and evaluates the model using **StratifiedGroupKFold**, ensuring that validation regions are not present in the corresponding training split.

The final pipeline combines:

* Region-aware cross-validation
* Four-band multispectral input: B, G, R, NIR
* Derived NDVI and NDWI features
* ConvNeXt-Tiny feature extraction
* Vision Transformer classification head
* GPU-based geometric and radiometric augmentation
* MixUp / CutMix
* Logit-adjusted cross-entropy
* AdamW with layer-wise learning-rate decay
* Exponential moving average (EMA)
* Early stopping using validation Macro-F1
* Eight-way test-time augmentation
* Fold ensembling
* Out-of-fold class-bias tuning for Macro-F1

---

## Classes

The classifier predicts six classes:

| ID | Class         |
| -: | ------------- |
|  0 | Forest        |
|  1 | Shrubland     |
|  2 | Grassland     |
|  3 | Cropland      |
|  4 | Built-up      |
|  5 | Water/Wetland |

---

## Input Data

The model expects four-band 64×64 image patches:

```text
Input shape:
(N, 4, 64, 64)

Bands:
1. Blue
2. Green
3. Red
4. Near Infrared (NIR)
```

The implementation optionally derives two additional spectral indices:

```text
NDVI = (NIR - Red) / (NIR + Red + eps)

NDWI = (Green - NIR) / (Green + NIR + eps)
```

Therefore, with `use_indices=True`, the model receives:

```text
6 channels = B + G + R + NIR + NDVI + NDWI
```

These additional features provide explicit spectral information related to vegetation and water/wetland characteristics.

---

# Methodology

## 1. Region-aware validation

Instead of using a conventional random train/validation split, we use:

```text
StratifiedGroupKFold
```

with geographic region as the grouping variable.

The objective is:

```text
Training regions != Validation regions
```

This provides a more realistic estimate of performance under geographic distribution shift.

Normalization statistics are also computed **only from the training portion of each fold**, preventing information leakage from the validation region.

Out-of-fold (OOF) predictions are collected across the complete training set for downstream analysis and class-bias tuning.

---

## 2. Feature engineering

The original four spectral bands are converted from uint8 to normalized floating-point values.

Two spectral indices are additionally calculated:

### NDVI

Measures vegetation-related spectral contrast:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

### NDWI

Provides additional information useful for distinguishing water and related land-cover conditions:

```text
NDWI = (Green - NIR) / (Green + NIR)
```

The resulting six-channel tensor is supplied to the neural network.

---

# Model Architecture

## ConvNeXt-Tiny + ViT Head

The backbone is a pretrained ConvNeXt-Tiny network.

Instead of using the complete backbone output, the implementation can stop after an earlier ConvNeXt stage and convert the spatial feature map into tokens.

For the default configuration:

```text
Input
  │
  ├── B
  ├── G
  ├── R
  ├── NIR
  ├── NDVI
  └── NDWI
  │
  ▼
ConvNeXt-Tiny
  │
  └── Features through stage 3
  │
  ▼
Spatial feature map
  │
  ▼
Flatten spatial dimensions into tokens
  │
  ▼
LayerNorm + Linear token projection
  │
  ▼
[CLS] token + positional embeddings
  │
  ▼
2-layer Transformer encoder
  │
  ▼
LayerNorm
  │
  ▼
MLP classifier
  │
  ▼
6-class logits
```

The model can also be configured with a GAP classification head for controlled ablation experiments.

---

# Pretrained Weight Adaptation

The original ConvNeXt stem is designed for three-channel RGB input.

The first convolution is therefore modified to support either:

```text
4 channels
```

or:

```text
6 channels
```

when the spectral indices are enabled.

The pretrained RGB filters are transferred into the corresponding multispectral channels, while the NIR and derived-index channels are initialized from the mean pretrained filters with smaller initial weights.

This allows the model to retain useful ImageNet representations while adapting the input layer to multispectral imagery.

---

# Data Augmentation

Augmentation is performed directly on tensors and is designed to improve robustness to geographic and radiometric variation.

## Geometric augmentation

A full 8-way dihedral transformation group is used:

```text
4 rotations × optional horizontal flip
= 8 possible transformations
```

Random zoom/translation crops are also applied.

## Radiometric augmentation

The training pipeline applies:

* Global gain jitter
* Per-channel gain jitter
* Per-channel additive bias
* Gaussian noise
* Value clipping

These transformations simulate changes in illumination and sensor/radiometric conditions between geographic regions.

## Batch-level augmentation

Low-probability:

* MixUp
* CutMix

are used to improve class-boundary regularisation.

---

# Loss Function

The default loss is:

```text
Logit-adjusted Cross-Entropy
```

with label smoothing.

The motivation is to account for class-prior imbalance directly in the decision boundary rather than relying only on explicit loss weighting.

An alternative weighted cross-entropy implementation is also available for ablation experiments.

---

# Optimisation

The model is trained using:

```text
Optimizer:
AdamW

Weight decay:
0.05

Backbone learning rate:
1e-4

Stem learning rate:
2e-4

Head learning rate:
5e-4

Layer-wise LR decay:
0.85

Gradient clipping:
1.0

EMA:
enabled

Learning-rate schedule:
warm-up + cosine decay
```

The backbone and classification head use different learning rates, allowing the pretrained backbone to change more conservatively while the newly initialized head learns faster.

The implementation also includes a persistent plateau-based learning-rate multiplier and early stopping based on validation Macro-F1.

---

# Evaluation Metric

The primary optimisation metric is:

```text
Macro-F1
```

Macro-F1 is computed as the unweighted mean of per-class F1 scores:

```text
Macro-F1 = (F1_1 + F1_2 + ... + F1_6) / 6
```

This treats all six classes equally and prevents dominant classes from completely determining the evaluation metric.

---

# Cross-Validation Pipeline

The complete training process is:

```text
Dataset
   │
   ▼
Extract regions
   │
   ▼
StratifiedGroupKFold
   │
   ├── Fold 0
   ├── Fold 1
   ├── Fold 2
   ├── Fold 3
   └── Fold 4
   │
   ▼
For each fold:
   ├── Compute train-only statistics
   ├── Train model
   ├── Track validation Macro-F1
   ├── Save best EMA checkpoint
   ├── Generate OOF predictions
   └── Generate test predictions
   │
   ▼
Average fold probabilities
   │
   ▼
8-way TTA
   │
   ▼
OOF analysis
   │
   ▼
Optional class log-probability bias tuning
   │
   ▼
Final submission
```

---

# Test-Time Augmentation and Ensemble

At inference time, each image can be evaluated under all eight dihedral transformations.

The resulting class probabilities are averaged:

```text
P_final = mean(P_1, P_2, ..., P_8)
```

Predictions are then averaged across the trained folds.

This produces the final ensemble probability distribution.

---

# OOF Class-Bias Tuning

After generating OOF predictions, the implementation optionally searches for a per-class log-probability offset:

```text
log(P_class) + bias_class
```

The class biases are selected using OOF predictions to maximise Macro-F1.

Two submissions are produced:

```text
submission_v2_nobias.csv
submission_v2_bias.csv
```

The README/reported leaderboard score should clearly distinguish between these two variants.

---

# Experiment Presets

The implementation contains several controlled presets for ablation studies:

| Preset     | Purpose                                |
| ---------- | -------------------------------------- |
| `tuned`    | Default configuration                  |
| `orig`     | Compare against earlier configuration  |
| `hires`    | Higher-resolution ConvNeXt feature map |
| `perimage` | Per-image band normalisation           |
| `gap`      | Replace ViT head with GAP head         |
| `regbal`   | Region-balanced sampling               |
| `tau05`    | Reduced logit-adjustment strength      |

These experiments help isolate the contribution of architectural, normalisation, sampling and loss-function choices.

---

# Results

## Cross-validation

Fill this table using the final competition run:

|     Fold | Best Val Macro-F1 | 8× TTA Macro-F1 |
| -------: | ----------------: | --------------: |
|        0 |         `XX.XXXX` |       `XX.XXXX` |
|        1 |         `XX.XXXX` |       `XX.XXXX` |
|        2 |         `XX.XXXX` |       `XX.XXXX` |
|        3 |         `XX.XXXX` |       `XX.XXXX` |
|        4 |         `XX.XXXX` |       `XX.XXXX` |
| **Mean** |       **XX.XXXX** |     **XX.XXXX** |

## Final OOF Performance

```text
Raw OOF Macro-F1:
XX.XXXX

Bias-tuned OOF Macro-F1:
XX.XXXX
```

## Competition Result

```text
Final leaderboard Macro-F1:
XX.XXXX
```

Do not report the leaderboard number here until it has been verified from the actual submission.

---

# Error Analysis

Recommended analyses:

1. Per-class precision, recall and F1
2. Confusion matrix
3. Region-wise accuracy
4. Hardest regions
5. Class distribution comparison
6. Examples of confidently misclassified samples

The training pipeline automatically generates:

```text
oof_confusion_matrix.csv
oof_region_accuracy.csv
classification report
```

These files should be included in `results/metrics/` for reproducibility.

---

# Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd geoshift-competition

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Layout

The training script expects the following files inside the configured data directory:

```text
data/
├── train_images.npy
├── train_labels.npy
├── train.csv
├── test_images.npy
├── test.csv
└── sample_submission.csv
```

The image tensors are expected to have the following structure:

```text
train_images.npy:
(N, 4, 64, 64)

test_images.npy:
(N_test, 4, 64, 64)
```

Dataset files are intentionally excluded from this repository.

---

# Running the Code

## 1. Smoke test

Run the end-to-end pipeline on synthetic data:

```bash
python src/geoshift_convnext_vit_v2.py smoke
```

This verifies that dataset creation, cross-validation, training, checkpointing, OOF generation and submission generation work correctly.

---

## 2. Full training

```bash
python src/geoshift_convnext_vit_v2.py train
```

Optional:

```bash
python src/geoshift_convnext_vit_v2.py train --folds 0 1 2
```

or:

```bash
python src/geoshift_convnext_vit_v2.py train --epochs 30
```

---

## 3. Re-generate predictions

After checkpoints have been created:

```bash
python src/geoshift_convnext_vit_v2.py predict
```

---

## Reproducibility

The implementation uses a fixed random seed and separately seeds individual folds.

For reproducible experiments, record:

* Git commit
* Python version
* PyTorch version
* torchvision version
* CUDA version
* GPU model
* preset/configuration
* fold IDs
* number of epochs
* final checkpoint
* resulting OOF Macro-F1
* competition submission ID, where applicable

---

# Repository Structure

```text
geoshift-competition/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   └── geoshift_convnext_vit_v2.py
│
├── configs/
│   └── experiment_notes.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_validation_analysis.ipynb
│   └── 03_error_analysis.ipynb
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── submissions/
│
├── docs/
│   ├── presentation.pptx
│   └── methodology.md
│
└── assets/
    ├── architecture.png
    └── pipeline.png
```

---

# Key Takeaways

The central design principle of this solution is to treat the competition as a **geographic generalisation problem**, not simply as random image classification.

The model therefore combines:

```text
Region-aware validation
        +
Multispectral features
        +
ConvNeXt representation learning
        +
Transformer-based spatial reasoning
        +
Shift-oriented augmentation
        +
Macro-F1-aware optimisation
        +
Fold ensemble + TTA
```

This combination is intended to improve robustness when the test distribution contains geographic conditions not directly represented in the training regions.

---

# Team

**Team:** `<TEAM NAME>`

**Members:**

* `<MEMBER 1>`
* `<MEMBER 2>`
* `<MEMBER 3>`
* `<MEMBER 4>`

**Competition:** `<COMPETITION NAME>`

**Repository:** `<GITHUB URL>`

**Final Presentation:** `docs/presentation.pptx`

---

# Acknowledgements

We acknowledge the competition organisers and dataset providers for providing the GeoShift benchmark and evaluation framework.
