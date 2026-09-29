# GeoShift — Region-Aware Multispectral Image Classification

> **A region-shift-robust classification pipeline for 64×64 multispectral imagery using ConvNeXt-Tiny + Vision Transformer**

## Overview

GeoShift is a six-class land-cover image classification problem based on small multispectral image patches.

The six classes are:

| Class ID | Class         |
| -------: | ------------- |
|        0 | Forest        |
|        1 | Shrubland     |
|        2 | Grassland     |
|        3 | Cropland      |
|        4 | Built-up      |
|        5 | Water/Wetland |

Each sample is a **64×64 four-band multispectral image** containing:

* Blue
* Green
* Red
* Near Infrared (NIR)

The primary evaluation metric is **Macro-F1**.

The central challenge is not merely classifying the training distribution. The competition is designed around **geographic distribution shift**, so the model must generalise to regions that may not have been represented during training.

This led us to design the solution around the following principle:

> **Validation, feature representation, augmentation and inference should all reflect the geographic shift present in the competition.**

---

## Our Approach

The final system combines five major ideas:

```text
                    GeoShift Pipeline
                           │
            ┌──────────────┴──────────────┐
            │                             │
     Region-aware CV              Multispectral features
            │                             │
            ▼                             ▼
 StratifiedGroupKFold               B,G,R,NIR
  by geographic region                 +
                                      NDVI
                                      NDWI
            │                             │
            └──────────────┬──────────────┘
                           ▼
                 ConvNeXt-Tiny Backbone
                           │
                           ▼
                    Spatial tokens
                           │
                           ▼
                Vision Transformer Head
                           │
                           ▼
               6-class classification
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      EMA weights     8-way TTA       Fold ensemble
          │                │                │
          └────────────────┴────────────────┘
                           ▼
                    OOF predictions
                           │
                           ▼
                  Class-bias tuning
                           │
                           ▼
                     Submission
```

The final implementation is contained in:

```text
src/geoshift_convnext_vit_v2.py
```

---

## What Did Not Work?

One of the main lessons from this competition was that increasing model complexity alone did not solve the problem.

Our model development therefore focused on identifying **why earlier approaches were failing**.

---

### Random 80/20 Validation

#### Initial approach

A conventional random train/validation split was initially considered.

```text
Random 80%
   │
   ├── Train
   │
   └── Validation
```

#### Problem

Samples from the same geographic region can appear in both sets.

That means the validation set can contain regional characteristics already represented during training.

For a geographic-shift problem, this can produce an overly optimistic estimate of generalisation.

#### Change

The final pipeline uses:

```text
StratifiedGroupKFold
```

with geographic region as the grouping variable.

Therefore:

```text
Validation regions
        ∩
Training regions
        =
        ∅
```

The uploaded implementation explicitly extracts region identifiers and uses them as groups in `StratifiedGroupKFold`.

#### Lesson

In GeoShift, validation strategy is part of the model design.

---
# Various Questions and Explanations Answered:
I have worked on this problem statement for past week, I have faced a lot of failures especially when I am trusting my intuition. I am keeping log here, because I feel that failures which made me to give a stronger model is also as important in this competition. These are the questions which I made-up to explain the complexity of my problem

### 4. Why a Standard CNN Was Not Enough

The problem contains two different kinds of information:

#### Local information

Examples:

* textures
* edges
* spectral patterns
* small spatial structures

#### Spatial context

Examples:

* arrangement of vegetation
* boundaries between land-cover types
* relationships between different spatial regions

A conventional CNN/GAP pipeline compresses spatial information aggressively before classification.

We therefore investigated a design where convolutional features are converted into spatial tokens and processed by a Transformer.

---

### Why the Original ConvNeXt + ViT Arrangement Was Weak

A particularly important issue was discovered in the feature extraction stage.

With a 64×64 input, using the full ConvNeXt backbone resulted in a very small spatial representation at the point where the Transformer receives its tokens.

The earlier pipeline effectively produced:

```text
2 × 2 spatial feature map
        ↓
4 spatial tokens
```

The problem is that the Transformer then has almost no spatial structure to model.

In addition, the deeper backbone provides additional capacity that can make it easier for the model to learn region-specific signatures rather than useful transferable features.

The final implementation therefore stops at **ConvNeXt stage 3**.

For the default configuration this produces:

```text
4 × 4 feature map
        ↓
16 spatial tokens
```

The uploaded source explicitly describes this motivation and contrasts it with the previous 2×2 representation.

#### Alternative tested

A higher-resolution configuration is also available:

```text
stem_stride = 2
```

which increases the spatial feature map to:

```text
8 × 8 = 64 tokens
```

#### This is provided as the `hires` experiment preset.

### Why RGB-Only Representation Was Insufficient

The dataset contains four spectral bands rather than ordinary RGB.

Using only RGB discards information contained in NIR.

The final model therefore operates directly on:

```text
Blue
Green
Red
NIR
```

and optionally derives two additional physically meaningful spectral indices.

---

### Feature Engineering

#### NDVI

```text
NDVI = (NIR - Red) / (NIR + Red + ε)
```

NDVI provides an explicit vegetation-related spectral relationship.

#### NDWI

```text
NDWI = (Green - NIR) / (Green + NIR + ε)
```

NDWI provides additional information useful for distinguishing water and related land-cover conditions.

The final model therefore uses:

```text
B + G + R + NIR + NDVI + NDWI
```

when:

```python
use_indices = True
```

The actual implementation constructs these channels directly in the training pipeline.

---

### Why Standard Augmentation Was Not Enough

Ordinary image augmentation mainly addresses spatial variation.

However, geographic shift can also produce differences in:

* illumination
* sensor response
* scene brightness
* per-band intensity
* regional imaging conditions

A model that relies too strongly on these radiometric properties can learn region-specific shortcuts.

Therefore, augmentation was designed specifically to simulate this type of variation.

---

### Shift-Oriented Data Augmentation

The training pipeline performs augmentation directly on tensors.

### Geometric augmentation

The complete 8-way dihedral group is used:

```text
4 rotations × optional horizontal flip(probable consequence) = 8 transformations
```

Random zoom / translation crops are also applied.

#### Radiometric augmentation

The model receives:

* global gain jitter
* per-channel gain jitter
* additive per-channel bias
* Gaussian noise
* clipping

#### Regularisation

Low-probability:

* MixUp
* CutMix

are applied after the image-level transformation stage.

These operations are intended to reduce dependence on superficial regional characteristics.

#### The actual implementation performs these transformations on the GPU.

### Loss Function: Why Weighted Cross-Entropy Was Reconsidered

The dataset contains multiple classes with unequal representation.

A direct approach is weighted cross-entropy.

However, because the competition metric is Macro-F1, our objective is to improve balanced per-class decision boundaries.

The final configuration therefore uses:

```text
Logit-adjusted Cross-Entropy
+
Label smoothing
```

instead of the earlier weighted-CE approach.

The implementation keeps weighted cross-entropy as an experimental alternative via:

```python
loss_mode = "weighted_ce"
```

while the default is:

```python
loss_mode = "logit_adjusted"
```

#### The two loss modes are implemented in the same training code for controlled comparison.

### Optimisation Problems Discovered During Development

Another issue was not model architecture but optimisation behaviour.

The earlier learning-rate plateau mechanism was effectively overwritten when the next epoch recomputed the learning rate from the base cosine schedule.

Therefore, the intended plateau reduction did not persist.

The final implementation introduced a persistent learning-rate multiplier:

```text
LR = base_LR × cosine_factor × plateau_scale
```

where:

```text
plateau_scale
```

is reduced when validation Macro-F1 stops improving.

This bug is explicitly documented in the source.

---

## Final Model Architecture

The final default model is:

```text
Input
64 × 64 × 6
(B, G, R, NIR, NDVI, NDWI)
        │
        ▼
Modified ConvNeXt-Tiny stem
        │
        ▼
ConvNeXt stages 1–3
        │
        ▼
4 × 4 × 384 feature map
        │
        ▼
16 spatial tokens
        │
        ▼
LayerNorm
        │
        ▼
Linear token projection
        │
        ▼
[CLS] token
+
positional embeddings
        │
        ▼
2-layer Transformer encoder
        │
        ▼
LayerNorm
        │
        ▼
MLP classification head
        │
        ▼
6 logits
```

The source dynamically determines the feature-map shape from the selected ConvNeXt stages rather than hard-coding the spatial dimensions.

---

## Pretrained Weight Adaptation

The original ConvNeXt model expects three RGB channels.

Our input contains four or six channels.

The first convolution is therefore replaced.

The pretrained RGB filters are transferred into the corresponding multispectral channels, while the additional NIR/index channels receive smaller initial weights.

Conceptually:

```text
Pretrained RGB convolution
          │
          ▼
Modified multispectral stem

RGB → corresponding spectral channels
NIR → mean pretrained filter
NDVI/NDWI → small initial weights
```

This allows us to retain useful pretrained representations without treating the multispectral image as an entirely new problem.

---

## Training Methodology

Each fold follows the same procedure.

```text
1. Split by region
       ↓
2. Compute train-only normalization statistics
       ↓
3. Build model
       ↓
4. Initialise optimizer
       ↓
5. Train with augmentation
       ↓
6. Update EMA model
       ↓
7. Evaluate validation Macro-F1
       ↓
8. Save best EMA checkpoint
       ↓
9. Generate OOF predictions
       ↓
10. Generate test probabilities
```

Normalization statistics are calculated separately for each fold using only the training samples.

#### The training loop uses mixed precision when CUDA is available, gradient clipping, EMA and early stopping.

### Why EMA Is Used

Instead of relying directly on the instantaneous model weights, the implementation maintains an:

```text
Exponential Moving Average (EMA)
```

of the model.

Validation is performed using the EMA model.

This provides a more stable representation of the model state across training updates.

The EMA decay is also warmed up during early training rather than immediately applying the full decay to randomly initialized weights.

---

### Learning Rate Strategy

Different parts of the network use different learning rates.

Default configuration:

```text
Backbone      1e-4
Stem          2e-4
Transformer   5e-4
Weight decay  0.05
LLRD          0.85
```

The backbone uses layer-wise learning-rate decay so that earlier layers change more conservatively than later layers.

This is useful because the backbone contains pretrained features while the Transformer head is largely task-specific.

The optimiser implementation creates separate parameter groups for the stem, backbone layers and head.

---

### Cross-Validation Strategy

The final system uses:

```text
5-fold StratifiedGroupKFold
```

with:

```text
group = geographic region
stratification = target class
```

Each validation fold therefore contains regions not used for training in that fold.

Example:

```text
Fold 0

Training:
R01 R02 R03 R04 ... R20

Validation:
R21 R22 R23
```

The exact region allocation is generated from the supplied metadata.

---

### Out-of-Fold Predictions

After each fold is trained, predictions are generated for the fold's validation samples.

These predictions are stored as:

```text
oof_fold0.npz
oof_fold1.npz
...
```

The OOF predictions are then combined into a complete prediction set covering the training data.

This allows:

* unbiased-ish validation analysis under the chosen grouped split
* confusion-matrix analysis
* per-class performance analysis
* region-wise error analysis
* class-bias tuning

The code reconstructs the complete OOF array from fold-level predictions.

---

### Test-Time Augmentation

At inference time, the image is transformed using all eight dihedral views:

```text
Original
Rotate 90°
Rotate 180°
Rotate 270°
Flip
Flip + Rotate 90°
Flip + Rotate 180°
Flip + Rotate 270°
```

Predictions from all views are averaged.

```text
P_TTA = mean(P_1 ... P_8)
```

The inference implementation performs this averaging explicitly.

---

### Fold Ensembling

Each fold produces a probability distribution for each test sample.

The final test probability is:

```text
P_ensemble =
mean(P_fold0,
     P_fold1,
     ...,
     P_fold4)
```

This reduces dependence on the particular region split of an individual model.

The final implementation averages the saved fold probabilities before creating the submission.

---

### OOF Class-Bias Tuning

The OOF predictions are also used to tune class-specific log-probability offsets.

For each class:

```text
Adjusted log probability
=
log(P_class) + bias_class
```

The bias values are selected by searching for an increase in OOF Macro-F1.

This produces two submissions:

```text
submission_v2_nobias.csv
submission_v2_bias.csv
```

#### The source itself warns that this is an in-sample optimisation on OOF predictions, so the apparent gain should not automatically be expected to transfer completely to the leaderboard.

### Current Pipeline — Exact Flow

The complete final pipeline can be summarised as:

```text
                     RAW DATA
                        │
                        ▼
                Region extraction
                        │
                        ▼
             StratifiedGroupKFold
                        │
          ┌─────────────┴─────────────┐
          │           ...             │
          ▼                           ▼
       Fold 0                       Fold 4
          │                           │
          ▼                           ▼
   Train-only stats           Train-only stats
          │                           │
          ▼                           ▼
   Data augmentation          Data augmentation
          │                           │
          ▼                           ▼
    ConvNeXt + ViT             ConvNeXt + ViT
          │                           │
          ▼                           ▼
       EMA model                    EMA model
          │                           │
          ▼                           ▼
     Best checkpoint             Best checkpoint
          │                           │
          └─────────────┬─────────────┘
                        ▼
                 OOF predictions
                        │
                        ▼
                 Error analysis
                        │
                        ▼
                Class-bias tuning
                        │
                        ▼
                Test predictions
                        │
                        ▼
                   8× TTA
                        │
                        ▼
                 Fold ensemble
                        │
                        ▼
                    Submission
```

---

## How to Use the Current Pipeline

### Requirements

Install:

```bash
pip install numpy pandas scikit-learn torch torchvision
```

For the final competition environment, the recommended practice is to use the exact versions recorded in `requirements.txt`.

---

### Dataset layout

The current code expects:

```text
data/
├── train_images.npy
├── train_labels.npy
├── train.csv
├── test_images.npy
├── test.csv
└── sample_submission.csv
```

Expected image shape:

```text
train_images.npy
(N, 4, 64, 64)

test_images.npy
(N_test, 4, 64, 64)
```

The code checks this shape explicitly.

---

### Configure the Data Directory

In the configuration:

```python
CFG = dict(
    data_dir="data",
    ...
)
```

The repository version should avoid hard-coded personal paths.

---

### Run a Smoke Test

Before training on the real dataset:

```bash
python src/geoshift_convnext_vit_v2.py smoke
```

The smoke test creates a small synthetic dataset in the same file format expected by the real pipeline and executes the training/submission flow.

A successful run ends with:

```text
SMOKE TEST PASSED — pipeline runs end-to-end.
```

The current code implements this synthetic end-to-end test.

---

### Run Full Training

```bash
python src/geoshift_convnext_vit_v2.py train
```

This performs:

```text
Data loading
→ region extraction
→ 5-fold CV
→ model training
→ checkpoint saving
→ OOF generation
→ test prediction
→ ensemble
→ submission creation
```

---

### Run Selected Folds

For a faster experiment:

```bash
python src/geoshift_convnext_vit_v2.py train --folds 0
```

Multiple folds:

```bash
python src/geoshift_convnext_vit_v2.py train --folds 0 1 2
```

This is useful when comparing configurations before committing to a complete five-fold run.

---

### Limit the Number of Epochs

```bash
python src/geoshift_convnext_vit_v2.py train --epochs 5
```

This is useful for:

* debugging
* hardware benchmarking
* configuration checks
* quick ablations

---

### Run Different Model Presets

The current pipeline contains several controlled experiment presets.

#### Default tuned model

```bash
python src/geoshift_convnext_vit_v2.py train --preset tuned
```

#### Higher-resolution feature representation

```bash
python src/geoshift_convnext_vit_v2.py train --preset hires
```

#### Per-image normalisation

```bash
python src/geoshift_convnext_vit_v2.py train --preset perimage
```

#### GAP classifier instead of Transformer

```bash
python src/geoshift_convnext_vit_v2.py train --preset gap
```

#### Region-balanced sampling

```bash
python src/geoshift_convnext_vit_v2.py train --preset regbal
```

#### Lower logit-adjustment temperature

```bash
python src/geoshift_convnext_vit_v2.py train --preset tau05
```

The presets are explicitly defined in the source.

---

### Re-generate Test Predictions

Once fold checkpoints exist:

```bash
python src/geoshift_convnext_vit_v2.py predict
```

This:

1. loads the saved fold checkpoints
2. generates test probabilities
3. applies the configured TTA
4. rebuilds the ensemble
5. writes the final submission files

---

### Output Files

The pipeline generates files such as:

```text
v2_tuned/
├── fold0.pth
├── fold1.pth
├── ...
├── history_fold0.csv
├── history_fold1.csv
├── ...
├── oof_fold0.npz
├── oof_fold1.npz
├── ...
├── oof_confusion_matrix.csv
├── oof_region_accuracy.csv
├── test_probs_fold0.npy
├── test_probs_fold1.npy
├── ...
├── test_probs_ensemble.npy
├── submission_v2_nobias.csv
├── submission_v2_bias.csv
└── ensemble_meta.json
```

The checkpoint stores the model configuration, fold, best epoch, validation score and normalization statistics required for later inference.

---

### Error Analysis

The repository includes analysis of:

#### Per-class performance

```text
Precision
Recall
F1
```

#### Confusion matrix

Which classes are systematically confused?

#### Region-wise performance

Which geographic regions are hardest?

#### Training vs validation behaviour

Does the model overfit the training regions?

#### Failure examples

Visualise:

* confident wrong predictions
* low-confidence predictions
* recurring class confusions
* difficult geographic regions

The current pipeline already computes a confusion matrix and region-level accuracy from OOF predictions.

---

### Interpreting Model Failures

A useful way to analyse a failure is:

```text
Wrong prediction
      │
      ├── Spectral ambiguity?
      │
      ├── Spatial ambiguity?
      │
      ├── Class imbalance?
      │
      ├── Region-specific appearance?
      │
      ├── Insufficient spatial resolution?
      │
      └── Annotation ambiguity?
```

This separates **model failure** from **information limitations in the input**.

A misclassified sample is not necessarily evidence that the architecture is inadequate; some classes may genuinely overlap in their spectral/spatial signatures.

---

### Configuration Reference

The main configuration is stored in `CFG`.

Important parameters include:

```python
num_classes = 6
n_folds = 5
epochs = 30
batch_size = 128

lr_backbone = 1e-4
lr_stem = 2e-4
lr_head = 5e-4

weight_decay = 0.05
drop_path = 0.3

backbone_stages = 3
head_type = "vit"

vit_dim = 384
vit_heads = 6
vit_layers = 2

use_indices = True
norm_mode = "global"

mixup_prob = 0.20
cutmix_prob = 0.10

test_tta = True
```

The complete configuration is defined at the top of the training script.

---

### Reproducibility

For every reported experiment, record:

```text
Git commit
Python version
PyTorch version
Torchvision version
GPU / accelerator
Random seed
Preset
Fold IDs
Epoch count
OOF Macro-F1
Leaderboard score
```

#### The training script uses a fixed seed and additionally offsets the seed by fold.

### Results

## Model comparison

Add the actual measured results here:

| Approach              | Validation Strategy | OOF Macro-F1 | Notes                       |
| --------------------- | ------------------- | -----------: | --------------------------- |
| Baseline CNN          | Random split        |      XX.XXXX | Region leakage risk         |
| ConvNeXt baseline     | Random split        |      XX.XXXX | Stronger backbone           |
| Region-aware ConvNeXt | Grouped CV          |      XX.XXXX | More realistic estimate     |
| ConvNeXt + ViT        | Grouped CV          |      XX.XXXX | Spatial token reasoning     |
| + NDVI/NDWI           | Grouped CV          |      XX.XXXX | Spectral feature enrichment |
| + Shift augmentation  | Grouped CV          |      XX.XXXX | Better robustness           |
| Final ensemble        | Grouped CV          |      XX.XXXX | TTA + fold ensemble         |

**Important:** replace every placeholder with the actual measured result. Do not claim an improvement unless the corresponding experiment was actually run.

---

### Main Findings

The main methodological findings from the development process are:

#### Geographic validation matters

A random split does not directly test the competition's geographic generalisation requirement.

#### Spatial representation matters

Giving the Transformer more than a handful of spatial tokens makes the attention mechanism more meaningful.

#### Spectral information matters

NIR and derived indices provide information unavailable to RGB-only models.

#### Radiometric robustness matters

The model should not over-rely on illumination or region-specific intensity distributions.

#### Optimisation details matter

Learning-rate scheduling, EMA and regularisation can materially affect training stability.

#### Inference matters

Fold ensembling and TTA provide a second level of robustness beyond the single trained model.

---

### Limitations

This solution still has limitations.

#### 1. Computational cost

Five-fold training combined with TTA is considerably more expensive than a single model.

#### 2. OOF bias tuning

The class-bias search is optimised using OOF predictions and can overestimate the real-world gain.

#### 3. Dataset dependence

The best feature engineering and augmentation strategy may change for another multispectral dataset.

#### 4. Limited spatial resolution

At 64×64 pixels, some classes may remain inherently ambiguous.

#### 5. Geographic grouping quality

The quality of the region metadata directly affects the validity of the grouped evaluation.

---

### Repository

The repository is organised so that a reviewer can move from:

```text
README
   ↓
Methodology
   ↓
Source Code
   ↓
Experiments
   ↓
Error Analysis
   ↓
Final Results
```

without needing to inspect the entire training script first.

---

## Final Takeaway

The final solution was not obtained simply by making the network larger.

The main development process was:

```text
Initial model
     ↓
Observed failure
     ↓
Identify cause
     ↓
Change methodology / architecture
     ↓
Evaluate under region-aware validation
     ↓
Repeat
     ↓
Final ensemble pipeline
```

The resulting system is therefore a combination of:

**region-aware evaluation + multispectral representation + spatial feature reasoning + shift-oriented augmentation + robust optimisation + ensemble inference.**
