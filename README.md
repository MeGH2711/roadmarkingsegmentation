# 🛣️ Road Marking Segmentation

> Semantic segmentation of road markings from aerial/UAV imagery using **DeepLabV3+** (ResNet-101) and **SegFormer** (MiT-B2), combined via an ensemble with soft & hard voting.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Classes](#classes)
- [Project Structure](#project-structure)
- [Models](#models)
  - [DeepLabV3+](#deeplabv3)
  - [SegFormer](#segformer)
  - [Ensemble](#ensemble)
- [Installation](#installation)
- [Usage](#usage)
  - [DeepLabV3+ Training](#deeplabv3-training)
  - [SegFormer Training](#segformer-training)
  - [Ensemble Inference](#ensemble-inference)
  - [Video Inference](#video-inference)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Results](#results)
- [Google Drive Structure](#google-drive-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project performs **pixel-level semantic segmentation** of road markings from images, supporting 13 distinct marking categories plus background. The pipeline includes:

- **YOLO polygon label → semantic mask conversion** for pre-processing ground truth annotations
- **Two independently trained segmentation models**: DeepLabV3+ and SegFormer
- **An ensemble inference module** that fuses both models via soft voting (probability averaging), hard voting (majority vote), or weighted soft voting
- **Optional Test-Time Augmentation (TTA)** with horizontal flipping
- **Video inference** support for frame-by-frame overlay generation
- **Pseudo-label generation** for self-training on unlabelled UAV footage

---

## Dataset

The dataset follows the **YOLO segmentation format**, with polygon annotations stored as `.txt` files (one per image). Labels use normalised polygon coordinates:

```
class_id  x1 y1 x2 y2 ... xn yn
```

Expected directory layout on Google Drive:

```
CVRoadMarkDetection/
└── dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

Labels are pre-converted to grayscale `.png` semantic masks and cached on Drive before training begins (pixel value = `class_id + 1`; `0` = background).

---

## Classes

| ID | Class Name | Colour (RGB) |
|----|-----------|-------------|
| 0  | Background | (0, 0, 0) — Black |
| 1  | BUS LANE | (0, 0, 255) — Blue |
| 2  | Yellow Markings | (0, 255, 255) — Cyan |
| 3  | Line 1 | (0, 255, 0) — Green |
| 4  | Line 2 | (0, 128, 0) — Dark Green |
| 5  | Crossing | (255, 255, 0) — Yellow |
| 6  | Romb | (128, 0, 255) — Purple |
| 7  | SLOW | (255, 0, 0) — Red |
| 8  | Left Arrow | (255, 128, 0) — Orange |
| 9  | Forward Arrow | (255, 0, 255) — Magenta |
| 10 | Fwd Arrow-Left | (128, 255, 0) — Lime |
| 11 | Fwd Arrow-Right | (0, 128, 255) — Sky Blue |
| 12 | Right Arrow | (255, 0, 128) — Pink |
| 13 | Bicycle | (128, 128, 0) — Olive |

**Total: 14 classes (13 road marking categories + background)**

---

## Project Structure

```
road-marking-segmentation/
├── DeepLabV3+_Training_2.ipynb     # DeepLabV3+ training pipeline (2-phase)
├── SegFormer_Training_6.ipynb      # SegFormer (MiT-B2) training pipeline
└── Ensemble_DeepLabV3_SegFormer.ipynb  # Ensemble inference & evaluation
```

All outputs (checkpoints, masks, logs, plots) are saved to Google Drive under `CVRoadMarkDetection/`.

---

## Models

### DeepLabV3+

A custom implementation of **DeepLabV3+** with a **ResNet-101** backbone (ImageNet pre-trained).

**Architecture highlights:**
- Dilated convolutions in ResNet-101 layers 3 & 4 (output stride = 16)
- **ASPP** (Atrous Spatial Pyramid Pooling) with rates `(6, 12, 18)` + global average pooling branch
- Low-level feature projection: `256 → 48` channels
- Decoder: concatenates ASPP output with low-level features → two 3×3 convolutions with dropout → 1×1 classification head

**Parameters:** ~59M

**Loss function:** Combined Cross-Entropy + Dice Loss (weights: 0.6 CE + 0.4 Dice)

**Training strategy:** Two-phase transfer learning
1. **Phase 1 — Pre-training** (50 epochs): Full model trained from ImageNet weights
2. **Phase 2 — Fine-tuning** (50 epochs): Resumed from best pre-train checkpoint with cosine annealing and differential learning rates

---

### SegFormer

Fine-tuned **SegFormer-B2** (`nvidia/mit-b2`) from HuggingFace Transformers.

**Architecture highlights:**
- Mix Transformer (MiT-B2) hierarchical encoder — efficient self-attention without positional encodings
- Lightweight All-MLP decoder head
- No interpolation artefacts at boundaries

**Parameters:** ~25M

**Loss function:** Combined Cross-Entropy + Tversky Loss (weights: 0.5 CE + 0.5 Tversky, α=0.7, β=0.3) — Tversky replaces Dice to penalise false negatives more heavily, improving recall on rare classes.

**Training strategy:**
- Differential learning rates: backbone `6e-5`, decode head `6e-4`
- Optimiser: AdamW (weight decay = 0.01)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- **Weighted Random Sampler**: samples are weighted by the rarest class present in each image to counteract severe class imbalance
- Class-frequency-weighted CE loss (pre-computed from training split)
- 97 epochs total

---

### Ensemble

The `EnsemblePredictor` class fuses DeepLabV3+ and SegFormer logits at inference time.

| Mode | Description |
|------|-------------|
| **Soft Vote** (default) | Average softmax probabilities pixel-wise → argmax. Best when models are well-calibrated. |
| **Hard Vote** | Each model casts a class vote per pixel → majority wins. |
| **Weighted Soft** | Soft vote scaled by per-model confidence weights. |

Optional **Test-Time Augmentation (TTA)**: horizontal flip — predictions on the original and flipped image are averaged before the final argmax.

---

## Installation

This project runs on **Google Colab** (CUDA GPU recommended). Run the dependency cell in each notebook, or install manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.37 accelerate timm
pip install opencv-python-headless albumentations tqdm matplotlib scikit-learn
```

---

## Usage

### DeepLabV3+ Training

Open `DeepLabV3+_Training_2.ipynb` in Google Colab.

1. **Mount Drive** and verify dataset paths in Cell 1–2.
2. **Build mask cache** (Cell 4): Converts YOLO polygon labels to grayscale PNG masks. Comment out if masks already exist.
3. **Visualise samples** (Cell 5): Sanity check images, masks, and overlays.
4. **Configure training** (Cell 3):

```python
NUM_CLASSES     = 14
IMG_SIZE        = 512
BATCH_SIZE      = 8
LR              = 1e-4
PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 50
```

5. **Run Phase 1** (Cell 9): Pre-trains the full model; saves `best_pretrain.pth`.
6. **Run Phase 2** (Cell 10): Fine-tunes from the best pre-train checkpoint; saves `best_finetune.pth`.
7. **Evaluate** (Cell 11–12): Per-class IoU and mIoU on the test set.
8. **Visualise predictions** (Cell 13–15): Side-by-side before/after comparisons.

---

### SegFormer Training

Open `SegFormer_Training_6.ipynb` in Google Colab.

1. **Mount Drive** and set paths.
2. **Build mask cache**: Identical YOLO → PNG conversion (comment out if already done).
3. **Configure training**:

```python
IMG_SIZE   = 512
BATCH_SIZE = 8
NUM_EPOCHS = 97
```

4. **Dataset setup** (with Weighted Random Sampler): Scans all training masks to compute per-sample weights based on the rarest class present in each image.
5. **Compute class weights**: Automatically computed from training loader and saved to Drive (`class_weights.pt`).
6. **Train** the model — checkpoints saved every epoch as `segformer_running.pth`; best mIoU checkpoint saved as `segformer_best.pth`.
7. **Training history** is logged to `training_history.csv` (columns: `epoch`, `train_loss`, `val_loss`, `train_miou`, `val_miou`).

---

### Ensemble Inference

Open `Ensemble_DeepLabV3_SegFormer.ipynb` in Google Colab.

1. Set checkpoint paths in Cell 3:

```python
DEEPLAB_CKPT   = '.../BestModels/best_finetune.pth'
SEGFORMER_CKPT = '.../BestModels/segformer_best.pth'
```

2. Both models are loaded and set to `eval()` mode (Cell 5).
3. Run the full evaluation loop (Cell 8) to get:
   - Per-class IoU for DeepLabV3+, SegFormer, Soft Ensemble, Hard Ensemble
   - Mean IoU comparison table
   - Qualitative side-by-side visualisations saved to `OUTPUT_DIR`

To change ensemble mode or enable TTA:

```python
predictor = EnsemblePredictor(
    models  = [deeplab_model, segformer_model],
    weights = [0.5, 0.5],   # adjust per model confidence
    mode    = 'soft',        # 'soft' | 'hard' | 'weighted_soft'
    tta     = True           # enable horizontal-flip TTA
)
```

---

### Video Inference

DeepLabV3+ supports frame-by-frame video segmentation:

```python
process_video(
    model_path        = CKPT_FT,
    input_video_path  = '/path/to/input.webm',
    output_video_path = '/path/to/output.mp4'
)
```

Each frame is processed at `IMG_SIZE × IMG_SIZE`, the predicted mask is colourised and blended (60% original / 40% mask overlay), then written back at the original video resolution.

---

## Training Details

| Hyperparameter | DeepLabV3+ | SegFormer |
|----------------|-----------|-----------|
| Input size | 512 × 512 | 512 × 512 |
| Batch size | 8 | 8 |
| Optimizer | Adam | AdamW |
| Learning rate | 1e-4 | backbone: 6e-5 / head: 6e-4 |
| Weight decay | 1e-4 | 0.01 |
| Epochs | 50 + 50 (2-phase) | 97 |
| Scheduler | CosineAnnealingLR | ReduceLROnPlateau |
| Loss | 0.6×CE + 0.4×Dice | 0.5×CE + 0.5×Tversky |
| Sampler | Standard shuffle | Weighted Random Sampler |

### Data Augmentation

**DeepLabV3+ (train):**
- RandomResizedCrop (scale 0.5–1.0)
- HorizontalFlip (p=0.5), VerticalFlip (p=0.3)
- RandomRotate90 (p=0.3)
- ColorJitter (brightness, contrast, saturation, hue)
- GaussianBlur (p=0.2)
- ImageNet normalisation

**SegFormer (train):**
- Resize to 512×512
- HorizontalFlip (p=0.5), RandomRotate90 (p=0.5)
- RandomBrightnessContrast (p=0.4)
- GaussNoise (p=0.2)
- ImageNet normalisation

**Drone augmentation** (optional, DeepLabV3+): Simulates UAV top-down perspective with additional VerticalFlip, HueSaturationValue, and ISONoise.

---

## Evaluation

Mean Intersection over Union (mIoU) is computed across foreground classes only (background excluded):

```python
def compute_miou(preds, targets, n_cls):
    ious = []
    for c in range(1, n_cls):   # skip background
        inter = ((preds == c) & (targets == c)).sum()
        union = ((preds == c) | (targets == c)).sum()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0
```

---

## Results

Checkpoints and per-run results are stored in the following Drive locations:

| Model | Checkpoint | Metric file |
|-------|-----------|-------------|
| DeepLabV3+ (pre-train) | `BestModels/best_pretrain.pth` | `phase1_history.png` |
| DeepLabV3+ (fine-tune) | `BestModels/best_finetune.pth` | `phase2_history.png` |
| SegFormer | `BestModels/segformer_best.pth` | `training_history.csv` |
| Ensemble outputs | `Ensemble/` | Per-class IoU table |

---

## Google Drive Structure

```
CVRoadMarkDetection/
├── dataset/
│   ├── train/ valid/ test/
├── DeepLabV3+/
│   ├── masks/
│   │   ├── train/  val/  test/
│   └── DeepLabV3+_Training2/
│       ├── best_pretrain.pth
│       ├── best_finetune.pth
│       ├── phase1_history.png
│       ├── phase2_history.png
│       └── before_vs_after.png
├── SegFormer/
│   ├── masks/
│   │   ├── train/  valid/  test/
│   └── Training6/
│       └── checkpoints/
│           ├── segformer_best.pth
│           ├── segformer_running.pth
│           ├── class_weights.pt
│           ├── id2label.json
│           └── training_history.csv
├── BestModels/
│   ├── best_finetune.pth
│   └── segformer_best.pth
└── Ensemble/
    └── (qualitative outputs, metric CSVs)
```

---

## Acknowledgements

- [DeepLabV3+](https://arxiv.org/abs/1802.02611) — Chen et al., 2018
- [SegFormer](https://arxiv.org/abs/2105.15203) — Xie et al., 2021
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — `nvidia/mit-b2` pretrained weights
- [Albumentations](https://albumentations.ai/) — augmentation library
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — annotation format
