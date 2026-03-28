# RepViT on Low-Resolution Datasets: A CIFAR Study

> **Core Insight:** RepViT was designed and benchmarked for **224×224 ImageNet** images on mobile hardware. When applied to low-resolution datasets like CIFAR-10/100 (32×32), its aggressive early downsampling collapses spatial information before the network has a chance to learn meaningful features — leading to a significant and avoidable accuracy gap.

This repository documents three independent contributions that each identify this flaw and address it differently across CIFAR-10 and CIFAR-100.

---

## The Flaw: Spatial Resolution Sensitivity

RepViT's original stem uses two consecutive stride-2 convolutions, reducing a 224×224 input to 56×56 before the first stage. On a 32×32 CIFAR image, this same stem reduces the feature map to just **8×8** — losing 93.75% of spatial information in the first two layers. This makes the network highly sensitive to input resolution and causes it to underperform significantly on small-image benchmarks despite its strong mobile-device results.

All three contributions below address this root cause, each taking a different approach.

---

## Contributors

| Contributor | Dataset | Model Variant | Best Top-1 |
|-------------|---------|---------------|-----------|
| Rudra | CIFAR-100 | RepViT-M0.9-LR + RepViT-M0.9-LR-RASE | 56.22% |
| Vaibhav | CIFAR-100 | RepViT-M0.9 + ECA + Training Recipe | 74.71% |
| Aaditya | CIFAR-10 | RepViT-M1.1 (stride-1 stem + selective SE) | 91.02 |

---

## Table of Contents

- [Rudra's Contribution](#1-rudras-contribution)
- [Vaibhav's Contribution](#2-vaibhavs-contribution)
- [Aaditya's Contribution](#3-aadityaas-contribution)
- [Latency Note](#latency-note)

---

## 1. Rudra's Contribution

**Dataset:** CIFAR-100 &nbsp;|&nbsp; **Base model:** RepViT-M0.9

### Flaw Identified

RepViT-M0.9 achieves only **43.07% Top-1** on CIFAR-100 out of the box — barely better than MobileNetV3-Large (29.55%) — because its macro design aggressively discards spatial information that is critical at 32×32 resolution.

### Novel Solution

Two new model variants were introduced:

**`repvit_m0_9_lr`** — Low-Resolution aware macro design
- Stem stride changed from `(2, 2)` → `(1, 2)`, preserving the initial 32×32 feature map
- Last downsampling step removed so spatial information is retained longer through the network

**`repvit_m0_9_lr_rase`** — Low-Resolution + Resolution-Aware SE schedule
- Keeps all `repvit_m0_9_lr` changes above
- Adds a resolution-aware SE placement strategy: earlier high-resolution stages receive more SE attention, later low-resolution stages receive less

### Architecture Changes

| Change | Original | Rudra's Variant | Effect |
|--------|----------|-----------------|--------|
| Stem stride | (2, 2) | (1, 2) | ✅ **Accuracy** — preserves spatial info at 32×32 |
| Final downsampling | Present | Removed | ✅ **Accuracy** — retains spatial detail through later stages |
| SE schedule | Uniform | Resolution-aware (more in early, less in late stages) | ✅ **Accuracy** + ✅ **Efficiency** |

### Results

| Model | Best Top-1 (%) | Params (M) | Mean Latency (ms) | Throughput (img/s) |
|-------|---------------|------------|-------------------|--------------------|
| MobileNetV3-Large-100 | 29.55 | 4.330 | 4.814 | 50,704 |
| RepViT-M0.9 (baseline) | 43.07 | 4.757 | 5.385 | 42,667 |
| RepViT-M0.9-LR | 53.99 | 4.758 | 5.497 | 37,648 |
| RepViT-M0.9-LR-RASE | **56.22** | 4.618 | 5.302 | 37,509 |

> `repvit_m0_9_lr_rase` is the best model and uses **138,600 fewer parameters** than baseline RepViT-M0.9 while outperforming it by +13.15 points.

### Code Structure

```
Rudra's contribution/
├── model/
│   └── repvit.py              # RepViT-M0.9-LR and LR-RASE definitions
├── main.py                    # Training and evaluation entry point
├── data/
│   └── datasets.py            # CIFAR-100 dataset builder
├── scripts/
│   ├── train_cifar100_suite.sh
│   ├── eval_cifar100_suite.sh
│   ├── benchmark_suite.sh
│   ├── collect_results.py
│   └── plot_results.py
├── results/generated/
│   ├── cifar100_summary.csv
│   ├── cifar100_curves.csv
│   └── plots/
└── requirements.txt
```

### Setup & Running

```bash
cd "Rudra's contribution"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train all models
bash scripts/train_cifar100_suite.sh data/cifar100 checkpoints/cifar100_suite

# Evaluate
bash scripts/eval_cifar100_suite.sh data/cifar100 checkpoints/cifar100_suite checkpoints/cifar100_eval

# Benchmark latency
bash scripts/benchmark_suite.sh checkpoints/cifar100_suite results/generated/benchmarks

# Regenerate plots
python scripts/collect_results.py \
  --run-root experiment_logs \
  --benchmark-dir results/generated/benchmarks \
  --summary-output results/generated/cifar100_summary.csv \
  --curve-output results/generated/cifar100_curves.csv

python scripts/plot_results.py \
  --summary-csv results/generated/cifar100_summary.csv \
  --curves-csv results/generated/cifar100_curves.csv \
  --output-dir results/generated/plots
```

**Training recipe used in reported runs:**

| Setting | Value |
|---------|-------|
| Epochs | 200 |
| Batch size | 128 |
| Optimizer | AdamW |
| LR | 0.00025 |
| Warmup | 5 epochs |
| Scheduler | Cosine |
| Augmentation | RandAugment `rand-m9-mstd0.5-inc1`, MixUp 0.8, CutMix 1.0, Random Erasing 0.25, Repeated Aug |
| Distillation | Disabled |

---

## 2. Vaibhav's Contribution

**Dataset:** CIFAR-100 &nbsp;|&nbsp; **Base model:** RepViT-M0.9

### Flaw Identified

Beyond the stem stride issue, the original RepViT also uses SE blocks uniformly across all stages. SE blocks rely on an FC bottleneck (C → C/4 → C) that is both computationally redundant in deeper stages and architecturally suboptimal — later stages already have sufficient representational power and benefit less from channel recalibration.

### Novel Solution

A two-pronged approach — architectural changes to the attention mechanism and placement strategy, combined with a stronger training recipe.

### Architecture Changes

| Change | Original | This Variant | Effect |
|--------|----------|--------------|--------|
| Stem Conv 1 stride | stride=2 | **stride=1** (preserves 32×32) | ✅ **Accuracy** — retains spatial info for small CIFAR images |
| Stem Conv 2 stride | stride=2 | stride=2 (unchanged) | — |
| Attention type | SE (FC bottleneck C→C/4→C) | **ECA** (1D conv, no reduction) | ✅ **Accuracy** + ✅ **Efficiency** — better recalibration, fewer params, lower latency |
| Stage 1 (2 blocks) | Uniform SE | **All blocks** have ECA | ✅ **Accuracy** — early layers have most feature diversity, benefit most from recalibration |
| Stage 2 (4 blocks) | Uniform SE | **Alternate blocks** (idx 0, 2) | ✅ **Accuracy** + ✅ **Efficiency** — good coverage at half the attention cost |
| Stage 3 (12 blocks) | Uniform SE | **Every 4th block** (idx 0, 4, 8) | ✅ **Efficiency** — deep layers need less recalibration |
| Stage 4 (2 blocks) | Some SE | **No attention** | ✅ **Efficiency** — final stage has sufficient power; removing attention saves latency |
| Auxiliary head | None | **Superclass head** (20 classes, train-time only) | ✅ **Accuracy** — regularizes backbone via CIFAR-100's class hierarchy; zero inference cost |

**Channel config:** `[48, 96, 192, 384]` &nbsp;|&nbsp; **Block depths:** `[2, 4, 12, 2]`

### Training Recipe Changes

| Change | Original | This Variant | Effect |
|--------|----------|--------------|--------|
| Augmentation | Basic flip + crop | **RandAugment** (2 ops, magnitude 9) | ✅ **Accuracy** — stronger regularization |
| MixUp | Not used | **MixUp** alpha=0.8 | ✅ **Accuracy** — effective on 100 fine-grained classes |
| CutMix | Not used | **CutMix** alpha=1.0 | ✅ **Accuracy** — forces learning from partial regions |
| Mix strategy | — | Random MixUp or CutMix per batch (50% prob) | ✅ **Accuracy** — diverse augmentation without overuse |
| Label smoothing | 0.1 | 0.1 (unchanged) | — |
| LR schedule | Cosine | **Cosine + linear warmup** (10 epochs) | ✅ **Accuracy** — prevents early instability |
| Optimizer | AdamW | AdamW (unchanged) | — |
| Gradient clipping | Not used | **Clip norm 5.0** | ✅ **Accuracy** — stabilizes training with noisy mixed-aug gradients |
| Auxiliary loss | Not used | **Superclass CE loss**, weight=0.3 | ✅ **Accuracy** — structured regularization via class hierarchy |
| Epochs | 300 | 200 | ⚠️ **Accuracy tradeoff** — fewer epochs; 300 would likely improve further |

> All training recipe changes are purely training-time — **zero impact on inference latency**.

### Results

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **74.71%** |
| Best Val Acc@5 | **92.18%** |
| Avg Inference Latency | ~3.7 ms |
| Model Parameters | ~5.1 M |

### Code Structure

```
Vaibhav's contribution/
├── repvit_cifar100.py     # Single-file training script (model + training loop)
├── plot_training.py       # Generates 7 diagnostic plots from training_log.csv
├── requirements.txt
└── README.md

# Generated after training:
├── checkpoints/
│   ├── best.pth
│   ├── ckpt_epoch50.pth
│   ├── ckpt_epoch100.pth
│   ├── ckpt_epoch150.pth
│   └── ckpt_epoch200.pth
├── training_log.csv
└── plots/
    ├── 1_train_val_loss.png
    ├── 2_train_val_acc1.png
    ├── 3_val_acc1_vs_acc5.png
    ├── 4_overfitting_monitor.png
    ├── 5_latency.png
    ├── 6_lr_schedule.png
    └── 7_val_acc_delta.png
```

### Setup & Running

```bash
pip install -r requirements.txt

# Train (downloads CIFAR-100 automatically)
python repvit_cifar100.py

# Generate plots
python plot_training.py
python plot_training.py --log path/to/training_log.csv --out my_plots/

# Resume from checkpoint
```
```python
checkpoint = torch.load("checkpoints/best.pth")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"]
```

All hyperparameters live in the `CFG` dict at the top of `repvit_cifar100.py` and can be edited directly:

```python
CFG = dict(
    epochs            = 200,
    batch_size        = 128,
    base_lr           = 1e-3,
    weight_decay      = 0.05,
    warmup_epochs     = 10,
    label_smoothing   = 0.1,
    mixup_alpha       = 0.8,
    cutmix_alpha      = 1.0,
    mixup_cutmix_prob = 0.5,
    aux_loss_weight   = 0.3,
    checkpoint_every  = 50,
)
```

---

## 3. Aaditya's Contribution

**Dataset:** CIFAR-10 &nbsp;|&nbsp; **Base model:** RepViT-M1.1

### Flaw Identified

The same spatial collapse issue from the original stem design, now studied on CIFAR-10. RepViT's `patch_embed` uses stride=2 convolutions that reduce a 32×32 input to 8×8 within the first two layers, discarding spatial structure before any meaningful feature learning occurs.

### Novel Solution

**`repvit_cifar10`** — a custom variant with two targeted changes:

- **Stride reduction:** `patch_embed` initial convolutions changed to `stride=1` to prevent early spatial collapse on 32×32 inputs
- **Selective SE retention:** SE blocks are kept alternating in Stage 1 and Stage 2 for feature diversity, and fully removed from Stage 3 and Stage 4 to reduce inference latency without hurting accuracy at CIFAR scale

### Architecture Changes

| Change | Original | Aaditya's Variant | Effect |
|--------|----------|-------------------|--------|
| `patch_embed` stride | stride=2 | **stride=1** | ✅ **Accuracy** — prevents spatial collapse on 32×32 |
| Stage 1 & 2 SE | Uniform | **Alternating** SE blocks | ✅ **Accuracy** + ✅ **Efficiency** |
| Stage 3 & 4 SE | Present | **Removed entirely** | ✅ **Efficiency** — no accuracy cost at CIFAR scale |

### Code Structure

```
Aaditya's contribution/
├── model/
│   └── repvit.py              # repvit_cifar10 builder + modified arch
├── train_cifar10.py           # Training pipeline (timm + Optuna HPO support)
├── measure_latency.py         # Latency and FPS benchmarking
├── generate_report_visuals.py # 12 publication-ready plots (PNG + SVG)
├── infer.py                   # Single-image inference utility
├── test_cifar10_shapes.py     # Shape sanity check for 32×32 inputs
├── run_experiments.ps1        # PowerShell script: trains all 3 models sequentially
└── requirements.txt
```

### Setup & Running

```bash
# Create environment
python -m venv repvit_env

# Activate (Linux/macOS)
source repvit_env/bin/activate

# Activate (Windows)
.\repvit_env\Scripts\activate

# Install PyTorch (CUDA 12.1 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

```powershell
# Train all 3 models (Baseline RepViT M1.1, RepViT CIFAR-10, MobileNetV3-Large)
powershell.exe -ExecutionPolicy Bypass -File .\run_experiments.ps1

# Benchmark latency and throughput
python measure_latency.py

# Generate all plots
python generate_report_visuals.py

# Verify model accepts 32x32 inputs without errors
python test_cifar10_shapes.py
```

Models trained: Baseline RepViT-M1.1, `repvit_cifar10`, MobileNetV3-Large — 100 epochs each with AdamW.

---

## Latency Note

Latency numbers across all three contributions are **hardware-specific and backend-specific** and should not be compared directly to the latency reported in the original RepViT paper.

The paper reports mobile latency on **iPhone 12 using Core ML and Apple's benchmark tool**. All experiments here were measured on **CUDA hardware using PyTorch**. Backend, compiler, runtime, and input resolution all affect latency significantly.

Results within each contribution are valid for **relative comparison on the same machine**. Cross-contribution latency comparisons are not meaningful unless run on identical hardware.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- numpy, pandas, matplotlib
- CUDA-capable GPU (strongly recommended)

```bash
pip install -r requirements.txt
```
