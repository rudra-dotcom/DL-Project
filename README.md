# RepViT on CIFAR-100: Rudra's Contribution

This submission packages only the code, logs, benchmark outputs, and plots that were actually used for the CIFAR-100 study based on the RepViT paper.

The curated project code lives in [`Rudra's contribution/`](./Rudra%27s%20contribution/).  


## 1. What came from the original RepViT codebase?

The original RepViT repository already provided:

- the core **PyTorch implementation** of RepViT
- **CUDA-based training/evaluation** workflows for the original paper tasks
- model definitions, utility functions, and official ImageNet/mobile benchmarking context

I did **not** create the base PyTorch/CUDA support from scratch.

## 2. What did I add for this project?

My contribution was the CIFAR-100 study and the accompanying experiment pipeline:

- a clean **CIFAR-100 experiment setup**
- two custom models:
  - `repvit_m0_9_lr`
  - `repvit_m0_9_lr_rase`
- CIFAR-100 training/evaluation/benchmark/plot scripts
- experiment logs, benchmark JSONs, and plots
- the study write-up and reproducibility notes in this README

The main code changes are inside:

- [`Rudra's contribution/model/repvit.py`](./Rudra%27s%20contribution/model/repvit.py)
- [`Rudra's contribution/main.py`](./Rudra%27s%20contribution/main.py)
- [`Rudra's contribution/data/datasets.py`](./Rudra%27s%20contribution/data/datasets.py)
- [`Rudra's contribution/scripts/`](./Rudra%27s%20contribution/scripts/)

## 3. Project idea

The paper optimizes RepViT for **ImageNet-1K**, **224x224** images, and **mobile-device latency**.

My project asks:

> What happens if the same family of models is moved to a low-resolution dataset such as CIFAR-100 (`32x32`)?

### Insight / flaw identified

RepViT's original macro design is strong for higher-resolution mobile inference, but on CIFAR-100 the network downsamples spatial information very aggressively.

### Novel solution

I introduced two variants:

1. **`repvit_m0_9_lr`**
   - changed the stem stride pattern from `(2, 2)` to `(1, 2)`
   - removed the last downsampling step so spatial information is preserved longer

2. **`repvit_m0_9_lr_rase`**
   - keeps the low-resolution-aware macro design above
   - adds a **resolution-aware SE schedule**
   - earlier high-resolution stages get more SE usage, later low-resolution stages get less

## 4. Folder contents

Inside [`Rudra's contribution/`](./Rudra%27s%20contribution/) you will find:

- source code used for the CIFAR-100 study
- benchmark JSON files
- regenerated result CSVs
- plots
- per-model experiment logs (`args.txt` and `log.txt`)

Intentionally omitted:

- SAM
- detection / segmentation project code used by the original repo but not by this submission
- model checkpoint binaries (`.pth`) to keep the package clean and lightweight

## 5. Environment setup

From inside [`Rudra's contribution/`](./Rudra%27s%20contribution/):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA server, the provided helper is:

```bash
bash scripts/install_cuda_env.sh cu121
source .venv/bin/activate
```

## 6. Dataset

This study uses **CIFAR-100**.

You only need to provide a root folder. The dataset download is handled automatically by the dataset builder.

Example:

```bash
mkdir -p data/cifar100
```

Then use:

```bash
--data-set CIFAR --data-path data/cifar100 --input-size 32
```

## 7. Training recipe actually used in the reported run

The reported run is the same fair recipe for all 4 models:

- dataset: CIFAR-100
- image size: `32x32`
- epochs: `200`
- batch size: `128`
- optimizer: AdamW
- scheduler: cosine
- effective learning rate recorded in the run: `0.00025`
- warmup: `5` epochs
- distillation: **disabled**
- training from scratch: **yes**
- augmentations:
  - `RandAugment rand-m9-mstd0.5-inc1`
  - mixup `0.8`
  - cutmix `1.0`
  - label smoothing `0.1`
  - random erasing `0.25`
  - repeated augmentation

## 8. Reproducibility commands

### Train all 4 models

```bash
export DEVICE=cuda
export NUM_GPUS=1
export EPOCHS=200
export BATCH_SIZE=128
bash scripts/train_cifar100_suite.sh data/cifar100 checkpoints/cifar100_suite
```

### Evaluate best checkpoints

```bash
bash scripts/eval_cifar100_suite.sh data/cifar100 checkpoints/cifar100_suite checkpoints/cifar100_eval
```

### Benchmark latency / throughput

```bash
export DEVICE=cuda
bash scripts/benchmark_suite.sh checkpoints/cifar100_suite results/generated/benchmarks
```

### Regenerate summary CSV + plots

```bash
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

## 8A. Note on `export_coreml.py`

The original RepViT repo includes an `export_coreml.py` helper. Its purpose is to convert a trained PyTorch model into a CoreML model for Apple-device deployment.

Why this exists:

- the paper reports mobile latency using **iPhone 12 + Core ML + Apple's benchmark tool**
- so CoreML export is the first step toward that deployment path

Why it is not central to this submission:

- this project's main experiments were run and benchmarked in **PyTorch/CUDA**
- exporting to CoreML alone does **not** reproduce the paper's published latency
- to get paper-style latency, the exported model still needs to be tested on Apple hardware, especially iPhone

So for this submission:

- `export_coreml.py` is relevant as a bridge from PyTorch to Apple deployment
- but the main analysis and plots remain based on the CUDA experiments actually executed for this project

## 9. Final results

| Model | Best Top-1 (%) | Params (M) | Mean Latency (ms) | Throughput (img/s) |
|---|---:|---:|---:|---:|
| MobileNetV3-Large-100 | 29.55 | 4.330 | 4.814 | 50,704.86 |
| RepViT-M0.9 | 43.07 | 4.757 | 5.385 | 42,667.69 |
| RepViT-M0.9-LR | 53.99 | 4.758 | 5.497 | 37,648.29 |
| RepViT-M0.9-LR-RASE | 56.22 | 4.618 | 5.302 | 37,509.08 |

### Main findings

- RepViT-M0.9 already beats MobileNetV3 by **+13.52** points.
- `repvit_m0_9_lr` beats original RepViT by **+10.92** points.
- `repvit_m0_9_lr_rase` beats original RepViT by **+13.15** points.
- `repvit_m0_9_lr_rase` is the **best model** and also uses **138,600 fewer parameters** than baseline RepViT-M0.9.

This is the main project result:

> Preserving spatial detail longer and redistributing SE usage toward earlier stages substantially improves CIFAR-100 accuracy.

## 10. Latency note

The latency measured in this project should **not** be compared directly to the latency reported in the paper.

Why:

- the paper reports mobile latency on **iPhone 12 using Core ML**
- this project measures latency on **different CUDA hardware**
- backend, compiler support, runtime, and input size all change latency behavior

So it is valid that MobileNetV3 is faster on this hardware while RepViT is more favorable in the paper's mobile setup.

This submission therefore reports latency as:

- **hardware-specific**
- **backend-specific**
- useful for relative comparison on the measured machine only

If you also run the optional CoreML benchmark flow, that result is a better approximation to the paper's deployment style than PyTorch CUDA, but it is still **not identical** to the paper unless measured on iPhone 12 with the Apple benchmark tool.

## 11. Included plots

### Validation accuracy curves

![Accuracy Curves](./Rudra%27s%20contribution/results/generated/plots/accuracy_curves.png)

### Training loss curves

![Loss Curves](./Rudra%27s%20contribution/results/generated/plots/loss_curves.png)

### Best accuracy bar chart

![Best Accuracy](./Rudra%27s%20contribution/results/generated/plots/best_accuracy_bar.png)

### Accuracy vs latency

![Accuracy vs Latency](./Rudra%27s%20contribution/results/generated/plots/accuracy_vs_latency.png)

### Accuracy vs parameters

![Accuracy vs Params](./Rudra%27s%20contribution/results/generated/plots/accuracy_vs_params.png)

## 12. Supplementary assets

These are kept outside the GitHub-ready folder in the submission package:

- `Submission Assets/Original Paper Diagrams/`
- `Submission Assets/Custom Diagrams/`
- `Submission Assets/Video Scripts/`

These include:

- original paper figure extracts
- custom architecture diagrams for my changes
- prepared speaking scripts for the two submission videos
