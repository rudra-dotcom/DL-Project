# RepViT for CIFAR-10

This repository contains a custom variant of the [RepViT](https://arxiv.org/abs/2307.09283) architecture optimized specifically for the 32x32 CIFAR-10 dataset.

The core modifications to the architecture include:
1. **Stride Reduction**: The `patch_embed` initial convolutions use `stride=1` instead of `stride=2` to prevent early spatial collapse on small 32x32 image inputs.
2. **SE Block Adjustments**: Squeeze-and-Excitation blocks have been selectively retained (alternating) in Stage 1 and Stage 2, and completely removed from Stage 3 and Stage 4 to optimize inference latency without severely impacting accuracy on CIFAR-scale tasks.

## File Overview
* `model/repvit.py`: Contains the `RepViT` architecture definitions, including the newly added `repvit_cifar10` builder function.
* `train_cifar10.py`: The PyTorch training pipeline, heavily leveraging `timm` models and dataset loaders. Includes support for automated Optuna hyperparameter sweeps.
* `run_experiments.ps1`: A PowerShell script that automates the sequential end-to-end training and evaluation of 3 models (Baseline RepViT m1.1, Custom RepViT CIFAR-10, and MobileNetV3 Large).
* `measure_latency.py`: Script to measure inference latency (in ms) and throughput (FPS) for the 3 models using a batch size of 32.
* `generate_report_visuals.py`: Parses the training and latency logs to automatically generate 12 publication-ready evaluation figures (PNG & SVG) inside the `visuals/` folder.
* `infer.py`: Utility script to run inference using an exported model checkpoint on an arbitrary image file (or dummy tensor).
* `test_cifar10_shapes.py`: Lightweight test to verify that the `repvit_cifar10` model successfully builds and accepts 32x32 resolution tensors without dimension crashes.

## Environment Setup (For New Machines)

1. **Create and Activate a Virtual Environment:**
   Run the following commands in your terminal or PowerShell from the root of this repository:
   ```cmd
   python -m venv repvit_env
   ```
   **Windows:**
   ```cmd
   .\repvit_env\Scripts\activate
   ```
   **Linux/macOS:**
   ```bash
   source repvit_env/bin/activate
   ```

2. **Install PyTorch:**
   Install a version of PyTorch compatible with your machine's hardware (CUDA recommended). For example, here is the command for **CUDA 12.1**:
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   *(If you are running CPU only, you can omit the `--index-url` flag or use the respective CPU package).*

3. **Install Dependencies:**
   Install the remaining repository dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## Running the CIFAR-10 Training Experiments

To kick off the comparative training runs (which will automatically download the dataset if it's missing), simply run the `run_experiments.ps1` script:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_experiments.ps1
```

This will run 3 sequential training loops (100 epochs each) utilizing the **AdamW** optimizer:
1. Unmodified **Baseline RepViT (`repvit_m1_1`)**
2. Modified **RepViT CIFAR-10**
3. **MobileNetV3-Large**

All textual logs and evaluation metrics will be saved to individual log files (e.g., `repvit_cifar10_cifar10_training.log`) inside the root directory during the run.

## Benchmarking & Generating Reports

1. **Measure Inference Latency**
   After training, you can benchmark the models for Latency and FPS:
   ```cmd
   python measure_latency.py
   ```
   This generates latency logs (e.g. `repvit_cifar10_latency.log`).

2. **Generate Visuals**
   Create comprehensive 300-dpi graphs and architecture diagrams comparing the models:
   ```cmd
   python generate_report_visuals.py
   ```
   All outputs will be saved to the `visuals/` directory in both PNG and SVG formats.

## Model Validation
If you manually train or modify the architecture configuration, you can verify it does not break by running the shape tester:
```cmd
python test_cifar10_shapes.py
```
