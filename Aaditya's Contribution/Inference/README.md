# RepViT — ImageNet Training & Mobile Inference Pipeline

This folder contains an end-to-end pipeline for training, evaluating, and deploying **RepViT** models. It covers:

- Full **ImageNet-1K training** with advanced augmentations, distillation, and EMA
- **GPU throughput benchmarking**
- **Core ML export** for on-device iOS inference
- A native **iOS application** (`RepViTClassifier`) for mobile inference on iPhone

The implementation is based on the official [RepViT](https://arxiv.org/abs/2307.09283) and [RepViT-SAM](https://arxiv.org/abs/2312.05760) papers (CVPR 2024).

---

## Repository Structure

```
Inference/
├── main.py              # Main training & evaluation script (ImageNet)
├── engine.py            # Per-epoch train/eval loop logic
├── losses.py            # Distillation loss implementation
├── utils.py             # Utilities: BN fusion, EMA, distributed helpers
├── export_coreml.py     # Export trained model to Core ML (.mlmodel)
├── speed_gpu.py         # GPU throughput benchmark (images/sec)
├── eval.sh              # Shell script for quick evaluation
├── requirements.txt     # Python dependencies
├── model/
│   ├── __init__.py
│   └── repvit.py        # RepViT architecture definitions (all variants)
├── data/
│   ├── datasets.py      # ImageNet / CIFAR dataset builders
│   ├── samplers.py      # Repeated Augmentation (RASampler) for training
│   └── threeaugment.py  # Three-augment data augmentation strategy
└── ios/
    └── RepViTClassifier/ # Native Xcode iOS app for on-device inference
```

---

## ImageNet Model Performance

Models are trained on **ImageNet-1K** and deployed on **iPhone 12** with Core ML Tools to measure latency.

| Model  | Top-1 (300e / 450e) | #Params | MACs  | Latency (iPhone 12) |
|:-------|:-------------------:|:-------:|:-----:|:-------------------:|
| M0.9   | 78.7% / 79.1%       | 5.1M    | 0.8G  | 0.9 ms              |
| M1.0   | 80.0% / 80.3%       | 6.8M    | 1.1G  | 1.0 ms              |
| M1.1   | 80.7% / 81.2%       | 8.2M    | 1.3G  | 1.1 ms              |
| M1.5   | 82.3% / 82.5%       | 14.0M   | 2.3G  | 1.5 ms              |
| M2.3   | 83.3% / 83.7%       | 22.9M   | 4.5G  | 2.3 ms              |

> RepViT-M1.0 is the first lightweight model to exceed **80% Top-1 accuracy under 1ms** latency on an iPhone 12.

---

## Environment Setup

### 1. Create a Virtual Environment

```bash
conda create -n repvit python=3.8
conda activate repvit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` includes:**
```
torch
torchvision
timm==0.5.4
fvcore
```

> **Note:** Install a PyTorch version compatible with your hardware. For CUDA 12.1:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## ImageNet Training

### Data Preparation

Download and extract ImageNet train and val images from [image-net.org](http://image-net.org/). The expected directory structure is:

```
/path/to/imagenet/
├── train/
└── val/
```

### Training

To train **RepViT-M0.9** on an 8-GPU machine:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 12346 \
    --use_env main.py \
    --model repvit_m0_9 \
    --data-path ~/imagenet \
    --dist-eval
```

Key training features in `main.py`:
- **Optimizer**: AdamW with adaptive gradient clipping (AGC mode)
- **LR Schedule**: Cosine decay with linear warmup
- **Augmentations**: RandAugment, Mixup, CutMix, Random Erasing, ThreeAugment
- **Regularization**: Label smoothing, weight decay
- **EMA**: Exponential Moving Average of model weights
- **Distillation**: Hard/soft distillation from a teacher model (e.g., `regnety_160`)
- **Logging**: Integrated W&B (Weights & Biases) experiment tracking

### Evaluation

To evaluate RepViT-M1.1 with a pre-trained checkpoint:

```bash
python main.py \
    --eval \
    --model repvit_m1_1 \
    --resume pretrain/repvit_m1_1_distill_300e.pth \
    --data-path ~/imagenet
```

Or use the convenience shell script:

```bash
bash eval.sh
```

---

## GPU Throughput Benchmarking

To measure the GPU throughput (images/sec) of any RepViT variant:

```bash
python speed_gpu.py --model repvit_m0_9
```

Configurable arguments:
- `--model`: Model name (e.g., `repvit_m0_9`, `repvit_m1_1`)
- `--resolution`: Input resolution (default: `224`)
- `--batch-size`: Batch size for benchmarking (default: `2048`)

The script performs a warm-up phase (5 seconds) followed by a timed measurement (10 seconds) to report stable throughput.

**Tip:** Before benchmarking or evaluation, fuse Conv-BN layers for faster inference-time speed:
```python
from timm.models import create_model
import utils

model = create_model('repvit_m0_9')
utils.replace_batchnorm(model)
```

---

## Mobile Deployment — Core ML Export

To export a trained RepViT model to Apple's Core ML format (`.mlmodel`) for on-device inference:

```bash
python export_coreml.py \
    --model repvit_m1_1 \
    --ckpt pretrain/repvit_m1_1_distill_300e.pth \
    --resolution 224
```

The script:
1. Loads the specified model and checkpoint
2. Fuses BatchNorm layers (`replace_batchnorm`) for deployment efficiency
3. Traces the model with TorchScript (`torch.jit.trace`)
4. Converts to Core ML format using `coremltools`
5. Saves the output as `coreml/<model_name>_<resolution>.mlmodel`

---

## iOS Application — RepViTClassifier

The `ios/RepViTClassifier/` directory contains a native **Swift/Xcode** iOS application that loads the exported `.mlmodel` file and runs **real-time image classification** on-device.

### App Structure

```
ios/RepViTClassifier/
├── RepViTClassifier.xcodeproj/   # Xcode project configuration
├── Sources/                      # Swift source files (inference logic, UI)
├── Resources/                    # App resources (labels, assets)
├── Scripts/                      # Build helper scripts
└── project.yml                   # XcodeGen project spec
```

### Running on iPhone

1. Export the model to Core ML (see above) and place the `.mlmodel` in the `Resources/` folder.
2. Open `RepViTClassifier.xcodeproj` in **Xcode 14+**.
3. Select your target iPhone device and press **Run**.
4. Latency measurements on **iPhone 12 (iOS 16)** are obtained via the XCode 14 benchmark tool.

---

## Acknowledgements

This implementation builds on top of:
- [LeViT](https://github.com/facebookresearch/LeViT)
- [PoolFormer](https://github.com/sail-sg/poolformer)
- [EfficientFormer](https://github.com/snap-research/EfficientFormer)

The detection and segmentation pipelines leverage [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

---

## Citation

```bibtex
@inproceedings{wang2024repvit,
  title     = {Repvit: Revisiting mobile cnn from vit perspective},
  author    = {Wang, Ao and Chen, Hui and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {15909--15920},
  year      = {2024}
}

@misc{wang2023repvitsam,
  title         = {RepViT-SAM: Towards Real-Time Segmenting Anything},
  author        = {Ao Wang and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
  year          = {2023},
  eprint        = {2312.05760},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
