# Remote CUDA Runbook

## 1. Copy the project to the GPU machine

```bash
scp -r /path/to/project user@server:/path/to/workdir/repvit-project
ssh user@server
cd /path/to/workdir/repvit-project
```

## 2. Check the GPU and choose the PyTorch wheel

```bash
nvidia-smi
```

Use the PyTorch wheel that matches the cluster recommendation. Common choices:

- CUDA 12.1: `cu121`
- CUDA 11.8: `cu118`

## 3. Create the environment

```bash
bash scripts/install_cuda_env.sh cu121
source .venv/bin/activate
```

If your cluster is on CUDA 11.8, replace `cu121` with `cu118`.

## 4. Optional: warm-start RepViT from the official ImageNet checkpoint

```bash
mkdir -p pretrain
wget -O pretrain/repvit_m0_9_distill_300e.pth \
  https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_9_distill_300e.pth
```

Then export this before training:

```bash
export REPVIT_FINETUNE_CKPT=pretrain/repvit_m0_9_distill_300e.pth
```

## 5. Train the CIFAR-100 suite

Single GPU:

```bash
export DEVICE=cuda
export EPOCHS=200
export BATCH_SIZE=128
bash scripts/train_cifar100_suite.sh data/cifar100 checkpoints/cifar100_suite
```

Multi-GPU:

```bash
export DEVICE=cuda
export NUM_GPUS=4
export EPOCHS=200
export BATCH_SIZE=128
bash scripts/train_cifar100_suite.sh data/cifar100 checkpoints/cifar100_suite
```

The script trains:

- `mobilenetv3_large_100`
- `repvit_m0_9`
- `repvit_m0_9_lr`
- `repvit_m0_9_lr_rase`

## 6. Evaluate the best checkpoints

```bash
bash scripts/eval_cifar100_suite.sh data/cifar100 checkpoints/cifar100_suite checkpoints/cifar100_eval
```

## 7. Benchmark latency and throughput on the GPU

```bash
bash scripts/benchmark_suite.sh checkpoints/cifar100_suite results/generated/benchmarks
```

## 8. Collect CSV summaries and generate plots

```bash
python scripts/collect_results.py \
  --run-root checkpoints/cifar100_suite \
  --benchmark-dir results/generated/benchmarks

python scripts/plot_results.py \
  --summary-csv results/generated/cifar100_summary.csv \
  --curves-csv results/generated/cifar100_curves.csv \
  --output-dir results/generated/plots
```

## 9. Inspect the spatial-resolution claim directly

```bash
python scripts/inspect_feature_maps.py --model repvit_m0_9 --input-size 32 --num-classes 100
python scripts/inspect_feature_maps.py --model repvit_m0_9_lr --input-size 32 --num-classes 100
```

This is the evidence for the paper extension: vanilla RepViT downsamples too aggressively on low-resolution inputs, while `repvit_m0_9_lr` preserves spatial structure longer.
