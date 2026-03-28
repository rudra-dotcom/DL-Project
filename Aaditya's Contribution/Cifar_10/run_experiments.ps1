# Script to run the three models for comparison on CIFAR-10

echo "Starting CIFAR-10 experiments..."

# 1. Unmodified Baseline RepViT (m1.1)
echo "=== 1. Training Baseline Unmodified RepViT (m1.1) ==="
c:\Users\aadit\DL_PROJECT\repvit_env\Scripts\python.exe train_cifar10.py --model_name repvit_m1_1 --epochs 30 --batch_size 128

# 2. Modified RepViT for CIFAR-10
echo "=== 2. Training Modified RepViT (repvit_cifar10) ==="
c:\Users\aadit\DL_PROJECT\repvit_env\Scripts\python.exe train_cifar10.py --model_name repvit_cifar10 --epochs 30 --batch_size 128

# 3. MobileNetV3 Large (from timm)
echo "=== 3. Training MobileNetV3 ==="
c:\Users\aadit\DL_PROJECT\repvit_env\Scripts\python.exe train_cifar10.py --model_name mobilenetv3_large_100 --epochs 30 --batch_size 128

echo "All experiments finished!"
echo "Check '*_cifar10_training.log' files for detailed results of each model."
