import torch
import time
import os
from timm.models import create_model
import model  # registers the custom repvit_cifar10 model

def measure_latency(model_name, ckpt_path, num_runs=200, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model: {model_name}")
    net = create_model(model_name, num_classes=10, pretrained=False)
    net = net.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint, strict=False)
    net.eval()
    
    # Create dummy input (CIFAR-10 size)
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(30):
            _ = net(dummy_input)
    
    print("Measuring latency...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = time.time()
                _ = net(dummy_input)
                torch.cuda.synchronize()
                end = time.time()
            else:
                start = time.time()
                _ = net(dummy_input)
                end = time.time()
            times.append((end - start) * 1000)
    
    avg_latency = sum(times) / len(times)
    fps = 1000.0 / avg_latency * batch_size
    
    # Print to console
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"FPS (batch={batch_size}): {fps:.1f}")
    print("="*70)
    
    # === Save to log file ===
    log_filename = f"{model_name}_latency.log"
    with open(log_filename, "w") as f:
        f.write("=== Latency Measurement Results ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Average Latency: {avg_latency:.2f} ms\n")
        f.write(f"FPS (batch={batch_size}): {fps:.1f}\n")
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("="*50 + "\n")
    
    print(f"✅ Results saved to: {log_filename}")
    return avg_latency, fps


if __name__ == "__main__":
    models_to_test = [
        ("repvit_cifar10", "checkpoints/repvit_cifar10_best.pth"),
        ("repvit_m1_1", "checkpoints/repvit_m1_1_best.pth"),
        ("mobilenetv3_large_100", "checkpoints/mobilenetv3_large_100_best.pth"),
    ]
    
    print("Starting latency measurement for all models...\n")
    for model_name, ckpt_path in models_to_test:
        if os.path.exists(ckpt_path):
            measure_latency(model_name, ckpt_path)
        else:
            print(f"⚠️  Skipping {model_name} - checkpoint not found at {ckpt_path}")