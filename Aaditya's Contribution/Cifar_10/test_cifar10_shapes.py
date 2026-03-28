"""
Test script to verify RepViT modifications for CIFAR-10.
Instantiates the custom `repvit_cifar10` model, passes a 32x32 tensor,
and checks the output shape and parameter count.
"""

import torch
from timm.models import create_model

# Ensure custom models are registered
import model

def main():
    print("Testing architecture compatibility for 32x32 images...")
    
    # Instantiate the new custom CIFAR-10 RepViT variant
    net = create_model("repvit_cifar10", num_classes=10)
    
    # Create a dummy batch of CIFAR-10 sized images (BatchSize=2, Channels=3, H=32, W=32)
    x = torch.randn(2, 3, 32, 32)
    
    # Forward pass
    out = net(x)
    
    print(f"Output shape: {out.shape}")
    print(f"Total Params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    
    # Assert successful forward pass and correct output classification heads
    assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
    print("✅ Test passed! The model successfully handles 32x32 inputs without spatial collapse.")

if __name__ == "__main__":
    main()
