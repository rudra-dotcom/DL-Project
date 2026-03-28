"""
Quick inference script for RepViT.
No ImageNet dataset or internet required.

Usage:
    # Run on your own image:
    python infer.py --img C:/path/to/your/photo.jpg

    # Run with a synthetic random image (just to verify the model works):
    python infer.py --synthetic

    # Use a different model variant:
    python infer.py --model repvit_m0_9 --ckpt pretrain/repvit_m0_9_distill_300e.pth --img photo.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from timm.models import create_model
import utils
import model  # registers RepViT models into timm

# A representative subset of 50 common ImageNet class labels (by index)
IMAGENET_LABELS = {
    0: "tench", 1: "goldfish", 2: "great white shark", 7: "rooster", 8: "hen",
    9: "ostrich", 11: "goldfinch", 15: "robin", 18: "magpie", 20: "water ouzel",
    22: "bald eagle", 30: "tree frog", 33: "loggerhead sea turtle", 56: "king snake",
    63: "worm snake", 65: "box turtle", 72: "platypus", 84: "grey fox",
    130: "flamingo", 145: "king penguin", 151: "Chihuahua", 157: "Maltese",
    162: "beagle", 165: "walker hound", 207: "golden retriever", 208: "Labrador retriever",
    215: "Brittany spaniel", 217: "English springer", 232: "Border collie",
    245: "French bulldog", 258: "Samoyed", 263: "Pembroke Welsh corgi",
    281: "tabby cat", 282: "tiger cat", 285: "Egyptian cat", 291: "lion",
    292: "tiger", 310: "box", 340: "zebra", 380: "African elephant",
    385: "Indian elephant", 386: "lesser panda", 387: "giant panda",
    388: "chimpanzee", 389: "gorilla", 440: "beer bottle", 444: "bicycle",
    456: "bow tie", 463: "broom", 468: "bucket",
    # Many more classes exist, these are just examples for display
}


def get_label(idx):
    return IMAGENET_LABELS.get(idx, f"ImageNet class #{idx}")


def main():
    parser = argparse.ArgumentParser("RepViT Quick Inference")
    parser.add_argument("--model", default="repvit_m1_1", type=str,
                        help="Model variant (default: repvit_m1_1)")
    parser.add_argument("--ckpt", default="pretrain/repvit_m1_1_distill_300e.pth",
                        type=str, help="Path to checkpoint .pth file")
    parser.add_argument("--img", default=None, type=str,
                        help="Path to your image file (jpg/png/etc.)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a random synthetic image (no real image needed)")
    parser.add_argument("--topk", default=5, type=int,
                        help="Number of top predictions to show")
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # --- Build model ---
    print(f"\nLoading model : {args.model}")
    net = create_model(args.model, num_classes=1000, pretrained=False)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    # Filter out distillation-only keys not in the inference model
    model_keys = set(net.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    msg = net.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint    : {args.ckpt}")
    if msg.missing_keys:
        print(f"Missing keys  : {msg.missing_keys}")

    # Fuse Conv-BN layers for faster inference
    utils.replace_batchnorm(net)
    net.to(device)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Params        : {n_params:.1f}M")

    # --- Prepare input ---
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.synthetic:
        print("\nUsing synthetic random image (for model verification only)")
        x = torch.randn(1, 3, 224, 224).to(device)
        img_source = "synthetic random tensor"
    elif args.img:
        img_path = Path(args.img)
        if not img_path.exists():
            print(f"\nError: Image not found at '{args.img}'")
            print("Tip: Pass --synthetic to run with a random image instead.")
            return
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        img_source = str(img_path)
    else:
        print("\nNo image specified. Pass --img <path> or --synthetic")
        print("Example: python infer.py --img C:/Users/you/Pictures/dog.jpg")
        return

    # --- Inference ---
    print(f"\nInput         : {img_source}")
    with torch.no_grad():
        logits = net(x)
        probs = torch.softmax(logits, dim=-1)

    topk_probs, topk_ids = probs.topk(args.topk, dim=-1)
    topk_probs = topk_probs.squeeze().cpu().tolist()
    topk_ids = topk_ids.squeeze().cpu().tolist()

    print(f"\nTop-{args.topk} Predictions:")
    print("─" * 45)
    for i, (idx, prob) in enumerate(zip(topk_ids, topk_probs)):
        label = get_label(idx)
        bar = "█" * int(prob * 35)
        print(f"  {i+1}. [{idx:4d}] {label:<22} {prob*100:5.1f}%  {bar}")
    print("─" * 45)
    print("\nDone!")


if __name__ == "__main__":
    main()
