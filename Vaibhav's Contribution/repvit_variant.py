import os
import csv
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    seed=42,
    num_classes=100,
    num_superclasses=20,
    epochs=200,
    batch_size=128,
    base_lr=1e-3,
    weight_decay=0.05,
    warmup_epochs=10,
    label_smoothing=0.1,
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    mixup_cutmix_prob=0.5,  # prob of applying mixup OR cutmix each batch
    aux_loss_weight=0.3,
    checkpoint_every=50,
    checkpoint_dir="checkpoints",
    log_file="training_log.csv",
    latency_warmup=50,
    latency_runs=200,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG["seed"])

# ──────────────────────────────────────────────────────────────────────────────
# ECA Block
# ──────────────────────────────────────────────────────────────────────────────


class ECA(nn.Module):
    """Efficient Channel Attention — 1-D conv over channels, no FC reduction."""

    def __init__(self, channels, k=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)  # B,1,C
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # B,C,1,1
        return x * self.sigmoid(y)


# ──────────────────────────────────────────────────────────────────────────────
# Basic conv building blocks
# ──────────────────────────────────────────────────────────────────────────────


class DWConvBN(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False
        )
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.dw(x))


class PWConvBN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pw = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        return self.bn(self.pw(x))


# ──────────────────────────────────────────────────────────────────────────────
# RepViT-style Mobile Block
# ──────────────────────────────────────────────────────────────────────────────


class MobileBlock(nn.Module):
    """DW3×3 → residual → PW expand → GELU → PW project → optional ECA → residual"""

    def __init__(self, dim, expand_ratio=2, use_eca=False):
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.dw = DWConvBN(dim)
        self.pw1 = nn.Sequential(PWConvBN(dim, hidden), nn.GELU())
        self.pw2 = PWConvBN(hidden, dim)
        self.eca = ECA(dim) if use_eca else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.dw(x) + x)  # dw + residual
        out = self.pw2(self.pw1(out))  # channel mixing
        out = self.eca(out)  # attention or identity
        return out + x  # final residual


class DownsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# RepViT-M0.9 for CIFAR-100
# ──────────────────────────────────────────────────────────────────────────────


class RepViTCIFAR(nn.Module):
    def __init__(self, num_classes=100, num_superclasses=20):
        super().__init__()
        dims = [48, 96, 192, 384]
        depths = [2, 4, 12, 2]

        # Stem: Conv1 stride=1 (32x32), Conv2 stride=2 (16x16)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(
                dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            *[MobileBlock(dims[0], use_eca=True) for _ in range(depths[0])]
        )
        self.down1 = DownsampleBlock(dims[0], dims[1])

        self.stage2 = nn.Sequential(
            *[MobileBlock(dims[1], use_eca=(i % 2 == 0)) for i in range(depths[1])]
        )
        self.down2 = DownsampleBlock(dims[1], dims[2])

        self.stage3 = nn.Sequential(
            *[MobileBlock(dims[2], use_eca=(i % 4 == 0)) for i in range(depths[2])]
        )
        self.down3 = DownsampleBlock(dims[2], dims[3])

        self.stage4 = nn.Sequential(
            *[MobileBlock(dims[3], use_eca=False) for _ in range(depths[3])]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[3], num_classes)
        self.aux_head = nn.Linear(dims[3], num_superclasses)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        return self.pool(x).flatten(1)

    def forward(self, x, return_aux=False):
        feat = self.forward_features(x)
        out = self.head(feat)
        if return_aux:
            return out, self.aux_head(feat)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Utilities & Augmentation
# ──────────────────────────────────────────────────────────────────────────────


def fine_to_super(labels):
    return labels // 5


def mixup_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.size()
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[idx], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_lr(epoch, cfg):
    if epoch < cfg["warmup_epochs"]:
        return cfg["base_lr"] * (epoch + 1) / cfg["warmup_epochs"]
    progress = (epoch - cfg["warmup_epochs"]) / max(
        1, cfg["epochs"] - cfg["warmup_epochs"]
    )
    return cfg["base_lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))


def measure_latency(model, device, input_size=(1, 3, 32, 32), warmup=50, runs=200):
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / runs * 1000
    return round(elapsed, 4)


def get_dataloaders(batch_size):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_ds = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_tf
    )
    val_ds = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=val_tf
    )
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training Functions
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        use_mix = random.random() < cfg["mixup_cutmix_prob"]

        if use_mix:
            if random.random() < 0.5:
                imgs, la, lb, lam = mixup_data(imgs, labels, cfg["mixup_alpha"])
            else:
                imgs, la, lb, lam = cutmix_data(imgs, labels, cfg["cutmix_alpha"])

        logits, aux_logits = model(imgs, return_aux=True)

        if use_mix:
            main_loss = mixed_criterion(criterion, logits, la, lb, lam)
            aux_loss = mixed_criterion(
                criterion, aux_logits, fine_to_super(la), fine_to_super(lb), lam
            )
        else:
            main_loss = criterion(logits, labels)
            aux_loss = criterion(aux_logits, fine_to_super(labels))

        loss = main_loss + cfg["aux_loss_weight"] * aux_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = top5_correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs, return_aux=False)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        top5 = logits.topk(5, dim=1).indices
        top5_correct += top5.eq(labels.unsqueeze(1)).any(1).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total, 100.0 * top5_correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────


def main():
    cfg = CFG
    device = cfg["device"]
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    train_loader, val_loader = get_dataloaders(cfg["batch_size"])
    model = RepViTCIFAR(
        num_classes=cfg["num_classes"], num_superclasses=cfg["num_superclasses"]
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["base_lr"], weight_decay=cfg["weight_decay"]
    )

    # Init CSV
    with open(cfg["log_file"], "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "lr",
                "tr_loss",
                "tr_acc",
                "vl_loss",
                "vl_acc1",
                "vl_acc5",
                "lat_ms",
            ]
        )

    best_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        lr = get_lr(epoch - 1, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg, device
        )
        vl_loss, vl_acc1, vl_acc5 = evaluate(model, val_loader, criterion, device)
        lat = measure_latency(
            model, device, warmup=cfg["latency_warmup"], runs=cfg["latency_runs"]
        )

        with open(cfg["log_file"], "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{lr:.6f}",
                    f"{tr_loss:.4f}",
                    f"{tr_acc:.2f}",
                    f"{vl_loss:.4f}",
                    f"{vl_acc1:.2f}",
                    f"{vl_acc5:.2f}",
                    f"{lat:.4f}",
                ]
            )

        print(f"Epoch {epoch:03d} | Acc: {vl_acc1:.2f}% | Latency: {lat}ms")

        if vl_acc1 > best_acc:
            best_acc = vl_acc1
            torch.save(
                model.state_dict(), os.path.join(cfg["checkpoint_dir"], "best.pth")
            )

        if epoch % cfg["checkpoint_every"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cfg["checkpoint_dir"], f"ckpt_e{epoch}.pth"),
            )


if __name__ == "__main__":
    main()
