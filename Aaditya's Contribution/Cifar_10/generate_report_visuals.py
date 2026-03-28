"""
Generate comprehensive report & presentation visuals for RepViT CIFAR-10 project.
Produces high-quality PNG and SVG figures in the visuals/ directory.
"""

import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
VISUALS_DIR = ROOT / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)

# ─── Design Tokens ──────────────────────────────────────────────────────────
PALETTE = {
    "repvit_cifar10":          "#0D9488",   # teal-600
    "repvit_m1_1":             "#3B82F6",   # blue-500
    "mobilenetv3_large_100":   "#F59E0B",   # amber-500
    "bg":          "#FFFFFF",
    "fg":          "#111827",
    "subtle":      "#64748B",
    "grid":        "#E2E8F0",
    "border":      "#CBD5E1",
    "green":       "#10B981",
    "red":         "#EF4444",
    "highlight":   "#7C3AED",
}

LABELS = {
    "repvit_cifar10":          "RepViT-CIFAR10 (Ours)",
    "repvit_m1_1":             "RepViT-m1.1 (Baseline)",
    "mobilenetv3_large_100":   "MobileNetV3-Large",
}

SHORT_LABELS = {
    "repvit_cifar10":          "RepViT-CIFAR10",
    "repvit_m1_1":             "RepViT-m1.1",
    "mobilenetv3_large_100":   "MobileNetV3-L",
}

TRAINING_LOGS = {
    "repvit_cifar10": ROOT / "repvit_cifar10_cifar10_training.log",
    "repvit_m1_1": ROOT / "repvit_m1_1_cifar10_training.log",
    "mobilenetv3_large_100": ROOT / "mobilenetv3_large_100_cifar10_training.log",
}

LATENCY_LOGS = {
    "repvit_cifar10": ROOT / "repvit_cifar10_latency.log",
    "repvit_m1_1": ROOT / "repvit_m1_1_latency.log",
    "mobilenetv3_large_100": ROOT / "mobilenetv3_large_100_latency.log",
}

PARAMS = {
    "repvit_cifar10": 4.52,
    "repvit_m1_1": 7.78,
    "mobilenetv3_large_100": 4.21,
}

EPOCH_PATTERN = re.compile(
    r"Epoch:\s*(\d+)\s*\|\s*LR:\s*([0-9.]+)\s*\|\s*Train Acc:\s*([0-9.]+)%\s*\|\s*"
    r"Test Acc:\s*([0-9.]+)%\s*\(Best:\s*([0-9.]+)%\)"
)
LATENCY_PATTERN = re.compile(r"Average Latency:\s*([0-9.]+)\s*ms")
FPS_PATTERN = re.compile(r"FPS \(batch=32\):\s*([0-9.]+)")


# ─── Global Style ────────────────────────────────────────────────────────────
def configure_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["border"],
        "axes.labelcolor": PALETTE["fg"],
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.color": PALETTE["subtle"],
        "ytick.color": PALETTE["subtle"],
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
        "font.family": "sans-serif",
        "font.size": 11,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


# ─── Parsing ─────────────────────────────────────────────────────────────────
def parse_training_log(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EPOCH_PATTERN.search(line)
        if m:
            epoch, lr, train_acc, test_acc, best_acc = m.groups()
            rows.append({
                "epoch": int(epoch), "lr": float(lr),
                "train_acc": float(train_acc), "test_acc": float(test_acc),
                "best_acc": float(best_acc),
            })
    if not rows:
        raise ValueError(f"No epoch rows parsed from {path}")
    return rows


def parse_latency_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lat = LATENCY_PATTERN.search(text)
    fps = FPS_PATTERN.search(text)
    if not lat or not fps:
        raise ValueError(f"Could not parse latency from {path}")
    return {"latency_ms": float(lat.group(1)), "fps": float(fps.group(1))}


def load_all():
    histories, metrics = {}, {}
    for name, log_path in TRAINING_LOGS.items():
        history = parse_training_log(log_path)
        latency = parse_latency_log(LATENCY_LOGS[name])
        histories[name] = history
        metrics[name] = {
            "best_test_acc": max(r["test_acc"] for r in history),
            "final_test_acc": history[-1]["test_acc"],
            "epochs": len(history),
            "latency_ms": latency["latency_ms"],
            "fps": latency["fps"],
            "params_m": PARAMS[name],
        }
    return histories, metrics


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(VISUALS_DIR / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  ✓ {stem}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Test Accuracy Curves
# ══════════════════════════════════════════════════════════════════════════════
def fig_test_accuracy(histories, metrics):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    for name in order:
        h = histories[name]
        ep = [r["epoch"] for r in h]
        ta = [r["test_acc"] for r in h]
        ax.plot(ep, ta, linewidth=2.4, color=PALETTE[name],
                label=f'{LABELS[name]}  (best {metrics[name]["best_test_acc"]:.2f}%)',
                alpha=0.92)
        best_i = int(np.argmax(ta))
        ax.scatter(ep[best_i], ta[best_i], s=90, color=PALETTE[name],
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(f"{ta[best_i]:.1f}%", (ep[best_i], ta[best_i]),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=9.5, color=PALETTE[name], fontweight="bold")

    ax.set_title("CIFAR-10 — Test Accuracy Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(1, 100)
    ax.set_ylim(20, 95)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.legend(loc="lower right", borderaxespad=1.0)
    fig.tight_layout()
    save(fig, "fig1_test_accuracy_curves")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Train vs Test Accuracy (overfitting analysis)
# ══════════════════════════════════════════════════════════════════════════════
def fig_train_vs_test(histories, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    for idx, name in enumerate(order):
        ax = axes[idx]
        h = histories[name]
        ep = [r["epoch"] for r in h]
        tra = [r["train_acc"] for r in h]
        tea = [r["test_acc"] for r in h]
        ax.plot(ep, tra, linewidth=2, color=PALETTE[name], label="Train", linestyle="--", alpha=0.7)
        ax.plot(ep, tea, linewidth=2, color=PALETTE[name], label="Test")
        # fill gap
        ax.fill_between(ep, tea, tra, alpha=0.08, color=PALETTE[name])
        gap = [tra[i] - tea[i] for i in range(len(ep))]
        final_gap = gap[-1]
        ax.set_title(f"{SHORT_LABELS[name]}\n(gap at end: {final_gap:.1f}%)", fontsize=12)
        ax.set_xlabel("Epoch")
        if idx == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(1, 100)
    fig.suptitle("Generalization Gap — Train vs Test Accuracy", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_train_vs_test")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Learning Rate Schedule
# ══════════════════════════════════════════════════════════════════════════════
def fig_lr_schedule(histories):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # All models use the same cosine schedule from lr=1e-3, just plot one
    name = "repvit_cifar10"
    h = histories[name]
    ep = [r["epoch"] for r in h]
    lr = [r["lr"] for r in h]
    ax.plot(ep, lr, linewidth=2.2, color=PALETTE["highlight"])
    ax.fill_between(ep, 0, lr, alpha=0.10, color=PALETTE["highlight"])
    ax.set_title("Cosine Annealing LR Schedule (All Models)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_xlim(1, 100)
    ax.grid(True, alpha=0.4)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, -3))
    fig.tight_layout()
    save(fig, "fig3_lr_schedule")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Bar Chart — Best Accuracy Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_accuracy_bars(metrics):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    labels = [SHORT_LABELS[n] for n in order]
    accs = [metrics[n]["best_test_acc"] for n in order]
    colors = [PALETTE[n] for n in order]

    bars = ax.bar(labels, accs, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=PALETTE["fg"])

    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Best Test Accuracy — Model Comparison")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig4_accuracy_bars")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Bar Chart — Latency & FPS
# ══════════════════════════════════════════════════════════════════════════════
def fig_latency_fps(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    labels = [SHORT_LABELS[n] for n in order]
    lats = [metrics[n]["latency_ms"] for n in order]
    fpss = [metrics[n]["fps"] for n in order]
    colors = [PALETTE[n] for n in order]

    bars1 = ax1.barh(labels, lats, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars1, lats):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f} ms", va="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_title("Inference Latency (batch=32)")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.4)

    bars2 = ax2.barh(labels, fpss, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars2, fpss):
        ax2.text(bar.get_width() + 80, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f}", va="center", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Throughput (images/sec)")
    ax2.set_title("Throughput (FPS, batch=32)")
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.4)

    fig.suptitle("Latency & Throughput Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_latency_fps")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Accuracy–Latency Scatter (bubble = params)
# ══════════════════════════════════════════════════════════════════════════════
def fig_tradeoff(metrics):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    for name in order:
        m = metrics[name]
        size = 250 * (m["params_m"] / 4.0)
        ax.scatter(m["latency_ms"], m["best_test_acc"], s=size, 
                   color=PALETTE[name], alpha=0.85, edgecolors="white", linewidths=2, zorder=5)
        ax.annotate(f'{SHORT_LABELS[name]}\n{m["params_m"]:.1f}M params',
                    (m["latency_ms"], m["best_test_acc"]),
                    textcoords="offset points", xytext=(14, 10),
                    fontsize=10, color=PALETTE["fg"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=PALETTE["border"], lw=1.2))

    ax.set_title("Accuracy–Latency Trade-off")
    ax.set_xlabel("Inference Latency (ms, batch=32)")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_xlim(2.0, 7.0)
    ax.set_ylim(70, 95)
    ax.grid(True, alpha=0.5)
    ax.text(2.2, 71, "Bubble size ∝ parameter count  •  Upper-left = ideal",
            fontsize=10, color=PALETTE["subtle"], style="italic")
    fig.tight_layout()
    save(fig, "fig6_accuracy_latency_tradeoff")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Parameter Count Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_params(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    labels = [SHORT_LABELS[n] for n in order]
    params = [metrics[n]["params_m"] for n in order]
    colors = [PALETTE[n] for n in order]

    bars = ax.bar(labels, params, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{p:.2f}M", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Size — Parameter Count")
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig7_parameter_count")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Radar / Summary Spider Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig_radar(metrics):
    categories = ["Accuracy\n(%)", "1/Latency\n(speed)", "1/Params\n(efficiency)", "Throughput\n(FPS)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    # Normalize to 0-1: higher is better
    max_acc = max(metrics[n]["best_test_acc"] for n in order)
    max_speed = max(1 / metrics[n]["latency_ms"] for n in order)
    max_eff = max(1 / metrics[n]["params_m"] for n in order)
    max_fps = max(metrics[n]["fps"] for n in order)

    for name in order:
        m = metrics[name]
        values = [
            m["best_test_acc"] / max_acc,
            (1 / m["latency_ms"]) / max_speed,
            (1 / m["params_m"]) / max_eff,
            m["fps"] / max_fps,
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.2, color=PALETTE[name], label=SHORT_LABELS[name])
        ax.fill(angles, values, alpha=0.12, color=PALETTE[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color=PALETTE["subtle"])
    ax.set_title("Multi-Metric Comparison (Normalized)", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    save(fig, "fig8_radar_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Architecture Diagram (RepViT-CIFAR10)
# ══════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.50, 0.96, "RepViT-CIFAR10 Architecture", fontsize=22, fontweight="bold",
            color=PALETTE["fg"], ha="center", va="top")
    ax.text(0.50, 0.91, "Optimized for small-resolution 32×32 CIFAR-10 images with stride-1 patch embed and selective SE removal",
            fontsize=11, color=PALETTE["subtle"], ha="center", va="top")

    # Stage boxes
    stages = [
        ("Patch\nEmbed", "32×32→32×32", "3→24→48\nStride 1×1",   "#DCFCE7", "#16A34A"),
        ("Stage 1",       "32×32",       "48 ch\n3 blocks\nAlt SE", "#DBEAFE", "#2563EB"),
        ("Stage 2",       "16×16",       "96 ch\n4 blocks\nAlt SE", "#E0E7FF", "#4F46E5"),
        ("Stage 3",       "8×8",         "192 ch\n16 blocks\nNo SE", "#FEF3C7", "#D97706"),
        ("Stage 4",       "4×4",         "384 ch\n3 blocks\nNo SE",  "#FCE7F3", "#DB2777"),
        ("Head",          "1×1",         "GAP →\nBN + Linear\n→10",  "#F1F5F9", "#475569"),
    ]

    n = len(stages)
    box_w, box_h = 0.12, 0.26
    gap = (0.90 - n * box_w) / (n - 1)
    y0 = 0.50

    for i, (title, res, details, bg, edge) in enumerate(stages):
        x = 0.05 + i * (box_w + gap)
        box = FancyBboxPatch((x, y0), box_w, box_h,
                             boxstyle="round,pad=0.015,rounding_size=0.02",
                             facecolor=bg, edgecolor=edge, linewidth=2)
        ax.add_patch(box)
        ax.text(x + box_w/2, y0 + box_h - 0.03, title, ha="center", va="top",
                fontsize=12, fontweight="bold", color=PALETTE["fg"])
        ax.text(x + box_w/2, y0 + box_h/2 - 0.01, details, ha="center", va="center",
                fontsize=9, color=PALETTE["subtle"], linespacing=1.3)
        ax.text(x + box_w/2, y0 + 0.02, res, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=edge,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=edge, alpha=0.8))
        
        # Arrow
        if i < n - 1:
            x_end = 0.05 + (i + 1) * (box_w + gap)
            ax.annotate("", xy=(x_end - 0.005, y0 + box_h/2),
                        xytext=(x + box_w + 0.005, y0 + box_h/2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#94A3B8"))

    # Key changes callouts
    callout_y = 0.32
    changes = [
        ("Δ₁  Stride Reduction", "Original: stride=2 → 4× reduction\nCIFAR10: stride=1 → 1× reduction\nPreserves spatial detail at 32×32", "#ECFDF5", PALETTE["green"]),
        ("Δ₂  Selective SE Removal", "Stages 1–2: alternate SE (every other block)\nStages 3–4: SE completely removed\nReduces latency, maintains accuracy", "#EFF6FF", PALETTE["repvit_m1_1"]),
        ("Δ₃  Channel Adaptation", "Matched RepViT-m0.9 channel widths\n(48→96→192→384) for smaller data\nResult: 4.52M vs 7.78M params", "#FFF7ED", PALETTE["mobilenetv3_large_100"]),
    ]
    for j, (title, body, bg, edge) in enumerate(changes):
        cx = 0.05 + j * 0.32
        cw, ch = 0.28, 0.17
        box = FancyBboxPatch((cx, callout_y - ch), cw, ch,
                             boxstyle="round,pad=0.015,rounding_size=0.02",
                             facecolor=bg, edgecolor=edge, linewidth=1.5)
        ax.add_patch(box)
        ax.text(cx + cw/2, callout_y - 0.015, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color=PALETTE["fg"])
        ax.text(cx + cw/2, callout_y - ch/2 - 0.01, body, ha="center", va="center",
                fontsize=8.5, color=PALETTE["subtle"], linespacing=1.4)

    ax.text(0.50, 0.36, "Key CIFAR-10 Adaptations ↓", ha="center", fontsize=13,
            fontweight="bold", color=PALETTE["fg"])

    # Result badge at bottom
    ax.text(0.50, 0.06, "Result  ·  91.02% Accuracy  ·  5.98 ms Latency  ·  4.52M Parameters",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["repvit_cifar10"], 
                      edgecolor="none", alpha=0.95))

    save(fig, "fig9_architecture_diagram")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Experiment Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.95, "Experimental Pipeline", fontsize=20, fontweight="bold",
            color=PALETTE["fg"], ha="center")
    ax.text(0.50, 0.90, "End-to-end workflow: data loading → training → evaluation → comparison",
            fontsize=11, color=PALETTE["subtle"], ha="center")

    boxes = [
        (0.02, "CIFAR-10\nDataset",    "50K train\n10K test\n32×32 RGB",  "#E0F2FE", "#0284C7"),
        (0.19, "Data\nAugmentation",    "Random Crop\nHoriz. Flip\nNormalize", "#DCFCE7", "#16A34A"),
        (0.36, "Model\nTraining",       "AdamW optimizer\nCosine LR\n100 epochs", "#FEF3C7", "#D97706"),
        (0.53, "Checkpoint\nSelection", "Save on best\ntest accuracy\n.pth file", "#F3E8FF", "#7C3AED"),
        (0.70, "Latency\nBenchmark",    "batch=32\n200 timed runs\nCUDA sync", "#FFE4E6", "#E11D48"),
        (0.86, "Visual\nReporting",     "Graphs\nCSV metrics\nComparison", "#F1F5F9", "#475569"),
    ]

    box_w, box_h = 0.12, 0.26
    y0 = 0.48
    for x, title, detail, bg, edge in boxes:
        box = FancyBboxPatch((x, y0), box_w, box_h,
                             boxstyle="round,pad=0.015,rounding_size=0.02",
                             facecolor=bg, edgecolor=edge, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x + box_w/2, y0 + box_h - 0.03, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color=PALETTE["fg"])
        ax.text(x + box_w/2, y0 + box_h * 0.35, detail, ha="center", va="center",
                fontsize=8.5, color=PALETTE["subtle"], linespacing=1.3)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + box_w
        x2 = boxes[i + 1][0]
        ax.annotate("", xy=(x2 - 0.005, y0 + box_h/2),
                    xytext=(x1 + 0.005, y0 + box_h/2),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#94A3B8"))

    # Model badges
    badge_y = 0.33
    ax.text(0.33, badge_y, "Models trained:", ha="right", va="center",
            fontsize=11, color=PALETTE["fg"], fontweight="bold")
    for j, name in enumerate(["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]):
        ax.text(0.35 + j * 0.17, badge_y, SHORT_LABELS[name], ha="left", va="center",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE[name], edgecolor="none"))

    # Scripts row
    script_y = 0.18
    ax.text(0.50, script_y + 0.08, "Repository Scripts", ha="center", fontsize=13,
            fontweight="bold", color=PALETTE["fg"])
    scripts = [
        ("train_cifar10.py", "Training + logging"),
        ("measure_latency.py", "Latency benchmark"),
        ("generate_report_visuals.py", "All visual outputs"),
        ("run_experiments.ps1", "Full experiment runner"),
    ]
    for k, (name, desc) in enumerate(scripts):
        sx = 0.10 + k * 0.22
        ax.text(sx, script_y, name, ha="center", fontsize=10, fontweight="bold",
                color=PALETTE["highlight"], family="monospace")
        ax.text(sx, script_y - 0.04, desc, ha="center", fontsize=9, color=PALETTE["subtle"])

    save(fig, "fig10_experiment_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: RepViT Block Detail
# ══════════════════════════════════════════════════════════════════════════════
def fig_block_detail():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.96, "RepViT Block — Internal Structure", fontsize=20,
            fontweight="bold", ha="center", color=PALETTE["fg"])
    ax.text(0.50, 0.91, "Each block consists of a Token Mixer followed by a Channel Mixer, with structural reparameterization for inference efficiency",
            fontsize=10.5, color=PALETTE["subtle"], ha="center")

    # --- Stride=1 Block (left) ---
    lx, ly = 0.05, 0.12
    ax.text(0.25, 0.85, "Stride-1 Block (identity)", ha="center", fontsize=14, fontweight="bold", color=PALETTE["repvit_cifar10"])

    # Token Mixer
    tm_blocks = [
        ("RepVGGDW",  "3×3 DW conv +\n1×1 DW conv +\nskip → BN", "#DCFCE7", "#16A34A"),
        ("SE (opt.)", "Squeeze-Excite\n(when enabled)", "#FEF3C7", "#D97706"),
    ]
    bw, bh = 0.18, 0.16
    for i, (t, d, bg, ec) in enumerate(tm_blocks):
        bx = lx + i * 0.21
        by = ly + 0.48
        box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.01,rounding_size=0.015",
                             facecolor=bg, edgecolor=ec, linewidth=1.5)
        ax.add_patch(box)
        ax.text(bx + bw/2, by + bh - 0.025, t, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.text(bx + bw/2, by + bh/2 - 0.015, d, ha="center", va="center", fontsize=8.5, color=PALETTE["subtle"], linespacing=1.3)

    ax.text(0.25, 0.67, "Token Mixer", ha="center", fontsize=12, fontweight="bold",
            color=PALETTE["fg"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor=PALETTE["border"]))

    # Arrow down
    ax.annotate("", xy=(0.25, 0.45), xytext=(0.25, 0.48),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#94A3B8"))

    # Channel Mixer
    cm_blocks = [
        ("Conv 1×1\n+BN",    "expand", "#DBEAFE", "#2563EB"),
        ("GELU",             "act",    "#E0E7FF", "#4F46E5"),
        ("Conv 1×1\n+BN",    "project","#DBEAFE", "#2563EB"),
    ]
    for i, (t, d, bg, ec) in enumerate(cm_blocks):
        bx = lx + i * 0.15
        by = ly + 0.18
        box = FancyBboxPatch((bx, by), 0.13, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                             facecolor=bg, edgecolor=ec, linewidth=1.5)
        ax.add_patch(box)
        ax.text(bx + 0.065, by + 0.10, t, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(bx + 0.065, by + 0.03, d, ha="center", va="bottom", fontsize=8, color=PALETTE["subtle"])

    ax.text(0.25, 0.35, "Channel Mixer (with residual)", ha="center", fontsize=12, fontweight="bold",
            color=PALETTE["fg"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor=PALETTE["border"]))

    # --- Stride=2 Block (right) ---
    rx = 0.55
    ax.text(0.75, 0.85, "Stride-2 Block (downsample)", ha="center", fontsize=14, fontweight="bold", color=PALETTE["repvit_m1_1"])

    tm2_blocks = [
        ("DW Conv\n3×3, s=2", "Depthwise\nstride=2", "#DBEAFE", "#2563EB"),
        ("SE (opt.)",          "Squeeze-Excite\n(when enabled)", "#FEF3C7", "#D97706"),
        ("Conv 1×1\n+BN",     "Channel\nprojection", "#E0E7FF", "#4F46E5"),
    ]
    for i, (t, d, bg, ec) in enumerate(tm2_blocks):
        bx = rx + i * 0.14
        by = ly + 0.48
        box = FancyBboxPatch((bx, by), 0.12, 0.16, boxstyle="round,pad=0.01,rounding_size=0.015",
                             facecolor=bg, edgecolor=ec, linewidth=1.5)
        ax.add_patch(box)
        ax.text(bx + 0.06, by + 0.16 - 0.025, t, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(bx + 0.06, by + 0.16/2 - 0.015, d, ha="center", va="center", fontsize=8, color=PALETTE["subtle"], linespacing=1.3)

    ax.text(0.75, 0.67, "Token Mixer (downsample)", ha="center", fontsize=12, fontweight="bold",
            color=PALETTE["fg"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor=PALETTE["border"]))
    ax.annotate("", xy=(0.75, 0.45), xytext=(0.75, 0.48),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#94A3B8"))

    cm2_blocks = [
        ("Conv 1×1\n+BN", "expand 2×", "#DBEAFE", "#2563EB"),
        ("GELU",          "act",       "#E0E7FF", "#4F46E5"),
        ("Conv 1×1\n+BN", "project",   "#DBEAFE", "#2563EB"),
    ]
    for i, (t, d, bg, ec) in enumerate(cm2_blocks):
        bx = rx + i * 0.15
        by = ly + 0.18
        box = FancyBboxPatch((bx, by), 0.13, 0.13, boxstyle="round,pad=0.01,rounding_size=0.015",
                             facecolor=bg, edgecolor=ec, linewidth=1.5)
        ax.add_patch(box)
        ax.text(bx + 0.065, by + 0.10, t, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(bx + 0.065, by + 0.03, d, ha="center", va="bottom", fontsize=8, color=PALETTE["subtle"])

    ax.text(0.75, 0.35, "Channel Mixer (with residual)", ha="center", fontsize=12, fontweight="bold",
            color=PALETTE["fg"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor=PALETTE["border"]))

    # Divider
    ax.axvline(x=0.50, ymin=0.12, ymax=0.85, color=PALETTE["border"], linewidth=1.5, linestyle="--")

    # Reparameterization note at bottom
    ax.text(0.50, 0.07, "At inference: Conv2d_BN → fused Conv2d;  RepVGGDW multi-branch → single 3×3 Conv;  BN_Linear → fused Linear",
            ha="center", fontsize=10, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["highlight"], edgecolor="none", alpha=0.9))

    save(fig, "fig11_repvit_block_detail")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Summary Table (as figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_summary_table(metrics):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    order = ["repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"]
    col_labels = ["Model", "Best Acc (%)", "Final Acc (%)", "Latency (ms)", "FPS", "Params (M)", "Epochs"]
    cell_text = []
    cell_colors = []
    for name in order:
        m = metrics[name]
        cell_text.append([
            SHORT_LABELS[name],
            f"{m['best_test_acc']:.2f}",
            f"{m['final_test_acc']:.2f}",
            f"{m['latency_ms']:.2f}",
            f"{m['fps']:.0f}",
            f"{m['params_m']:.2f}",
            str(m["epochs"]),
        ])
        cell_colors.append([PALETTE[name] + "18"] * 7)  # light tint

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                     loc="center", colColours=["#E2E8F0"] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color=PALETTE["fg"])
            cell.set_facecolor("#CBD5E1")
        else:
            cell.set_facecolor(cell_colors[row - 1][col])
        cell.set_edgecolor(PALETTE["border"])

    ax.set_title("Model Comparison Summary", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    save(fig, "fig12_summary_table")


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════
def write_csv(metrics):
    path = VISUALS_DIR / "model_metrics_summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "label", "best_test_acc", "final_test_acc", "latency_ms", "fps", "params_m", "epochs"])
        for name, m in metrics.items():
            w.writerow([name, SHORT_LABELS[name], f"{m['best_test_acc']:.2f}",
                        f"{m['final_test_acc']:.2f}", f"{m['latency_ms']:.2f}",
                        f"{m['fps']:.1f}", f"{m['params_m']:.2f}", m["epochs"]])
    print(f"  ✓ model_metrics_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    configure_style()
    print("Loading experiment data…")
    histories, metrics = load_all()
    print(f"  Loaded {len(histories)} models\n")

    print("Generating figures:")
    fig_test_accuracy(histories, metrics)
    fig_train_vs_test(histories, metrics)
    fig_lr_schedule(histories)
    fig_accuracy_bars(metrics)
    fig_latency_fps(metrics)
    fig_tradeoff(metrics)
    fig_params(metrics)
    fig_radar(metrics)
    fig_architecture()
    fig_pipeline()
    fig_block_detail()
    fig_summary_table(metrics)
    write_csv(metrics)

    print(f"\nAll visuals saved to: {VISUALS_DIR}")
    print("Files are available in both PNG (300 dpi) and SVG formats.")


if __name__ == "__main__":
    main()
