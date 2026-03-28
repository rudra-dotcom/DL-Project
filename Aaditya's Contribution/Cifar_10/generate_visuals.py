import csv
import os
import re
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
VISUALS_DIR = ROOT / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)

COLORS = {
    "repvit_cifar10": "#0F766E",
    "repvit_m1_1": "#2563EB",
    "mobilenetv3_large_100": "#D97706",
    "accent": "#111827",
    "grid": "#CBD5E1",
    "note": "#475569",
}

LABELS = {
    "repvit_cifar10": "RepViT-CIFAR10",
    "repvit_m1_1": "RepViT-m1.1",
    "mobilenetv3_large_100": "MobileNetV3-Large",
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

MODEL_SUMMARY = {
    "repvit_cifar10": {
        "params_m": 4.52,
        "stages": [
            ("Patch Embed", "32x32", "3->24->48", "3x3 conv, stride 1 twice"),
            ("Stage 1", "32x32", "48", "3 blocks, alternating SE"),
            ("Stage 2", "16x16", "96", "4 blocks, alternating SE"),
            ("Stage 3", "8x8", "192", "16 blocks, SE removed"),
            ("Stage 4", "4x4", "384", "3 blocks, SE removed"),
            ("Head", "1x1", "384->10", "global pool + classifier"),
        ],
    },
    "repvit_m1_1": {
        "params_m": 7.78,
        "stages": [
            ("Patch Embed", "8x8", "3->32->64", "3x3 conv, stride 2 twice"),
            ("Stage 1", "8x8", "64", "3 blocks"),
            ("Stage 2", "4x4", "128", "4 blocks"),
            ("Stage 3", "2x2", "256", "14 blocks"),
            ("Stage 4", "1x1", "512", "3 blocks"),
            ("Head", "1x1", "512->10", "global pool + classifier"),
        ],
    },
    "mobilenetv3_large_100": {
        "params_m": 4.21,
    },
}

EPOCH_PATTERN = re.compile(
    r"Epoch:\s*(\d+)\s*\|\s*LR:\s*([0-9.]+)\s*\|\s*Train Acc:\s*([0-9.]+)%\s*\|\s*"
    r"Test Acc:\s*([0-9.]+)%\s*\(Best:\s*([0-9.]+)%\)"
)

LATENCY_PATTERN = re.compile(r"Average Latency:\s*([0-9.]+)\s*ms")
FPS_PATTERN = re.compile(r"FPS \(batch=32\):\s*([0-9.]+)")


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#111827",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "font.size": 11,
            "savefig.bbox": "tight",
        }
    )


def parse_training_log(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        match = EPOCH_PATTERN.search(line)
        if match:
            epoch, lr, train_acc, test_acc, best_acc = match.groups()
            rows.append(
                {
                    "epoch": int(epoch),
                    "lr": float(lr),
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                    "best_acc": float(best_acc),
                }
            )
    if not rows:
        raise ValueError(f"No epoch rows parsed from {path}")
    return rows


def parse_latency_log(path: Path):
    text = path.read_text()
    latency_match = LATENCY_PATTERN.search(text)
    fps_match = FPS_PATTERN.search(text)
    if not latency_match or not fps_match:
        raise ValueError(f"Could not parse latency metrics from {path}")
    return {
        "latency_ms": float(latency_match.group(1)),
        "fps": float(fps_match.group(1)),
    }


def load_metrics():
    metrics = {}
    histories = {}
    for model_name, log_path in TRAINING_LOGS.items():
        history = parse_training_log(log_path)
        latency = parse_latency_log(LATENCY_LOGS[model_name])
        histories[model_name] = history
        metrics[model_name] = {
            "best_test_acc": max(row["test_acc"] for row in history),
            "final_test_acc": history[-1]["test_acc"],
            "epochs": len(history),
            "latency_ms": latency["latency_ms"],
            "fps": latency["fps"],
            "params_m": MODEL_SUMMARY.get(model_name, {}).get("params_m"),
        }
    return histories, metrics


def save_figure(fig, stem: str):
    for ext in ("png", "svg"):
        fig.savefig(VISUALS_DIR / f"{stem}.{ext}", dpi=240 if ext == "png" else None)
    plt.close(fig)


def add_model_badge(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        color="white",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="none"),
    )


def plot_training_curves(histories, metrics):
    fig, ax = plt.subplots(figsize=(12, 7))
    for model_name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        test_acc = [row["test_acc"] for row in history]
        ax.plot(
            epochs,
            test_acc,
            linewidth=2.6,
            color=COLORS[model_name],
            label=f"{LABELS[model_name]} ({metrics[model_name]['best_test_acc']:.2f}%)",
        )
        best_idx = max(range(len(test_acc)), key=test_acc.__getitem__)
        ax.scatter(
            epochs[best_idx],
            test_acc[best_idx],
            s=75,
            color=COLORS[model_name],
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
        )

    ax.set_title("CIFAR-10 Test Accuracy Across Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(1, max(len(v) for v in histories.values()))
    ax.set_ylim(20, 95)
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        1,
        22.5,
        "Custom RepViT reaches the highest accuracy while tracking closely with the baseline in latency.",
        fontsize=10.5,
        color=COLORS["note"],
    )
    save_figure(fig, "training_curves")


def plot_tradeoff(metrics):
    fig, ax = plt.subplots(figsize=(11, 7))
    for model_name in ("repvit_cifar10", "repvit_m1_1", "mobilenetv3_large_100"):
        item = metrics[model_name]
        bubble_size = 900 * (item["params_m"] / 4.0)
        ax.scatter(
            item["latency_ms"],
            item["best_test_acc"],
            s=bubble_size,
            color=COLORS[model_name],
            alpha=0.88,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.text(
            item["latency_ms"] + 0.05,
            item["best_test_acc"] + 0.25,
            f"{LABELS[model_name]}\n{item['params_m']:.2f}M params",
            fontsize=10,
            color=COLORS["accent"],
            ha="left",
            va="bottom",
        )

    ax.set_title("Accuracy-Latency Trade-off")
    ax.set_xlabel("Average Inference Latency (ms, batch=32)")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_xlim(2.4, 6.4)
    ax.set_ylim(72, 93)
    ax.grid(True)
    ax.text(
        2.45,
        72.7,
        "Bubble size encodes parameter count. Lower-left is faster; upper-right is more accurate.",
        fontsize=10,
        color=COLORS["note"],
    )
    save_figure(fig, "accuracy_latency_tradeoff")


def draw_box(ax, xy, width, height, facecolor, title, subtitle, edgecolor="#CBD5E1"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + width / 2, y + height * 0.34, subtitle, ha="center", va="center", fontsize=9.2, color=COLORS["note"])


def plot_architecture_overview():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.95, "RepViT-CIFAR10 Architecture", fontsize=20, fontweight="bold", color=COLORS["accent"])
    ax.text(
        0.02,
        0.90,
        "The custom model preserves early spatial detail and removes SE blocks in deeper stages to balance accuracy and latency on 32x32 inputs.",
        fontsize=11,
        color=COLORS["note"],
    )

    stage_colors = ["#DCFCE7", "#DBEAFE", "#FEF3C7", "#FCE7F3", "#EDE9FE", "#E2E8F0"]
    x_positions = [0.04, 0.20, 0.35, 0.51, 0.67, 0.83]
    width = 0.12
    height = 0.18

    for idx, (title, res, channels, note) in enumerate(MODEL_SUMMARY["repvit_cifar10"]["stages"]):
        draw_box(ax, (x_positions[idx], 0.53), width, height, stage_colors[idx], title, f"{res}\n{channels}\n{note}")
        if idx < len(x_positions) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.01, 0.62),
                xytext=(x_positions[idx] + width + 0.01, 0.62),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#64748B"),
            )

    draw_box(ax, (0.07, 0.18), 0.23, 0.18, "#ECFDF5", "Change 1: Gentle Patch Embed", "Baseline m1.1: stride 2 + stride 2\nCIFAR variant: stride 1 + stride 1")
    draw_box(ax, (0.39, 0.18), 0.23, 0.18, "#EFF6FF", "Change 2: Selective SE Usage", "Stage 1-2: alternating SE\nStage 3-4: SE fully removed")
    draw_box(ax, (0.71, 0.18), 0.23, 0.18, "#FFF7ED", "Outcome", "Best accuracy: 91.02%\nLatency: 5.98 ms\nParams: 4.52M")

    ax.text(0.07, 0.40, "CIFAR-specific design decisions", fontsize=13, fontweight="bold", color=COLORS["accent"])
    save_figure(fig, "repvit_cifar10_architecture")


def plot_experiment_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.92, "Experiment Pipeline Used In This Repo", fontsize=20, fontweight="bold", color=COLORS["accent"])
    ax.text(
        0.02,
        0.86,
        "From CIFAR-10 data loading to checkpoint selection and latency benchmarking, the flow below mirrors the repository scripts.",
        fontsize=11,
        color=COLORS["note"],
    )

    boxes = [
        (0.03, 0.42, 0.15, 0.22, "#E0F2FE", "CIFAR-10 Dataset", "32x32 inputs\n50k train / 10k test"),
        (0.22, 0.42, 0.16, 0.22, "#DCFCE7", "Augmentation", "Random crop + flip\nNormalization"),
        (0.42, 0.42, 0.18, 0.22, "#FEF3C7", "Training", "AdamW + cosine LR\nPer-model logging"),
        (0.64, 0.42, 0.15, 0.22, "#F3E8FF", "Best Checkpoints", "Saved on best test acc"),
        (0.83, 0.42, 0.14, 0.22, "#FFE4E6", "Latency Eval", "Dummy 32x32 batch\n200 timed runs"),
    ]

    for x, y, w, h, color, title, subtitle in boxes:
        draw_box(ax, (x, y), w, h, color, title, subtitle)

    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][0] + boxes[idx][2]
        x2 = boxes[idx + 1][0]
        ax.annotate(
            "",
            xy=(x2 - 0.01, 0.53),
            xytext=(x1 + 0.01, 0.53),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#64748B"),
        )

    add_model_badge(ax, 0.48, 0.26, "RepViT-CIFAR10", COLORS["repvit_cifar10"])
    add_model_badge(ax, 0.61, 0.26, "RepViT-m1.1", COLORS["repvit_m1_1"])
    add_model_badge(ax, 0.77, 0.26, "MobileNetV3-Large", COLORS["mobilenetv3_large_100"])
    ax.text(0.37, 0.26, "Compared models", fontsize=11, color=COLORS["accent"], ha="right", va="center")
    save_figure(fig, "experiment_pipeline")


def write_metrics_csv(metrics):
    path = VISUALS_DIR / "model_metrics_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "label", "best_test_acc", "final_test_acc", "latency_ms", "fps", "params_m", "epochs"])
        for model_name, values in metrics.items():
            writer.writerow(
                [
                    model_name,
                    LABELS[model_name],
                    f"{values['best_test_acc']:.2f}",
                    f"{values['final_test_acc']:.2f}",
                    f"{values['latency_ms']:.2f}",
                    f"{values['fps']:.1f}",
                    f"{values['params_m']:.2f}" if values["params_m"] is not None else "",
                    values["epochs"],
                ]
            )


def write_visuals_readme(metrics):
    lines = [
        "# Visual Assets",
        "",
        "Generated from the repository's training and latency logs.",
        "",
        "## Key Findings",
        "",
        f"- RepViT-CIFAR10 achieved the best test accuracy: {metrics['repvit_cifar10']['best_test_acc']:.2f}%.",
        f"- RepViT-m1.1 reached {metrics['repvit_m1_1']['best_test_acc']:.2f}% with similar latency ({metrics['repvit_m1_1']['latency_ms']:.2f} ms).",
        f"- MobileNetV3-Large was the fastest at {metrics['mobilenetv3_large_100']['latency_ms']:.2f} ms, but peaked at {metrics['mobilenetv3_large_100']['best_test_acc']:.2f}%.",
        "",
        "## Files",
        "",
        "- `training_curves.png/.svg`: test-accuracy curves over training epochs.",
        "- `accuracy_latency_tradeoff.png/.svg`: latency vs. accuracy comparison.",
        "- `repvit_cifar10_architecture.png/.svg`: architecture summary of the custom CIFAR-10 RepViT.",
        "- `experiment_pipeline.png/.svg`: end-to-end repo workflow diagram.",
        "- `model_metrics_summary.csv`: report-friendly metrics table.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python generate_visuals.py",
        "```",
    ]
    (VISUALS_DIR / "README.md").write_text("\n".join(lines))


def main():
    configure_style()
    histories, metrics = load_metrics()
    plot_training_curves(histories, metrics)
    plot_tradeoff(metrics)
    plot_architecture_overview()
    plot_experiment_pipeline()
    write_metrics_csv(metrics)
    write_visuals_readme(metrics)
    print(f"Saved visuals to {VISUALS_DIR}")


if __name__ == "__main__":
    main()
