import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_accuracy_curves(curve_df, output_dir):
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=curve_df, x='epoch', y='test_acc1', hue='model')
    plt.title('CIFAR-100 Validation Accuracy')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curves.png', dpi=200)
    plt.close()


def save_loss_curves(curve_df, output_dir):
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=curve_df, x='epoch', y='train_loss', hue='model')
    plt.title('CIFAR-100 Training Loss')
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=200)
    plt.close()


def save_best_accuracy_bar(summary_df, output_dir):
    ordered = summary_df.sort_values(by='best_acc1', ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=ordered, x='model', y='best_acc1')
    plt.title('Best CIFAR-100 Accuracy by Model')
    plt.ylabel('Best Top-1 Accuracy (%)')
    plt.xlabel('Model')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'best_accuracy_bar.png', dpi=200)
    plt.close()


def save_scatter(summary_df, x_col, y_col, output_path, title, xlabel):
    valid = summary_df.dropna(subset=[x_col, y_col])
    if valid.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=valid, x=x_col, y=y_col, hue='model', s=120)
    for _, row in valid.iterrows():
        plt.text(row[x_col], row[y_col], row['model'], fontsize=9, ha='left', va='bottom')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Best Top-1 Accuracy (%)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots from collected experiment CSVs.')
    parser.add_argument('--summary-csv', default='results/generated/cifar100_summary.csv')
    parser.add_argument('--curves-csv', default='results/generated/cifar100_curves.csv')
    parser.add_argument('--output-dir', default='results/generated/plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style='whitegrid')

    summary_df = pd.read_csv(args.summary_csv)
    curve_df = pd.read_csv(args.curves_csv)

    if summary_df.empty or curve_df.empty:
        raise RuntimeError('Summary or curve CSV is empty. Run training and collect_results.py first.')

    save_accuracy_curves(curve_df, output_dir)
    save_loss_curves(curve_df, output_dir)
    save_best_accuracy_bar(summary_df, output_dir)
    save_scatter(
        summary_df,
        'latency_mean_ms',
        'best_acc1',
        output_dir / 'accuracy_vs_latency.png',
        'Accuracy vs GPU Latency',
        'Mean Latency (ms, batch=1)',
    )
    save_scatter(
        summary_df,
        'flops_g',
        'best_acc1',
        output_dir / 'accuracy_vs_flops.png',
        'Accuracy vs FLOPs',
        'FLOPs (G)',
    )
    save_scatter(
        summary_df,
        'params_m',
        'best_acc1',
        output_dir / 'accuracy_vs_params.png',
        'Accuracy vs Parameters',
        'Parameters (Millions)',
    )

    print(f'Wrote plots to {output_dir}')


if __name__ == '__main__':
    main()
