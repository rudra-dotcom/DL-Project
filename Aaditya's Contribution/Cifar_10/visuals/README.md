# Visual Assets

Generated from the repository's training and latency logs.

## Key Findings

- RepViT-CIFAR10 achieved the best test accuracy: 91.02%.
- RepViT-m1.1 reached 87.39% with similar latency (5.95 ms).
- MobileNetV3-Large was the fastest at 2.98 ms, but peaked at 75.77%.

## Files

- `training_curves.png/.svg`: test-accuracy curves over training epochs.
- `accuracy_latency_tradeoff.png/.svg`: latency vs. accuracy comparison.
- `repvit_cifar10_architecture.png/.svg`: architecture summary of the custom CIFAR-10 RepViT.
- `experiment_pipeline.png/.svg`: end-to-end repo workflow diagram.
- `model_metrics_summary.csv`: report-friendly metrics table.

Regenerate with:

```bash
python generate_visuals.py
```