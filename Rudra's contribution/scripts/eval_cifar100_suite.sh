#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:?usage: scripts/eval_cifar100_suite.sh <cifar_root> [run_root] [output_root]}"
RUN_ROOT="${2:-checkpoints/cifar100_suite}"
OUTPUT_ROOT="${3:-checkpoints/cifar100_eval}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"

while IFS= read -r checkpoint_path; do
  run_dir="$(dirname "$checkpoint_path")"
  model_name="$(basename "$(dirname "$run_dir")")"
  echo "Evaluating ${model_name} from ${checkpoint_path}"
  python main.py \
    --eval \
    --model "$model_name" \
    --resume "$checkpoint_path" \
    --data-set CIFAR \
    --data-path "$DATA_PATH" \
    --input-size 32 \
    --batch-size 256 \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --distillation-type none \
    --output_dir "$OUTPUT_ROOT"
done < <(find "$RUN_ROOT" -name checkpoint_best.pth | sort)
