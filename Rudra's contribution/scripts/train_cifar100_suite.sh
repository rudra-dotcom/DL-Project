#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:?usage: scripts/train_cifar100_suite.sh <cifar_root> [output_root]}"
OUTPUT_ROOT="${2:-checkpoints/cifar100_suite}"

DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PROJECT="${PROJECT:-repvit-cifar100}"
REPVIT_FINETUNE_CKPT="${REPVIT_FINETUNE_CKPT:-}"

COMMON_ARGS=(
  --data-set CIFAR
  --data-path "$DATA_PATH"
  --input-size 32
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --output_dir "$OUTPUT_ROOT"
  --device "$DEVICE"
  --distillation-type none
  --project "$PROJECT"
)

launch() {
  local model_name="$1"
  shift

  if (( NUM_GPUS > 1 )); then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
      main.py --model "$model_name" --dist-eval "${COMMON_ARGS[@]}" "$@"
  else
    python main.py --model "$model_name" "${COMMON_ARGS[@]}" "$@"
  fi
}

repvit_extra_args=()
if [[ -n "$REPVIT_FINETUNE_CKPT" ]]; then
  repvit_extra_args=(--finetune "$REPVIT_FINETUNE_CKPT")
fi

launch mobilenetv3_large_100
launch repvit_m0_9 "${repvit_extra_args[@]}"
launch repvit_m0_9_lr "${repvit_extra_args[@]}"
launch repvit_m0_9_lr_rase "${repvit_extra_args[@]}"
