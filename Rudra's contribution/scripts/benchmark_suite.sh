#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: scripts/benchmark_suite.sh <run_root> [output_root]}"
OUTPUT_ROOT="${2:-results/generated/benchmarks}"

DEVICE="${DEVICE:-cuda}"
INPUT_SIZE="${INPUT_SIZE:-32}"
NUM_CLASSES="${NUM_CLASSES:-100}"
LATENCY_STEPS="${LATENCY_STEPS:-200}"
THROUGHPUT_BATCH_SIZE="${THROUGHPUT_BATCH_SIZE:-256}"

mkdir -p "$OUTPUT_ROOT"

while IFS= read -r checkpoint_path; do
  run_dir="$(dirname "$checkpoint_path")"
  model_name="$(basename "$(dirname "$run_dir")")"
  fuse_args=()
  if [[ "$model_name" == repvit_* ]]; then
    fuse_args=(--fuse-bn)
  fi

  python scripts/benchmark_latency.py \
    --model "$model_name" \
    --checkpoint "$checkpoint_path" \
    --device "$DEVICE" \
    --input-size "$INPUT_SIZE" \
    --num-classes "$NUM_CLASSES" \
    --latency-steps "$LATENCY_STEPS" \
    --throughput-batch-size "$THROUGHPUT_BATCH_SIZE" \
    --output "$OUTPUT_ROOT/${model_name}.json" \
    "${fuse_args[@]}"
done < <(find "$RUN_ROOT" -name checkpoint_best.pth | sort)
