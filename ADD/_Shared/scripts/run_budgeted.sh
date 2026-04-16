#!/usr/bin/env bash
set -euo pipefail

# Determine repo root (two levels up from this script: _Shared/scripts -> repo root)
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_experiments.py"

device=${1:-cuda}
max_hours=4
seeds=(1 2 3)

cd "$REPO_ROOT"

echo "Repo root: $REPO_ROOT"
for seed in "${seeds[@]}"; do
  echo "=== Seed ${seed}: YOLO train (max ${max_hours}h) ==="
  timeout "${max_hours}h" python "$RUNNER" --action yolo-train --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: YOLO eval ==="
  python "$RUNNER" --action yolo-eval --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: YOLO eval on SYNTHIA test ==="
  python "$RUNNER" --action yolo-eval-synthia --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: FRCNN train (max ${max_hours}h) ==="
  python "$RUNNER" --action frcnn-train --seed "$seed" --device "$device" --max-hours "$max_hours"

  echo "=== Seed ${seed}: FRCNN eval ==="
  python "$RUNNER" --action frcnn-eval --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: FRCNN eval on SYNTHIA test ==="
  python "$RUNNER" --action frcnn-eval-synthia --seed "$seed" --device "$device"

done

echo "All seeds completed."
