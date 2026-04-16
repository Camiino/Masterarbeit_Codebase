#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_experiments.py"

device=${1:-cuda}
max_hours=4
seeds=(1 2 3)

cd "$REPO_ROOT"

echo "Repo root: $REPO_ROOT"
for seed in "${seeds[@]}"; do
  echo "=== Seed ${seed}: SYNTHIA YOLO train (max ${max_hours}h) ==="
  timeout "${max_hours}h" python "$RUNNER" --action synthia-yolo-train --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: SYNTHIA YOLO eval on SYNTHIA test ==="
  python "$RUNNER" --action synthia-yolo-eval-indomain --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: SYNTHIA YOLO eval on BDD internal test ==="
  python "$RUNNER" --action synthia-yolo-eval --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: SYNTHIA FRCNN train (max ${max_hours}h) ==="
  python "$RUNNER" --action synthia-frcnn-train --seed "$seed" --device "$device" --max-hours "$max_hours"

  echo "=== Seed ${seed}: SYNTHIA FRCNN eval on SYNTHIA test ==="
  python "$RUNNER" --action synthia-frcnn-eval-indomain --seed "$seed" --device "$device"

  echo "=== Seed ${seed}: SYNTHIA FRCNN eval on BDD internal test ==="
  python "$RUNNER" --action synthia-frcnn-eval --seed "$seed" --device "$device"
done

echo "All SYNTHIA seeds completed."
