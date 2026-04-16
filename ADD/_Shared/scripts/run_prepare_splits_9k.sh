#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SCENARIO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

seed=${1:-0}
cap=${2:-9000}

cd "$SCENARIO_ROOT"

echo "=== Preparing BDD real split with cap=${cap} images ==="
python _Shared/scripts/run_experiments.py \
  --action make-splits \
  --seed "$seed" \
  --n-images "$cap"

python _Shared/scripts/run_experiments.py \
  --action materialize \
  --seed "$seed"

echo "=== Ensuring SYNTHIA split is capped at ${cap} images ==="
python _Shared/scripts/run_experiments.py \
  --action synthia-prepare \
  --seed "$seed" \
  --subset-size "$cap"

python _Shared/scripts/run_experiments.py \
  --action synthia-yaml \
  --seed "$seed"

echo "=== Rebuilding mixed datasets from capped real/synthetic splits ==="
for mix in "70 30" "50 50" "30 70"; do
  set -- $mix
  real_pct="$1"
  synth_pct="$2"
  python _Shared/scripts/11_prepare_mixed_yolo_dataset.py \
    --real-pct "$real_pct" \
    --synth-pct "$synth_pct" \
    --seed "$seed"
done

echo "ADD capped split preparation completed."
