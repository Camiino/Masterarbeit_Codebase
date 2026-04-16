#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SCENARIO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "$SCENARIO_ROOT"

python _Shared/scripts/00_build_label_mapping.py \
  --dataset-root _Shared/data/is/raw/synthetic_yolov9_all_images \
  --out _Shared/data/is/label_mapping_synthetic.json

python _Shared/scripts/01_prepare_real_yolo_splits.py \
  --dataset-root _Shared/data/is/raw/synthetic_yolov9_all_images \
  --out-root _Shared/data/is/synthetic_yolo_splits \
  --name is_synthetic \
  --n-images 488 \
  --seed 0
