#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SCENARIO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

device_arg=${1:-cuda}
if [[ "$device_arg" == "cuda" ]]; then
  yolo_device="0"
  frcnn_device="cuda"
elif [[ "$device_arg" =~ ^[0-9]+$ ]]; then
  yolo_device="$device_arg"
  frcnn_device="cuda:${device_arg}"
else
  yolo_device="$device_arg"
  frcnn_device="$device_arg"
fi

SYNTH_ROOT="${SCENARIO_ROOT}/_Shared/data/is/synthetic_yolo_splits"
REAL_ROOT="${SCENARIO_ROOT}/_Shared/data/is/real_yolo_splits"

YOLO_RUN_ROOT="${SCENARIO_ROOT}/One-Stage/Synthetic/project/runs_is"
YOLO_METRICS_REAL_ROOT="${SCENARIO_ROOT}/One-Stage/Synthetic/project/metrics_synthetic"
YOLO_METRICS_INDOMAIN_ROOT="${SCENARIO_ROOT}/One-Stage/Synthetic/project/metrics_synthetic_indomain"
FRCNN_RUN_ROOT="${SCENARIO_ROOT}/Two-Stage/Synthetic/project/runs_is_frcnn"
FRCNN_METRICS_REAL_ROOT="${SCENARIO_ROOT}/Two-Stage/Synthetic/project/metrics"
FRCNN_METRICS_INDOMAIN_ROOT="${SCENARIO_ROOT}/Two-Stage/Synthetic/project/metrics_indomain"

cd "$SCENARIO_ROOT"

"_Shared/scripts/run_synthetic_prepare.sh"
"_Shared/scripts/run_real_only_prepare.sh"

mkdir -p "$YOLO_METRICS_REAL_ROOT" "$YOLO_METRICS_INDOMAIN_ROOT" "$FRCNN_METRICS_REAL_ROOT" "$FRCNN_METRICS_INDOMAIN_ROOT"

for seed in 1 2 3; do
  yolo_name="IS_SYNTH_seed${seed}_yolov8m"
  frcnn_name="IS_SYNTH_seed${seed}_frcnn"
  yolo_weights="${YOLO_RUN_ROOT}/${yolo_name}/weights/best.pt"
  frcnn_weights="${FRCNN_RUN_ROOT}/${frcnn_name}/final.pt"

  if [[ -f "$yolo_weights" ]]; then
    echo "=== Seed ${seed}: Synthetic YOLO eval on synthetic test_internal ==="
    python _Shared/scripts/03_eval_yolo_coco.py \
      --model "$yolo_weights" \
      --dataset-root "$SYNTH_ROOT" \
      --split-name test_internal \
      --out-dir "${YOLO_METRICS_INDOMAIN_ROOT}/IS_SYNTH_seed${seed}_yolov8m_eval" \
      --device "$yolo_device"

    echo "=== Seed ${seed}: Synthetic YOLO eval on real test_internal ==="
    python _Shared/scripts/03_eval_yolo_coco.py \
      --model "$yolo_weights" \
      --dataset-root "$REAL_ROOT" \
      --split-name test_internal \
      --out-dir "${YOLO_METRICS_REAL_ROOT}/IS_SYNTH_seed${seed}_yolov8m_on_real_eval" \
      --device "$yolo_device"
  else
    echo "Skipping seed ${seed} YOLO evals; missing $yolo_weights"
  fi

  if [[ -f "$frcnn_weights" ]]; then
    echo "=== Seed ${seed}: Synthetic FRCNN eval on synthetic test_internal ==="
    python _Shared/scripts/05_eval_frcnn_yolo.py \
      --weights "$frcnn_weights" \
      --split test_internal_yolo \
      --yolo-split-root "$SYNTH_ROOT" \
      --out "${FRCNN_METRICS_INDOMAIN_ROOT}/IS_SYNTH_seed${seed}_frcnn_eval.json" \
      --device "$frcnn_device"

    echo "=== Seed ${seed}: Synthetic FRCNN eval on real test_internal ==="
    python _Shared/scripts/05_eval_frcnn_yolo.py \
      --weights "$frcnn_weights" \
      --split test_internal_yolo \
      --yolo-split-root "$REAL_ROOT" \
      --out "${FRCNN_METRICS_REAL_ROOT}/IS_SYNTH_seed${seed}_frcnn_on_real_eval.json" \
      --device "$frcnn_device"
  else
    echo "Skipping seed ${seed} FRCNN evals; missing $frcnn_weights"
  fi
done
