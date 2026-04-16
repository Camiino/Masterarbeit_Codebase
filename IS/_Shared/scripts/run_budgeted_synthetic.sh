#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SCENARIO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

device_arg=${1:-cuda}
max_hours=${2:-4}
seeds=(1 2 3)

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
SYNTH_YAML="${SYNTH_ROOT}/is_synthetic.yaml"
REAL_ROOT="${SCENARIO_ROOT}/_Shared/data/is/real_yolo_splits"

YOLO_RUN_ROOT="${SCENARIO_ROOT}/One-Stage/Synthetic/project/runs_is"
YOLO_METRICS_REAL_ROOT="${SCENARIO_ROOT}/One-Stage/Synthetic/project/metrics_synthetic"
YOLO_METRICS_INDOMAIN_ROOT="${SCENARIO_ROOT}/One-Stage/Synthetic/project/metrics_synthetic_indomain"
FRCNN_RUN_ROOT="${SCENARIO_ROOT}/Two-Stage/Synthetic/project/runs_is_frcnn"
FRCNN_METRICS_REAL_ROOT="${SCENARIO_ROOT}/Two-Stage/Synthetic/project/metrics"
FRCNN_METRICS_INDOMAIN_ROOT="${SCENARIO_ROOT}/Two-Stage/Synthetic/project/metrics_indomain"

YOLO_EPOCHS=${YOLO_EPOCHS:-100}
YOLO_BATCH=${YOLO_BATCH:-16}
YOLO_WORKERS=${YOLO_WORKERS:-8}
FRCNN_EPOCHS=${FRCNN_EPOCHS:-24}
FRCNN_BATCH=${FRCNN_BATCH:-4}
FRCNN_WORKERS=${FRCNN_WORKERS:-0}
YOLO_MODEL="${SCRIPT_DIR}/yolov8m.pt"
if [[ ! -f "$YOLO_MODEL" ]]; then
  YOLO_MODEL="yolov8m.pt"
fi

cd "$SCENARIO_ROOT"

echo "Scenario root: $SCENARIO_ROOT"
echo "Preparing/refreshing synthetic and real splits..."
"_Shared/scripts/run_synthetic_prepare.sh"
"_Shared/scripts/run_real_only_prepare.sh"

mkdir -p "$YOLO_RUN_ROOT" "$YOLO_METRICS_REAL_ROOT" "$YOLO_METRICS_INDOMAIN_ROOT" "$FRCNN_RUN_ROOT" "$FRCNN_METRICS_REAL_ROOT" "$FRCNN_METRICS_INDOMAIN_ROOT"

for seed in "${seeds[@]}"; do
  yolo_name="IS_SYNTH_seed${seed}_yolov8m"
  frcnn_name="IS_SYNTH_seed${seed}_frcnn"

  echo "=== Seed ${seed}: Synthetic YOLO train (max ${max_hours}h) ==="
  timeout "${max_hours}h" python _Shared/scripts/02_train_yolov8m_real.py \
    --data-yaml "$SYNTH_YAML" \
    --model "$YOLO_MODEL" \
    --run-root "$YOLO_RUN_ROOT" \
    --name "$yolo_name" \
    --seed "$seed" \
    --epochs "$YOLO_EPOCHS" \
    --batch "$YOLO_BATCH" \
    --workers "$YOLO_WORKERS" \
    --device "$yolo_device"

  echo "=== Seed ${seed}: Synthetic YOLO eval on synthetic test_internal ==="
  python _Shared/scripts/03_eval_yolo_coco.py \
    --model "${YOLO_RUN_ROOT}/${yolo_name}/weights/best.pt" \
    --dataset-root "$SYNTH_ROOT" \
    --split-name test_internal \
    --out-dir "${YOLO_METRICS_INDOMAIN_ROOT}/IS_SYNTH_seed${seed}_yolov8m_eval" \
    --device "$yolo_device"

  echo "=== Seed ${seed}: Synthetic YOLO eval on real test_internal ==="
  python _Shared/scripts/03_eval_yolo_coco.py \
    --model "${YOLO_RUN_ROOT}/${yolo_name}/weights/best.pt" \
    --dataset-root "$REAL_ROOT" \
    --split-name test_internal \
    --out-dir "${YOLO_METRICS_REAL_ROOT}/IS_SYNTH_seed${seed}_yolov8m_on_real_eval" \
    --device "$yolo_device"

  echo "=== Seed ${seed}: Synthetic FRCNN train (max ${max_hours}h) ==="
  python _Shared/scripts/04_train_frcnn_real.py \
    --yolo-split-root "$SYNTH_ROOT" \
    --split-dir "${SYNTH_ROOT}/splits" \
    --run-root "$FRCNN_RUN_ROOT" \
    --run-name "$frcnn_name" \
    --seed "$seed" \
    --epochs "$FRCNN_EPOCHS" \
    --batch "$FRCNN_BATCH" \
    --workers "$FRCNN_WORKERS" \
    --device "$frcnn_device" \
    --max-hours "$max_hours"

  echo "=== Seed ${seed}: Synthetic FRCNN eval on synthetic test_internal ==="
  python _Shared/scripts/05_eval_frcnn_yolo.py \
    --weights "${FRCNN_RUN_ROOT}/${frcnn_name}/final.pt" \
    --split test_internal_yolo \
    --yolo-split-root "$SYNTH_ROOT" \
    --out "${FRCNN_METRICS_INDOMAIN_ROOT}/IS_SYNTH_seed${seed}_frcnn_eval.json" \
    --device "$frcnn_device"

  echo "=== Seed ${seed}: Synthetic FRCNN eval on real test_internal ==="
  python _Shared/scripts/05_eval_frcnn_yolo.py \
    --weights "${FRCNN_RUN_ROOT}/${frcnn_name}/final.pt" \
    --split test_internal_yolo \
    --yolo-split-root "$REAL_ROOT" \
    --out "${FRCNN_METRICS_REAL_ROOT}/IS_SYNTH_seed${seed}_frcnn_on_real_eval.json" \
    --device "$frcnn_device"
done

echo "All synthetic seeds completed."
