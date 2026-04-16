#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SCENARIO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

device_arg=${1:-cuda}
kitti_root=${2:-"${SCENARIO_ROOT}/_Shared/data/ad/kitti_raw"}
split_name=${KITTI_SPLIT_NAME:-test_external}

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

KITTI_ROOT="${SCENARIO_ROOT}/_Shared/data/ad/kitti_yolo_splits"
YOLO_OUT_ROOT="${SCENARIO_ROOT}/One-Stage/External/project/metrics_kitti"
FRCNN_OUT_ROOT="${SCENARIO_ROOT}/Two-Stage/External/project/metrics_kitti"
BEST_MANIFEST="${SCENARIO_ROOT}/External/project/best_seed_checkpoints.json"

cd "$SCENARIO_ROOT"

echo "=== Preparing KITTI external split ==="
python _Shared/scripts/12_prepare_kitti_external.py \
  --kitti-root "$kitti_root" \
  --out-root "$KITTI_ROOT" \
  --split-name "$split_name" \
  ${KITTI_PREPARE_ARGS:-}

mkdir -p "$YOLO_OUT_ROOT" "$FRCNN_OUT_ROOT"

echo "=== Selecting best seed checkpoints from existing BDD metrics ==="
python _Shared/scripts/14_select_best_seed_checkpoints.py --out "$BEST_MANIFEST"

echo "=== Evaluating selected YOLO weights on KITTI ==="
while IFS=$'\t' read -r setup weights; do
  [[ -n "$weights" ]] || continue
  rel="${weights#${SCENARIO_ROOT}/}"
  run_name="$(basename "$(dirname "$(dirname "$weights")")")"
  case "$rel" in
    One-Stage/Real/*) group="real" ;;
    One-Stage/Synthetic/*) group="synthetic" ;;
    One-Stage/Mixed/*) group="mixed" ;;
    *) group="other" ;;
  esac
  out_dir="${YOLO_OUT_ROOT}/${group}/${run_name}_kitti_eval"
  echo "--- YOLO ${group}/${run_name} ---"
  python _Shared/scripts/05_eval_internal_test_coco_custom_scale.py \
    --model "$weights" \
    --dataset-root "$KITTI_ROOT" \
    --split-name "$split_name" \
    --out-dir "$out_dir" \
    --device "$yolo_device"
done < <(python - "$BEST_MANIFEST" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for row in data["checkpoints"]:
    if row["model"] == "YOLO":
        print(f'{row["setup"]}\t{row["weights"]}')
PY
)

echo "=== Evaluating selected FRCNN weights on KITTI ==="
while IFS=$'\t' read -r setup weights; do
  [[ -n "$weights" ]] || continue
  rel="${weights#${SCENARIO_ROOT}/}"
  run_name="$(basename "$(dirname "$weights")")"
  case "$rel" in
    Two-Stage/Real/*) group="real" ;;
    Two-Stage/Synthetic/*) group="synthetic" ;;
    Two-Stage/Mixed/*) group="mixed" ;;
    *) group="other" ;;
  esac
  out="${FRCNN_OUT_ROOT}/${group}/${run_name}_kitti_eval.json"
  echo "--- FRCNN ${group}/${run_name} ---"
  python _Shared/scripts/07_eval_frcnn_coco.py \
    --weights "$weights" \
    --data-root "${SCENARIO_ROOT}/_Shared/data/ad/bdd_coco" \
    --split test_internal_yolo \
    --yolo-split-root "$KITTI_ROOT" \
    --yolo-split-name "$split_name" \
    --out "$out" \
    --device "$frcnn_device"
done < <(python - "$BEST_MANIFEST" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for row in data["checkpoints"]:
    if row["model"] == "FRCNN":
        print(f'{row["setup"]}\t{row["weights"]}')
PY
)

echo "=== Summarizing KITTI metrics ==="
python _Shared/scripts/13_summarize_kitti_external.py

echo "KITTI external evaluation completed."
