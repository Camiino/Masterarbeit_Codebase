#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

device=${1:-cuda}
real_pct=${2:-70}
synth_pct=${3:-30}
mix_seed=${4:-0}
max_hours=${5:-4}

MIX_SLUG="mixed_bdd${real_pct}_synth${synth_pct}_seed${mix_seed}"
MIX_ROOT="${REPO_ROOT}/_Shared/data/ad/${MIX_SLUG}"
MIX_YAML="${MIX_ROOT}/${MIX_SLUG}.yaml"

YOLO_RUN_ROOT="${REPO_ROOT}/One-Stage/Mixed/project/runs_mix"
YOLO_METRICS_ROOT="${REPO_ROOT}/One-Stage/Mixed/project/metrics"
FRCNN_RUN_ROOT="${REPO_ROOT}/Two-Stage/Mixed/project/runs_mix_frcnn"
FRCNN_METRICS_ROOT="${REPO_ROOT}/Two-Stage/Mixed/project/metrics"

BDD_ROOT="${REPO_ROOT}/_Shared/data/ad/bdd_yolo_splits"
SYNTH_ROOT="${REPO_ROOT}/_Shared/data/ad/synthia_yolo_splits"

SEEDS=(1 2 3)

cd "$REPO_ROOT"

echo "=== Preparing mixed dataset ${MIX_SLUG} ==="
python "${SCRIPT_DIR}/11_prepare_mixed_yolo_dataset.py" \
  --real-pct "$real_pct" \
  --synth-pct "$synth_pct" \
  --seed "$mix_seed"

mkdir -p "$YOLO_RUN_ROOT" "$YOLO_METRICS_ROOT" "$FRCNN_RUN_ROOT" "$FRCNN_METRICS_ROOT"

for seed in "${SEEDS[@]}"; do
  yolo_name="MIX_${real_pct}_${synth_pct}_yolov8m_seed${seed}"
  frcnn_name="MIX_${real_pct}_${synth_pct}_frcnn_seed${seed}"

  echo "=== Seed ${seed}: YOLO train on ${MIX_SLUG} (max ${max_hours}h) ==="
  timeout "${max_hours}h" python "${SCRIPT_DIR}/04_train_yolov8m_e1.py" \
    --data-yaml "$MIX_YAML" \
    --run-root "$YOLO_RUN_ROOT" \
    --name "$yolo_name" \
    --seed "$seed" \
    --device "$device"

  echo "=== Seed ${seed}: YOLO eval on BDD and SYNTHIA test ==="
  python "${SCRIPT_DIR}/05_eval_internal_test_coco_custom_scale.py" \
    --model "${YOLO_RUN_ROOT}/${yolo_name}/weights/best.pt" \
    --dataset-root "$BDD_ROOT" \
    --split-name test_internal \
    --extra-dataset-root "$SYNTH_ROOT" \
    --extra-name synthia \
    --extra-split-name test \
    --out-dir "${YOLO_METRICS_ROOT}/MIX_${real_pct}_${synth_pct}_seed${seed}_yolo_dual_eval" \
    --device "$device"

  echo "=== Seed ${seed}: FRCNN train on ${MIX_SLUG} (max ${max_hours}h) ==="
  python "${SCRIPT_DIR}/06_train_frcnn_e6.py" \
    --yolo-split-root "$MIX_ROOT" \
    --split-dir "${MIX_ROOT}/splits" \
    --run-root "$FRCNN_RUN_ROOT" \
    --run-name "$frcnn_name" \
    --seed "$seed" \
    --device "$device" \
    --max-hours "$max_hours"

  echo "=== Seed ${seed}: FRCNN eval on BDD test ==="
  python "${SCRIPT_DIR}/07_eval_frcnn_coco.py" \
    --weights "${FRCNN_RUN_ROOT}/${frcnn_name}/final.pt" \
    --data-root "${REPO_ROOT}/_Shared/data/ad/bdd_coco" \
    --split test_internal_yolo \
    --yolo-split-root "$BDD_ROOT" \
    --out "${FRCNN_METRICS_ROOT}/MIX_${real_pct}_${synth_pct}_seed${seed}_frcnn_on_bdd_eval.json" \
    --device "$device"

  echo "=== Seed ${seed}: FRCNN eval on SYNTHIA test ==="
  python "${SCRIPT_DIR}/07_eval_frcnn_coco.py" \
    --weights "${FRCNN_RUN_ROOT}/${frcnn_name}/final.pt" \
    --data-root "${REPO_ROOT}/_Shared/data/ad/bdd_coco" \
    --split test_internal_yolo \
    --yolo-split-root "$SYNTH_ROOT" \
    --out "${FRCNN_METRICS_ROOT}/MIX_${real_pct}_${synth_pct}_seed${seed}_frcnn_on_synthia_eval.json" \
    --device "$device"
done

echo "All mixed runs completed for ${MIX_SLUG}."
