#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SCENARIO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

device=${1:-cuda}
DOWNLOAD_DIR=${2:-"${SCENARIO_ROOT}/_Shared/downloads/KITTI"}
RAW_ROOT=${3:-"${SCENARIO_ROOT}/_Shared/data/ad/kitti_raw"}
LOG_DIR="${SCENARIO_ROOT}/External/project/logs"
LOG_FILE="${LOG_DIR}/kitti_external_eval_$(date +%Y%m%d_%H%M%S).log"

IMAGE_ZIP="${DOWNLOAD_DIR}/data_object_image_2.zip"
LABEL_ZIP="${DOWNLOAD_DIR}/data_object_label_2.zip"

mkdir -p "$LOG_DIR" "$RAW_ROOT"

{
  echo "Started KITTI external evaluation: $(date)"
  echo "Scenario root: $SCENARIO_ROOT"
  echo "Download dir: $DOWNLOAD_DIR"
  echo "Raw root: $RAW_ROOT"
  echo "Device: $device"

  if [[ ! -d "${RAW_ROOT}/training/image_2" ]]; then
    if [[ ! -f "$IMAGE_ZIP" ]]; then
      echo "Missing image zip: $IMAGE_ZIP" >&2
      exit 1
    fi
    echo "Extracting KITTI images..."
    unzip -oq "$IMAGE_ZIP" -d "$RAW_ROOT"
  else
    echo "KITTI images already extracted."
  fi

  if [[ ! -d "${RAW_ROOT}/training/label_2" ]]; then
    if [[ ! -f "$LABEL_ZIP" ]]; then
      echo "Missing label zip: $LABEL_ZIP" >&2
      exit 1
    fi
    echo "Extracting KITTI labels..."
    unzip -oq "$LABEL_ZIP" -d "$RAW_ROOT"
  else
    echo "KITTI labels already extracted."
  fi

  image_count=$(find "${RAW_ROOT}/training/image_2" -maxdepth 1 -type f | wc -l)
  label_count=$(find "${RAW_ROOT}/training/label_2" -maxdepth 1 -type f | wc -l)
  echo "KITTI extracted counts: images=${image_count}, labels=${label_count}"

  echo "Running all KITTI external evals sequentially..."
  "${SCRIPT_DIR}/run_external_kitti_eval.sh" "$device" "$RAW_ROOT"

  echo "Finished KITTI external evaluation: $(date)"
  echo "Summary CSV: ${SCENARIO_ROOT}/External/project/kitti_summary.csv"
  echo "Summary MD: ${SCENARIO_ROOT}/External/project/kitti_summary.md"
} 2>&1 | tee "$LOG_FILE"

echo "Log written to: $LOG_FILE"
