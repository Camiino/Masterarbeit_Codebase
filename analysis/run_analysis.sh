#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$ROOT_DIR"

python analysis/collect_metrics.py
python analysis/aggregate_results.py
python analysis/summarize_effects.py
python analysis/plot_results.py
python analysis/export_latex_tables.py
python analysis/export_qualitative_examples.py
python analysis/write_results_inventory.py
python analysis/write_thesis_summary.py

echo "Analysis outputs written to analysis/outputs"
