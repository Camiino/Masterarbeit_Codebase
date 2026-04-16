#!/usr/bin/env python3
"""Select the best ADD checkpoint per setup from existing real-target metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

SCENARIO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write best-seed checkpoint manifests for KITTI external eval.")
    p.add_argument("--out", type=Path, default=SCENARIO_ROOT / "External" / "project" / "best_seed_checkpoints.json")
    return p.parse_args()


def metric_ap(path: Path) -> Tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["all"]["AP_50_95"]), float(data["all"]["AP_50"])


def add_candidate(rows: List[Dict], model: str, setup: str, seed: int, metric_path: Path, weights: Path) -> None:
    if not metric_path.is_file() or not weights.is_file():
        return
    ap, ap50 = metric_ap(metric_path)
    rows.append(
        {
            "model": model,
            "setup": setup,
            "seed": seed,
            "AP_50_95": ap,
            "AP_50": ap50,
            "metric": str(metric_path),
            "weights": str(weights),
        }
    )


def collect_candidates() -> List[Dict]:
    root = SCENARIO_ROOT
    rows: List[Dict] = []

    for path in (root / "One-Stage" / "Real" / "project" / "metrics").glob("E1_seed*_yolov8m_eval/internal/metrics.json"):
        seed = int(re.search(r"seed(\d+)", str(path)).group(1))
        add_candidate(
            rows,
            "YOLO",
            "Real",
            seed,
            path,
            root / "One-Stage" / "Real" / "project" / "runs_ad" / f"E1_yolov8m_real_seed{seed}" / "weights" / "best.pt",
        )

    for path in (root / "One-Stage" / "Synthetic" / "project" / "metrics_synthia").glob(
        "SYNTHIA_seed*_yolov8m_on_bdd_eval/internal/metrics.json"
    ):
        seed = int(re.search(r"seed(\d+)", str(path)).group(1))
        add_candidate(
            rows,
            "YOLO",
            "Synthetic",
            seed,
            path,
            root / "One-Stage" / "Synthetic" / "project" / "runs_synthia" / f"SYNTHIA_yolov8m_seed{seed}" / "weights" / "best.pt",
        )

    for path in (root / "One-Stage" / "Mixed" / "project" / "metrics").glob("MIX_*_seed*_yolo_dual_eval/internal/metrics.json"):
        match = re.search(r"MIX_(\d+_\d+)_seed(\d+)", str(path))
        if not match:
            continue
        mix, seed_s = match.groups()
        seed = int(seed_s)
        add_candidate(
            rows,
            "YOLO",
            f"Mix_{mix}",
            seed,
            path,
            root / "One-Stage" / "Mixed" / "project" / "runs_mix" / f"MIX_{mix}_yolov8m_seed{seed}" / "weights" / "best.pt",
        )

    for path in (root / "Two-Stage" / "Real" / "project" / "metrics").glob("E6_seed*_frcnn_eval.json"):
        seed = int(re.search(r"seed(\d+)", str(path)).group(1))
        add_candidate(
            rows,
            "FRCNN",
            "Real",
            seed,
            path,
            root / "Two-Stage" / "Real" / "project" / "runs_ad_frcnn" / f"E6_frcnn_real_seed{seed}" / "final.pt",
        )

    for path in (root / "Two-Stage" / "Synthetic" / "project" / "metrics").glob("SYNTHIA_seed*_frcnn_on_bdd_eval.json"):
        seed = int(re.search(r"seed(\d+)", str(path)).group(1))
        add_candidate(
            rows,
            "FRCNN",
            "Synthetic",
            seed,
            path,
            root / "Two-Stage" / "Synthetic" / "project" / "runs_synthia_frcnn" / f"SYNTHIA_frcnn_seed{seed}" / "final.pt",
        )

    for path in (root / "Two-Stage" / "Mixed" / "project" / "metrics").glob("MIX_*_seed*_frcnn_on_bdd_eval.json"):
        match = re.search(r"MIX_(\d+_\d+)_seed(\d+)", str(path))
        if not match:
            continue
        mix, seed_s = match.groups()
        seed = int(seed_s)
        add_candidate(
            rows,
            "FRCNN",
            f"Mix_{mix}",
            seed,
            path,
            root / "Two-Stage" / "Mixed" / "project" / "runs_mix_frcnn" / f"MIX_{mix}_frcnn_seed{seed}" / "final.pt",
        )

    return rows


def main() -> None:
    args = parse_args()
    candidates = collect_candidates()
    selected = []
    for key in sorted({(row["model"], row["setup"]) for row in candidates}):
        rows = [row for row in candidates if (row["model"], row["setup"]) == key]
        rows.sort(key=lambda row: (row["AP_50_95"], row["AP_50"], -row["seed"]), reverse=True)
        selected.append(rows[0])

    out = {"selection_metric": "BDD real-target AP_50_95, tie-break AP_50", "checkpoints": selected}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
