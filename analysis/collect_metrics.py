#!/usr/bin/env python3
"""Collect experiment metric JSON files into one long-format CSV.

The collector intentionally ignores prediction/ground-truth JSON files and
training traces. It only consumes JSONs with the evaluation schema:
{"all": {...}, "per_class": {...}}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
OUT_CSV = OUT_DIR / "results_long.csv"

REGIME_ORDER = {
    "real_only": (100, 0),
    "synthetic_only": (0, 100),
    "hybrid_70_30": (70, 30),
    "hybrid_50_50": (50, 50),
    "hybrid_30_70": (30, 70),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ADD/IS evaluation metrics into a long CSV.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root.")
    parser.add_argument("--out", type=Path, default=OUT_CSV, help="Output CSV path.")
    return parser.parse_args()


def metric_files(root: Path) -> Iterable[Path]:
    for scenario in ["ADD", "IS"]:
        base = root / scenario
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            name = path.name
            if name.startswith(("gt_", "preds_")) or name in {"metrics_all.json", "last_metrics.json"}:
                continue
            if "runs_" in str(path):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict) and isinstance(data.get("all"), dict) and isinstance(data.get("per_class"), dict):
                yield path


def seed_from_path(path: Path) -> int | None:
    matches = re.findall(r"seed(\d+)", str(path))
    return int(matches[-1]) if matches else None


def infer_architecture(path: Path) -> str:
    text = str(path)
    if "/One-Stage/" in text:
        return "YOLOv8m"
    if "/Two-Stage/" in text:
        return "FasterRCNN"
    raise ValueError(f"Cannot infer architecture from {path}")


def infer_regime(path: Path) -> str:
    text = str(path)
    if re.search(r"MIX[_-]70[_-]30|mixed.*70.*30", text):
        return "hybrid_70_30"
    if re.search(r"MIX[_-]50[_-]50|mixed.*50.*50", text):
        return "hybrid_50_50"
    if re.search(r"MIX[_-]30[_-]70|mixed.*30.*70", text):
        return "hybrid_30_70"
    if "/Synthetic/" in text or "SYNTHIA_" in text or "IS_SYNTH" in text:
        return "synthetic_only"
    if "/Real/" in text or "E1_" in text or "E6_" in text or "IS_REAL" in text:
        return "real_only"
    raise ValueError(f"Cannot infer regime from {path}")


def infer_eval(path: Path, scenario: str) -> tuple[str, str]:
    text = str(path)
    if "metrics_kitti" in text or "kitti_eval" in text:
        return "external", "KITTI"

    if scenario == "ADD":
        synthetic_tokens = [
            "on_synthia",
            "metrics_synthia_transfer",
            "metrics_synthia_indomain",
            "/synthia/metrics.json",
            "/metrics_indomain/SYNTHIA_",
        ]
        if any(token in text for token in synthetic_tokens):
            return "synthetic_internal", "SYNTHIA"
        return "real_internal", "BDD"

    synthetic_tokens = [
        "on_synthetic",
        "metrics_synthetic_indomain",
        "/synthetic/metrics.json",
        "/metrics_indomain/IS_SYNTH",
    ]
    if any(token in text for token in synthetic_tokens):
        return "synthetic_internal", "IS_synth"
    return "real_internal", "IS_real"


def rows_for_metric_file(path: Path, root: Path) -> list[dict[str, Any]]:
    rel = path.relative_to(root)
    scenario = rel.parts[0]
    scale = "large" if scenario == "ADD" else "small"
    architecture = infer_architecture(path)
    regime = infer_regime(path)
    real_pct, synthetic_pct = REGIME_ORDER[regime]
    seed = seed_from_path(path)
    eval_domain, eval_dataset = infer_eval(path, scenario)
    selected_seed = eval_domain == "external"

    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    def add_metric(class_name: str, metrics: dict[str, Any]) -> None:
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "scenario": scenario,
                        "scale": scale,
                        "architecture": architecture,
                        "regime": regime,
                        "real_pct": real_pct,
                        "synthetic_pct": synthetic_pct,
                        "seed": seed,
                        "eval_domain": eval_domain,
                        "eval_dataset": eval_dataset,
                        "metric": metric,
                        "class": class_name,
                        "value": float(value),
                        "selected_seed": bool(selected_seed),
                        "source_file": str(rel),
                    }
                )

    add_metric("all", data["all"])
    for class_name, metrics in data["per_class"].items():
        if isinstance(metrics, dict):
            add_metric(str(class_name), metrics)
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for path in metric_files(args.root):
        try:
            rows.extend(rows_for_metric_file(path, args.root))
        except Exception as exc:  # noqa: BLE001 - continue collecting other metrics
            skipped.append((str(path), str(exc)))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No evaluation metrics found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} metric rows to {args.out}")
    print(f"Metric files parsed: {df['source_file'].nunique()}")
    if skipped:
        print(f"Skipped {len(skipped)} files:")
        for path, reason in skipped[:20]:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()
