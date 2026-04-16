#!/usr/bin/env python3
"""Summarize ADD KITTI external evaluation JSONs into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

SCENARIO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize KITTI external metrics.")
    p.add_argument(
        "--one-stage-root",
        type=Path,
        default=SCENARIO_ROOT / "One-Stage" / "External" / "project" / "metrics_kitti",
    )
    p.add_argument(
        "--two-stage-root",
        type=Path,
        default=SCENARIO_ROOT / "Two-Stage" / "External" / "project" / "metrics_kitti",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=SCENARIO_ROOT / "External" / "project" / "kitti_summary",
        help="Output prefix without extension.",
    )
    return p.parse_args()


def infer_setup(name: str) -> str:
    if name.startswith("E1_") or name.startswith("E6_"):
        return "Real"
    if name.startswith("SYNTHIA_"):
        return "Synthetic"
    match = re.search(r"MIX_(\d+)_(\d+)", name)
    if match:
        return f"Mix {match.group(1)}/{match.group(2)}"
    return "Other"


def infer_seed(name: str) -> int | None:
    match = re.search(r"seed(\d+)", name)
    return int(match.group(1)) if match else None


def remove_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def row_from_metrics(model: str, group: str, run_name: str, path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    per_class = data.get("per_class", {})
    return {
        "model": model,
        "group": group,
        "setup": infer_setup(run_name),
        "seed": infer_seed(run_name),
        "run": run_name,
        "AP_50_95": data["all"]["AP_50_95"],
        "AP_50": data["all"]["AP_50"],
        "AP_75": data["all"]["AP_75"],
        "AP_car": per_class.get("car", {}).get("AP_50_95"),
        "AP_pedestrian": per_class.get("pedestrian", {}).get("AP_50_95"),
        "AP_cyclist": per_class.get("cyclist", {}).get("AP_50_95"),
        "path": str(path),
    }


def collect_yolo(root: Path) -> List[Dict]:
    rows = []
    for path in sorted(root.glob("*/*_kitti_eval/internal/metrics.json")):
        group = path.relative_to(root).parts[0]
        run_name = remove_suffix(path.relative_to(root).parts[1], "_kitti_eval")
        rows.append(row_from_metrics("YOLO", group, run_name, path))
    return rows


def collect_frcnn(root: Path) -> List[Dict]:
    rows = []
    for path in sorted(root.glob("*/*_kitti_eval.json")):
        group = path.relative_to(root).parts[0]
        run_name = remove_suffix(path.stem, "_kitti_eval")
        rows.append(row_from_metrics("FRCNN", group, run_name, path))
    return rows


def write_markdown(rows: List[Dict], out_path: Path) -> None:
    headers = ["model", "setup", "seed", "AP_50_95", "AP_50", "AP_car", "AP_pedestrian", "AP_cyclist"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for header in headers:
            val = row[header]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            elif val is None:
                vals.append("")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_yolo(args.one_stage_root) + collect_frcnn(args.two_stage_root)
    rows.sort(key=lambda r: (r["model"], r["setup"], r["seed"] or 0, r["run"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = args.out.with_suffix(".csv")
    md_path = args.out.with_suffix(".md")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
