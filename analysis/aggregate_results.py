#!/usr/bin/env python3
"""Aggregate normalized thesis metrics into summary CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
RESULTS_LONG = OUT_DIR / "results_long.csv"

KEYS = [
    "scenario",
    "scale",
    "architecture",
    "regime",
    "real_pct",
    "synthetic_pct",
    "eval_domain",
    "eval_dataset",
    "metric",
    "class",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate results_long.csv into summary CSVs.")
    parser.add_argument("--input", type=Path, default=RESULTS_LONG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # COCOeval emits -1 for undefined class metrics. Treat those as missing in summaries.
    out.loc[out["value"] < 0, "value"] = pd.NA
    return out


def aggregate_internal(df: pd.DataFrame) -> pd.DataFrame:
    internal = df[df["eval_domain"].isin(["real_internal", "synthetic_internal"])].copy()
    grouped = (
        internal.groupby(KEYS, dropna=False)["value"]
        .agg(mean="mean", std="std", min="min", max="max", n="count")
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def kitti_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = KEYS + ["seed", "selected_seed", "value", "source_file"]
    return df[df["eval_domain"] == "external"][cols].copy()


def combined_for_deltas(internal: pd.DataFrame, kitti: pd.DataFrame) -> pd.DataFrame:
    internal_mean = internal.copy()
    internal_mean["seed"] = pd.NA
    internal_mean["selected_seed"] = False
    internal_mean["summary_value"] = internal_mean["mean"]
    internal_mean["summary_std"] = internal_mean["std"]
    internal_mean = internal_mean[KEYS + ["seed", "selected_seed", "summary_value", "summary_std"]]

    kitti_mean = kitti.copy()
    kitti_mean["summary_value"] = kitti_mean["value"]
    kitti_mean["summary_std"] = pd.NA
    kitti_mean = kitti_mean[KEYS + ["seed", "selected_seed", "summary_value", "summary_std"]]
    return pd.concat([internal_mean, kitti_mean], ignore_index=True)


def delta_vs_baseline(summary: pd.DataFrame, baseline_regime: str) -> pd.DataFrame:
    compare_keys = [
        "scenario",
        "scale",
        "architecture",
        "eval_domain",
        "eval_dataset",
        "metric",
        "class",
    ]
    baseline = summary[summary["regime"] == baseline_regime][compare_keys + ["summary_value"]].rename(
        columns={"summary_value": "baseline_value"}
    )
    merged = summary.merge(baseline, on=compare_keys, how="left")
    merged["baseline_regime"] = baseline_regime
    merged["delta"] = merged["summary_value"] - merged["baseline_value"]
    return merged


def architecture_gap(summary: pd.DataFrame) -> pd.DataFrame:
    compare_keys = [
        "scenario",
        "scale",
        "regime",
        "real_pct",
        "synthetic_pct",
        "eval_domain",
        "eval_dataset",
        "metric",
        "class",
    ]
    pivot = summary.pivot_table(
        index=compare_keys,
        columns="architecture",
        values="summary_value",
        aggfunc="first",
    ).reset_index()
    if "YOLOv8m" not in pivot.columns or "FasterRCNN" not in pivot.columns:
        pivot["architecture_gap"] = pd.NA
    else:
        pivot["architecture_gap"] = pivot["YOLOv8m"] - pivot["FasterRCNN"]
    return pivot


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = clean_values(pd.read_csv(args.input))

    internal = aggregate_internal(df)
    class_summary = internal[internal["class"] != "all"].copy()
    kitti = kitti_summary(df)
    summary = combined_for_deltas(internal, kitti)

    internal[internal["class"] == "all"].to_csv(args.out_dir / "internal_summary.csv", index=False)
    class_summary.to_csv(args.out_dir / "class_summary.csv", index=False)
    kitti.to_csv(args.out_dir / "kitti_summary.csv", index=False)
    delta_vs_baseline(summary, "real_only").to_csv(args.out_dir / "delta_vs_real.csv", index=False)
    delta_vs_baseline(summary, "synthetic_only").to_csv(args.out_dir / "delta_vs_synth.csv", index=False)
    architecture_gap(summary).to_csv(args.out_dir / "architecture_gap.csv", index=False)

    print(f"Wrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
