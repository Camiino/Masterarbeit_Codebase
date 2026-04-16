#!/usr/bin/env python3
"""Export compact LaTeX tables for thesis inclusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
TABLE_DIR = OUT_DIR / "tables"
REGIME_ORDER = ["real_only", "synthetic_only", "hybrid_70_30", "hybrid_50_50", "hybrid_30_70"]
REGIME_LABELS = {
    "real_only": "Real-only",
    "synthetic_only": "Synthetic-only",
    "hybrid_70_30": "Hybrid 70/30",
    "hybrid_50_50": "Hybrid 50/50",
    "hybrid_30_70": "Hybrid 30/70",
}
METRIC_LABELS = {
    "AP_50_95": "AP$_{50:95}$",
    "AP_50": "AP$_{50}$",
    "AP_75": "AP$_{75}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaTeX result tables.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def fmt_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f} $\\pm$ {0.0 if pd.isna(std) else std:.3f}"


def fmt_value(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def write_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_latex(index=False, escape=False, caption=caption, label=label)
    path.write_text(text, encoding="utf-8")


def internal_table(out_dir: Path, scenario: str, dataset: str, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "internal_summary.csv")
    sub = df[
        (df["scenario"] == scenario)
        & (df["eval_domain"] == "real_internal")
        & (df["eval_dataset"] == dataset)
        & (df["metric"].isin(["AP_50_95", "AP_50", "AP_75"]))
        & (df["class"] == "all")
    ].copy()
    rows = []
    for regime in REGIME_ORDER:
        for arch in ["YOLOv8m", "FasterRCNN"]:
            rec = {"Regime": REGIME_LABELS[regime], "Architecture": arch}
            for metric in ["AP_50_95", "AP_50", "AP_75"]:
                row = sub[(sub["regime"] == regime) & (sub["architecture"] == arch) & (sub["metric"] == metric)]
                rec[METRIC_LABELS[metric]] = (
                    fmt_mean_std(float(row["mean"].iloc[0]), float(row["std"].iloc[0])) if not row.empty else "--"
                )
            rows.append(rec)
    write_table(pd.DataFrame(rows), path, caption, label)


def kitti_table(out_dir: Path) -> None:
    df = pd.read_csv(out_dir / "kitti_summary.csv")
    sub = df[(df["metric"].isin(["AP_50_95", "AP_50", "AP_75"])) & (df["class"] == "all")].copy()
    rows = []
    for regime in REGIME_ORDER:
        for arch in ["YOLOv8m", "FasterRCNN"]:
            rec = {"Regime": REGIME_LABELS[regime], "Architecture": arch}
            for metric in ["AP_50_95", "AP_50", "AP_75"]:
                row = sub[(sub["regime"] == regime) & (sub["architecture"] == arch) & (sub["metric"] == metric)]
                if not row.empty:
                    rec[METRIC_LABELS[metric]] = fmt_value(float(row["value"].iloc[0]))
                else:
                    rec[METRIC_LABELS[metric]] = "--"
            rows.append(rec)
    write_table(
        pd.DataFrame(rows),
        TABLE_DIR / "table_add_kitti.tex",
        "ADD external KITTI performance.",
        "tab:add_kitti_results",
    )


def class_table(out_dir: Path, scenario: str, dataset: str, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "class_summary.csv")
    sub = df[
        (df["scenario"] == scenario)
        & (df["eval_domain"] == "real_internal")
        & (df["eval_dataset"] == dataset)
        & (df["metric"] == "AP_50_95")
    ].copy()
    classes = list(sub["class"].drop_duplicates())
    rows = []
    for regime in REGIME_ORDER:
        for arch in ["YOLOv8m", "FasterRCNN"]:
            rec = {"Regime": REGIME_LABELS[regime], "Architecture": arch}
            for cls in classes:
                row = sub[(sub["regime"] == regime) & (sub["architecture"] == arch) & (sub["class"] == cls)]
                rec[cls] = fmt_mean_std(float(row["mean"].iloc[0]), float(row["std"].iloc[0])) if not row.empty else "--"
            rows.append(rec)
    write_table(pd.DataFrame(rows), path, caption, label)


def effect_delta_table(out_dir: Path) -> None:
    df = pd.read_csv(out_dir / "effect_summary.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["regime"].str.startswith("hybrid"))
        & (df["eval_dataset"].isin(["BDD", "IS_real"]))
    ].copy()
    rows = []
    for _, row in sub.sort_values(["scenario", "architecture", "regime"]).iterrows():
        rows.append(
            {
                "Scenario": row["scenario"],
                "Architecture": row["architecture"],
                "Regime": REGIME_LABELS.get(row["regime"], row["regime"]),
                "AP": fmt_value(row["summary_value"]),
                "$\\Delta$ vs real": fmt_value(row["delta_vs_real"]),
                "$\\Delta$ vs synthetic": fmt_value(row["delta_vs_synthetic"]),
                "YOLO--FRCNN gap": fmt_value(row["architecture_gap"]),
            }
        )
    write_table(
        pd.DataFrame(rows),
        TABLE_DIR / "table_effect_deltas.tex",
        "Hybrid-regime descriptive effects on real internal test sets.",
        "tab:effect_deltas",
    )


def main() -> None:
    args = parse_args()
    global TABLE_DIR
    TABLE_DIR = args.out_dir / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    internal_table(
        args.out_dir,
        "ADD",
        "BDD",
        TABLE_DIR / "table_add_internal.tex",
        "ADD internal real-test performance aggregated across seeds.",
        "tab:add_internal_results",
    )
    internal_table(
        args.out_dir,
        "IS",
        "IS_real",
        TABLE_DIR / "table_is_internal.tex",
        "IS internal real-test performance aggregated across seeds.",
        "tab:is_internal_results",
    )
    kitti_table(args.out_dir)
    class_table(
        args.out_dir,
        "ADD",
        "BDD",
        TABLE_DIR / "table_class_ap_add.tex",
        "ADD class-specific AP on the internal real test set.",
        "tab:add_class_ap",
    )
    class_table(
        args.out_dir,
        "IS",
        "IS_real",
        TABLE_DIR / "table_class_ap_is.tex",
        "IS class-specific AP on the internal real test set.",
        "tab:is_class_ap",
    )
    effect_delta_table(args.out_dir)
    print(f"Wrote LaTeX tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
