#!/usr/bin/env python3
"""Export compact thesis-ready LaTeX tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thesis_style import (
    ARCH_LABELS,
    ARCH_ORDER,
    METRIC_LABELS,
    METRIC_ORDER,
    OUT_DIR,
    REGIME_LABELS,
    REGIME_ORDER,
    TABLE_DIR as DEFAULT_TABLE_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaTeX result tables.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def fmt_float(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def fmt_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std) or std == 0:
        return fmt_float(mean)
    return f"{fmt_float(mean)} $\\pm$ {fmt_float(std)}"


def write_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="ll" + "r" * (len(df.columns) - 2),
    )
    path.write_text(latex, encoding="utf-8")


def add_pretty_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["architecture_label"] = out["architecture"].map(ARCH_LABELS).fillna(out["architecture"])
    out["regime_label"] = out["regime"].map(REGIME_LABELS).fillna(out["regime"])
    return out


def internal_main_table(out_dir: Path, scenario: str, dataset: str, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "internal_summary.csv")
    df = add_pretty_names(
        df[
            (df["scenario"] == scenario)
            & (df["eval_domain"] == "real_internal")
            & (df["eval_dataset"] == dataset)
            & (df["class"] == "all")
        ].copy()
    )
    rows = []
    for arch in ARCH_ORDER:
        for regime in REGIME_ORDER:
            sub = df[(df["architecture"] == arch) & (df["regime"] == regime)]
            rec = {
                "Architecture": ARCH_LABELS[arch],
                "Regime": REGIME_LABELS[regime],
            }
            for metric in METRIC_ORDER:
                row = sub[sub["metric"] == metric]
                rec[METRIC_LABELS[metric]] = (
                    fmt_mean_std(float(row["mean"].iloc[0]), float(row["std"].iloc[0])) if not row.empty else "--"
                )
            rows.append(rec)
    write_table(pd.DataFrame(rows), path, caption, label)


def kitti_table(out_dir: Path, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "kitti_summary.csv")
    df = add_pretty_names(df[(df["class"] == "all") & (df["metric"].isin(["AP_50_95", "AP_50", "AP_75"]))].copy())
    rows = []
    for arch in ARCH_ORDER:
        for regime in REGIME_ORDER:
            sub = df[(df["architecture"] == arch) & (df["regime"] == regime)]
            rec = {
                "Architecture": ARCH_LABELS[arch],
                "Regime": REGIME_LABELS[regime],
            }
            for metric in ["AP_50_95", "AP_50", "AP_75"]:
                row = sub[sub["metric"] == metric]
                rec[METRIC_LABELS[metric]] = fmt_float(float(row["value"].iloc[0])) if not row.empty else "--"
            rows.append(rec)
    write_table(pd.DataFrame(rows), path, caption, label)


def class_ap_table(out_dir: Path, scenario: str, dataset: str, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "class_summary.csv")
    df = add_pretty_names(
        df[
            (df["scenario"] == scenario)
            & (df["eval_domain"] == "real_internal")
            & (df["eval_dataset"] == dataset)
            & (df["metric"] == "AP_50_95")
        ].copy()
    )
    classes = list(df["class"].drop_duplicates())
    rows = []
    for arch in ARCH_ORDER:
        for regime in REGIME_ORDER:
            sub = df[(df["architecture"] == arch) & (df["regime"] == regime)]
            rec = {
                "Architecture": ARCH_LABELS[arch],
                "Regime": REGIME_LABELS[regime],
            }
            for cls in classes:
                row = sub[sub["class"] == cls]
                rec[cls] = fmt_mean_std(float(row["mean"].iloc[0]), float(row["std"].iloc[0])) if not row.empty else "--"
            rows.append(rec)
    write_table(pd.DataFrame(rows), path, caption, label)


def effect_delta_table(out_dir: Path, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "effect_summary.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["eval_dataset"].isin(["BDD", "IS_real"]))
    ].copy()
    rows = []
    for _, row in sub.sort_values(["scenario", "architecture", "regime"]).iterrows():
        rows.append(
            {
                "Scenario": row["scenario"],
                "Architecture": ARCH_LABELS.get(row["architecture"], row["architecture"]),
                "Regime": REGIME_LABELS.get(row["regime"], row["regime"]),
                METRIC_LABELS["AP_50_95"]: fmt_float(row["summary_value"]),
                "$\\Delta$ vs Real-only": fmt_float(row["delta_vs_real"]),
                "$\\Delta$ vs Synthetic-only": fmt_float(row["delta_vs_synthetic"]),
                "YOLOv8m - Faster R-CNN": fmt_float(row["architecture_gap"]),
            }
        )
    write_table(pd.DataFrame(rows), path, caption, label)


def scale_delta_table(out_dir: Path, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "delta_vs_real.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
        & (df["regime"] != "real_only")
        & (df["eval_dataset"].isin(["BDD", "IS_real"]))
    ].copy()
    rows = []
    for arch in ARCH_ORDER:
        for regime in REGIME_ORDER[1:]:
            rec = {
                "Architecture": ARCH_LABELS[arch],
                "Regime": REGIME_LABELS[regime],
            }
            for scenario, dataset in [("ADD", "BDD"), ("IS", "IS_real")]:
                row = sub[
                    (sub["architecture"] == arch)
                    & (sub["regime"] == regime)
                    & (sub["scenario"] == scenario)
                    & (sub["eval_dataset"] == dataset)
                ]
                rec[scenario] = fmt_float(float(row["delta"].iloc[0])) if not row.empty else "--"
            rows.append(rec)
    write_table(pd.DataFrame(rows), path, caption, label)


def seed_stability_table(out_dir: Path, path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(out_dir / "internal_summary.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
        & (df["eval_dataset"].isin(["BDD", "IS_real"]))
    ].copy()
    rows = []
    for _, row in sub.sort_values(["scenario", "architecture", "regime"]).iterrows():
        rows.append(
            {
                "Scenario": row["scenario"],
                "Architecture": ARCH_LABELS.get(row["architecture"], row["architecture"]),
                "Regime": REGIME_LABELS.get(row["regime"], row["regime"]),
                "Mean": fmt_float(row["mean"]),
                "Std": fmt_float(row["std"]),
                "Min": fmt_float(row["min"]),
                "Max": fmt_float(row["max"]),
                "n": int(row["n"]),
            }
        )
    write_table(pd.DataFrame(rows), path, caption, label)


def main() -> None:
    args = parse_args()
    table_dir = args.out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    internal_main_table(
        args.out_dir,
        "ADD",
        "BDD",
        table_dir / "table_add_internal.tex",
        "Internal evaluation on the ADD real test set (mean $\\pm$ standard deviation across three seeds).",
        "tab:add_internal_results",
    )
    kitti_table(
        args.out_dir,
        table_dir / "table_add_kitti.tex",
        "External evaluation on KITTI using the selected best seed per setup.",
        "tab:add_kitti_results",
    )
    internal_main_table(
        args.out_dir,
        "IS",
        "IS_real",
        table_dir / "table_is_internal.tex",
        "Internal evaluation on the IS real test set (mean $\\pm$ standard deviation across three seeds).",
        "tab:is_internal_results",
    )
    class_ap_table(
        args.out_dir,
        "ADD",
        "BDD",
        table_dir / "table_class_ap_add.tex",
        "Per-class internal real-test AP on the ADD scenario.",
        "tab:add_class_ap",
    )
    class_ap_table(
        args.out_dir,
        "IS",
        "IS_real",
        table_dir / "table_class_ap_is.tex",
        "Per-class internal real-test AP on the IS scenario.",
        "tab:is_class_ap",
    )
    effect_delta_table(
        args.out_dir,
        table_dir / "table_effect_deltas.tex",
        "Regime effects relative to real-only and synthetic-only training on internal real-test AP.",
        "tab:effect_deltas",
    )
    scale_delta_table(
        args.out_dir,
        table_dir / "table_scale_delta_comparison.tex",
        "Cross-scenario comparison of regime deltas relative to real-only training.",
        "tab:scale_delta_comparison",
    )
    seed_stability_table(
        args.out_dir,
        table_dir / "table_seed_stability_internal.tex",
        "Seed stability summary for internal real-test AP across scenarios, architectures, and regimes.",
        "tab:seed_stability",
    )
    print(f"Wrote LaTeX tables to {table_dir}")


if __name__ == "__main__":
    main()
