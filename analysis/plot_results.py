#!/usr/bin/env python3
"""Generate thesis-ready figures from analysis CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_style import (
    ARCH_LABELS,
    ARCH_ORDER,
    FIG_DIR as DEFAULT_FIG_DIR,
    OUT_DIR,
    REGIME_LABELS,
    REGIME_ORDER,
    SCENARIO_LABELS,
    scenario_title,
)

ARCH_COLORS = {"YOLOv8m": "#1f77b4", "FasterRCNN": "#d62728"}
ARCH_HATCHES = {"YOLOv8m": "", "FasterRCNN": "//"}
ARCH_LINESTYLES = {"YOLOv8m": "-", "FasterRCNN": "--"}
ARCH_MARKERS = {"YOLOv8m": "o", "FasterRCNN": "s"}
SCENARIO_COLORS = {"ADD": "#1f77b4", "IS": "#2ca02c"}
SCENARIO_LINESTYLES = {"ADD": "-", "IS": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot thesis result figures.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def load_internal_summary(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "internal_summary.csv")


def load_seed_summary(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "results_long.csv")


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


def bar_regime(
    df: pd.DataFrame,
    scenario: str,
    eval_dataset: str,
    out_path: Path,
    title: str,
) -> None:
    sub = df[
        (df["scenario"] == scenario)
        & (df["eval_domain"] == "real_internal")
        & (df["eval_dataset"] == eval_dataset)
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(REGIME_ORDER))
    width = 0.38
    for idx, arch in enumerate(ARCH_ORDER):
        vals = []
        errs = []
        for regime in REGIME_ORDER:
            row = sub[(sub["architecture"] == arch) & (sub["regime"] == regime)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + (idx - 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=4,
            label=ARCH_LABELS[arch],
            color=ARCH_COLORS[arch],
            hatch=ARCH_HATCHES[arch],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
        )
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], rotation=16, ha="right")
    style_axis(ax, r"AP$_{50:95}$")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    save(fig, out_path)


def kitti_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "kitti_summary.csv")
    sub = df[(df["metric"] == "AP_50_95") & (df["class"] == "all")].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(REGIME_ORDER))
    width = 0.38
    for idx, arch in enumerate(ARCH_ORDER):
        vals = []
        for regime in REGIME_ORDER:
            row = sub[(sub["architecture"] == arch) & (sub["regime"] == regime)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + (idx - 0.5) * width,
            vals,
            width,
            label=ARCH_LABELS[arch],
            color=ARCH_COLORS[arch],
            hatch=ARCH_HATCHES[arch],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
        )
    ax.set_title(scenario_title("ADD", "KITTI", external=True))
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], rotation=16, ha="right")
    style_axis(ax, r"AP$_{50:95}$")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    save(fig, out_path)


def delta_vs_real_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "delta_vs_real.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
        & (df["regime"] != "real_only")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=False)
    for ax, scenario, dataset in zip(axes, ["ADD", "IS"], ["BDD", "IS_real"]):
        data = sub[(sub["scenario"] == scenario) & (sub["eval_dataset"] == dataset)]
        regimes = REGIME_ORDER[1:]
        x = np.arange(len(regimes))
        width = 0.38
        for idx, arch in enumerate(ARCH_ORDER):
            vals = []
            for regime in regimes:
                row = data[(data["architecture"] == arch) & (data["regime"] == regime)]
                vals.append(float(row["delta"].iloc[0]) if not row.empty else np.nan)
            ax.bar(
                x + (idx - 0.5) * width,
                vals,
                width,
                label=ARCH_LABELS[arch],
                color=ARCH_COLORS[arch],
                hatch=ARCH_HATCHES[arch],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.95,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{SCENARIO_LABELS[scenario]}: Delta Relative to Real-only")
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in regimes], rotation=16, ha="right")
        style_axis(ax, r"$\Delta$ AP$_{50:95}$ vs Real-only")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save(fig, out_path)


def scale_comparison_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "delta_vs_real.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
        & (df["regime"] != "real_only")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
    x = np.arange(len(REGIME_ORDER[1:]))
    for ax, arch in zip(axes, ARCH_ORDER):
        data = sub[sub["architecture"] == arch]
        for scenario in ["ADD", "IS"]:
            dataset = "BDD" if scenario == "ADD" else "IS_real"
            vals = []
            for regime in REGIME_ORDER[1:]:
                row = data[
                    (data["scenario"] == scenario)
                    & (data["eval_dataset"] == dataset)
                    & (data["regime"] == regime)
                ]
                vals.append(float(row["delta"].iloc[0]) if not row.empty else np.nan)
            ax.plot(
                x,
                vals,
                marker="o",
                markersize=7,
                linewidth=2,
                label=SCENARIO_LABELS[scenario],
                color=SCENARIO_COLORS[scenario],
                linestyle=SCENARIO_LINESTYLES[scenario],
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(ARCH_LABELS[arch])
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER[1:]], rotation=16, ha="right")
        style_axis(ax, r"$\Delta$ AP$_{50:95}$ vs Real-only")
        ax.legend(frameon=False)
    fig.suptitle("Regime Effects by Detection Scale Scenario", y=1.03, fontsize=14)
    save(fig, out_path)


def heatmap(ax: plt.Axes, data: pd.DataFrame, title: str) -> plt.AxesImage:
    regimes = [r for r in REGIME_ORDER if r in set(data["regime"])]
    classes = sorted(data["class"].dropna().unique().tolist())
    matrix = np.full((len(regimes), len(classes)), np.nan)
    for i, regime in enumerate(regimes):
        for j, cls in enumerate(classes):
            row = data[(data["regime"] == regime) & (data["class"] == cls)]
            if not row.empty:
                matrix[i, j] = float(row["mean"].iloc[0])
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([REGIME_LABELS[r] for r in regimes])
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    for i in range(len(regimes)):
        for j in range(len(classes)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.45 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    return im


def class_heatmaps(out_dir: Path, fig_dir: Path) -> None:
    df = pd.read_csv(out_dir / "class_summary.csv")
    sub = df[(df["eval_domain"] == "real_internal") & (df["metric"] == "AP_50_95")].copy()
    for scenario, dataset in [("ADD", "BDD"), ("IS", "IS_real")]:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
        images = []
        for ax, arch in zip(axes, ARCH_ORDER):
            data = sub[
                (sub["scenario"] == scenario)
                & (sub["eval_dataset"] == dataset)
                & (sub["architecture"] == arch)
            ]
            images.append(heatmap(ax, data, f"{SCENARIO_LABELS[scenario]} {ARCH_LABELS[arch]}"))
        fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.9, label=r"AP$_{50:95}$")
        fig.suptitle(f"Per-class Internal Real-Test Performance: {SCENARIO_LABELS[scenario]}", y=1.02, fontsize=14)
        save(fig, fig_dir / f"class_heatmap_{scenario.lower()}_real_internal")


def synthetic_real_gap_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "internal_summary.csv")
    sub = df[(df["metric"] == "AP_50_95") & (df["class"] == "all")].copy()
    piv = sub.pivot_table(
        index=["scenario", "architecture", "regime"],
        columns="eval_domain",
        values="mean",
        aggfunc="first",
    ).reset_index()
    if "synthetic_internal" not in piv.columns or "real_internal" not in piv.columns:
        return
    piv["gap"] = piv["synthetic_internal"] - piv["real_internal"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=False)
    for ax, scenario in zip(axes, ["ADD", "IS"]):
        data = piv[piv["scenario"] == scenario]
        x = np.arange(len(REGIME_ORDER))
        width = 0.38
        for idx, arch in enumerate(ARCH_ORDER):
            vals = []
            for regime in REGIME_ORDER:
                row = data[(data["architecture"] == arch) & (data["regime"] == regime)]
                vals.append(float(row["gap"].iloc[0]) if not row.empty else np.nan)
            ax.bar(
                x + (idx - 0.5) * width,
                vals,
                width,
                label=ARCH_LABELS[arch],
                color=ARCH_COLORS[arch],
                hatch=ARCH_HATCHES[arch],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.95,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{SCENARIO_LABELS[scenario]}: Synthetic-to-Real Transfer Gap")
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], rotation=16, ha="right")
        style_axis(ax, r"AP$_{50:95,synthetic}$ - AP$_{50:95,real}$")
    axes[1].legend(frameon=False, loc="upper right")
    save(fig, out_path)


def architecture_gap_summary_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "internal_summary.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
        & (df["regime"].isin(["real_only", "hybrid_70_30", "synthetic_only"]))
        & (df["scenario"].isin(["ADD", "IS"]))
    ].copy()
    pivot = sub.pivot_table(
        index=["scenario", "regime"],
        columns="architecture",
        values="mean",
        aggfunc="first",
    ).reset_index()
    pivot["gap"] = pivot["YOLOv8m"] - pivot["FasterRCNN"]

    scenario_order = ["ADD", "IS"]
    regime_order = ["real_only", "hybrid_70_30", "synthetic_only"]
    labels = []
    values = []
    colors = []
    for scenario in scenario_order:
        for regime in regime_order:
            row = pivot[(pivot["scenario"] == scenario) & (pivot["regime"] == regime)]
            if row.empty:
                continue
            labels.append(f"{scenario}\n{REGIME_LABELS[regime]}")
            values.append(float(row["gap"].iloc[0]))
            colors.append(SCENARIO_COLORS[scenario])

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5, width=0.68)
    for idx, bar in enumerate(bars):
        if idx >= 3:
            bar.set_hatch("//")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Performance Gap Between One-Stage and Two-Stage Detectors")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax, r"$\Delta$ AP$_{50:95}$ (YOLOv8m $-$ Faster R-CNN)")
    ax.text(0.99, 0.97, "Positive values: YOLOv8m performs better", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    save(fig, out_path)


def seed_stability_plot(out_dir: Path, out_path: Path) -> None:
    df = load_seed_summary(out_dir)
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
    for ax, scenario, dataset in zip(axes, ["ADD", "IS"], ["BDD", "IS_real"]):
        data = sub[(sub["scenario"] == scenario) & (sub["eval_dataset"] == dataset)]
        x = np.arange(len(REGIME_ORDER))
        width = 0.38
        for idx, arch in enumerate(ARCH_ORDER):
            means = []
            stds = []
            for regime in REGIME_ORDER:
                rows = data[(data["architecture"] == arch) & (data["regime"] == regime)]
                means.append(float(rows["value"].mean()) if not rows.empty else np.nan)
                stds.append(float(rows["value"].std(ddof=1)) if len(rows) > 1 else 0.0)
            ax.errorbar(
                x + (idx - 0.5) * width / 2,
                means,
                yerr=stds,
                marker=ARCH_MARKERS[arch],
                markersize=6,
                linewidth=1.8,
                capsize=4,
                label=ARCH_LABELS[arch],
                color=ARCH_COLORS[arch],
                linestyle=ARCH_LINESTYLES[arch],
            )
        ax.set_title(f"{SCENARIO_LABELS[scenario]}: Seed Stability on Internal Real-Test AP")
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], rotation=16, ha="right")
        style_axis(ax, r"AP$_{50:95}$ (mean $\pm$ std)")
    axes[1].legend(frameon=False, loc="upper right")
    save(fig, out_path)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    internal = load_internal_summary(args.out_dir)
    bar_regime(
        internal,
        "ADD",
        "BDD",
        fig_dir / "add_internal_real_regime_comparison",
        scenario_title("ADD", "BDD"),
    )
    bar_regime(
        internal,
        "IS",
        "IS_real",
        fig_dir / "is_internal_real_regime_comparison",
        scenario_title("IS", "IS_real"),
    )
    kitti_plot(args.out_dir, fig_dir / "add_kitti_external_comparison")
    delta_vs_real_plot(args.out_dir, fig_dir / "delta_vs_real_internal")
    scale_comparison_plot(args.out_dir, fig_dir / "scale_delta_comparison")
    class_heatmaps(args.out_dir, fig_dir)
    synthetic_real_gap_plot(args.out_dir, fig_dir / "synthetic_to_real_gap")
    architecture_gap_summary_plot(args.out_dir, fig_dir / "architecture_gap_summary")
    seed_stability_plot(args.out_dir, fig_dir / "seed_stability_internal_ap50_95")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
