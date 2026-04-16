#!/usr/bin/env python3
"""Generate thesis-ready figures from analysis CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
FIG_DIR = OUT_DIR / "figures"
REGIME_ORDER = ["real_only", "synthetic_only", "hybrid_70_30", "hybrid_50_50", "hybrid_30_70"]
REGIME_LABELS = {
    "real_only": "Real",
    "synthetic_only": "Synth",
    "hybrid_70_30": "70/30",
    "hybrid_50_50": "50/50",
    "hybrid_30_70": "30/70",
}
ARCHES = ["YOLOv8m", "FasterRCNN"]
ARCH_COLORS = {"YOLOv8m": "#2b6cb0", "FasterRCNN": "#c2410c"}
SCENARIO_COLORS = {"ADD": "#2b6cb0", "IS": "#0f766e"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot thesis result figures.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def metric_summary(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "internal_summary.csv")


def bar_regime(
    df: pd.DataFrame,
    scenario: str,
    eval_dataset: str,
    title: str,
    out_path: Path,
) -> None:
    sub = df[
        (df["scenario"] == scenario)
        & (df["eval_domain"] == "real_internal")
        & (df["eval_dataset"] == eval_dataset)
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
    ].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(REGIME_ORDER))
    width = 0.36
    for i, arch in enumerate(ARCHES):
        vals = []
        errs = []
        for regime in REGIME_ORDER:
            row = sub[(sub["architecture"] == arch) & (sub["regime"] == regime)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=3,
            label=arch,
            color=ARCH_COLORS[arch],
            alpha=0.9,
        )
    ax.set_title(title)
    ax.set_ylabel("AP$_{50:95}$")
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_path)


def kitti_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "kitti_summary.csv")
    sub = df[(df["metric"] == "AP_50_95") & (df["class"] == "all")].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(REGIME_ORDER))
    width = 0.36
    for i, arch in enumerate(ARCHES):
        vals = []
        for regime in REGIME_ORDER:
            row = sub[(sub["architecture"] == arch) & (sub["regime"] == regime)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, vals, width, label=arch, color=ARCH_COLORS[arch], alpha=0.9)
    ax.set_title("ADD External KITTI Evaluation")
    ax.set_ylabel("AP$_{50:95}$")
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_path)


def delta_vs_real_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "delta_vs_real.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=False)
    for ax, scenario, dataset in zip(axes, ["ADD", "IS"], ["BDD", "IS_real"]):
        data = sub[(sub["scenario"] == scenario) & (sub["eval_dataset"] == dataset)]
        x = np.arange(len(REGIME_ORDER))
        width = 0.36
        for i, arch in enumerate(ARCHES):
            vals = []
            for regime in REGIME_ORDER:
                row = data[(data["architecture"] == arch) & (data["regime"] == regime)]
                vals.append(float(row["delta"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=arch, color=ARCH_COLORS[arch], alpha=0.9)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{scenario}: Delta vs Real-Only")
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], rotation=0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AP$_{50:95}$ delta")
    axes[1].legend(frameon=False)
    save(fig, out_path)


def scale_comparison_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "delta_vs_real.csv")
    sub = df[
        (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
        & (df["regime"] != "real_only")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=False)
    x = np.arange(len(REGIME_ORDER[1:]))
    for ax, arch in zip(axes, ARCHES):
        data = sub[sub["architecture"] == arch]
        for scenario in ["ADD", "IS"]:
            vals = []
            for regime in REGIME_ORDER[1:]:
                dataset = "BDD" if scenario == "ADD" else "IS_real"
                row = data[
                    (data["scenario"] == scenario)
                    & (data["eval_dataset"] == dataset)
                    & (data["regime"] == regime)
                ]
                vals.append(float(row["delta"].iloc[0]) if not row.empty else np.nan)
            ax.plot(x, vals, marker="o", label=scenario, color=SCENARIO_COLORS[scenario])
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(arch)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER[1:]])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("AP$_{50:95}$ delta vs real-only")
    save(fig, out_path)


def heatmap(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    regimes = [r for r in REGIME_ORDER if r in set(data["regime"])]
    classes = list(data["class"].drop_duplicates())
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
    ax.set_xticklabels(classes, rotation=35, ha="right")
    for i in range(len(regimes)):
        for j in range(len(classes)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.45 else "black", fontsize=8)
    return im


def class_heatmaps(out_dir: Path) -> None:
    df = pd.read_csv(out_dir / "class_summary.csv")
    sub = df[(df["eval_domain"] == "real_internal") & (df["metric"] == "AP_50_95")].copy()
    for scenario, dataset in [("ADD", "BDD"), ("IS", "IS_real")]:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
        ims = []
        for ax, arch in zip(axes, ARCHES):
            data = sub[(sub["scenario"] == scenario) & (sub["eval_dataset"] == dataset) & (sub["architecture"] == arch)]
            ims.append(heatmap(ax, data, f"{scenario} {arch}: Class AP$_{{50:95}}$"))
        fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.85, label="AP$_{50:95}$")
        save(fig, FIG_DIR / f"class_heatmap_{scenario.lower()}_real_internal")


def synthetic_real_gap_plot(out_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(out_dir / "internal_summary.csv")
    sub = df[(df["metric"] == "AP_50_95") & (df["class"] == "all")].copy()
    piv = sub.pivot_table(
        index=["scenario", "scale", "architecture", "regime"],
        columns="eval_domain",
        values="mean",
        aggfunc="first",
    ).reset_index()
    if "synthetic_internal" not in piv.columns or "real_internal" not in piv.columns:
        return
    piv["gap"] = piv["synthetic_internal"] - piv["real_internal"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=False)
    for ax, scenario in zip(axes, ["ADD", "IS"]):
        data = piv[piv["scenario"] == scenario]
        x = np.arange(len(REGIME_ORDER))
        width = 0.36
        for i, arch in enumerate(ARCHES):
            vals = []
            for regime in REGIME_ORDER:
                row = data[(data["architecture"] == arch) & (data["regime"] == regime)]
                vals.append(float(row["gap"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=arch, color=ARCH_COLORS[arch], alpha=0.9)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{scenario}: Synthetic-to-Real Gap")
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AP$_{synthetic}$ - AP$_{real}$")
    axes[1].legend(frameon=False)
    save(fig, out_path)


def main() -> None:
    args = parse_args()
    global FIG_DIR
    FIG_DIR = args.out_dir / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    internal = metric_summary(args.out_dir)
    bar_regime(internal, "ADD", "BDD", "ADD Internal Real-Test Performance", FIG_DIR / "add_internal_real_regime_comparison")
    bar_regime(internal, "IS", "IS_real", "IS Internal Real-Test Performance", FIG_DIR / "is_internal_real_regime_comparison")
    kitti_plot(args.out_dir, FIG_DIR / "add_kitti_external_comparison")
    delta_vs_real_plot(args.out_dir, FIG_DIR / "delta_vs_real_internal")
    scale_comparison_plot(args.out_dir, FIG_DIR / "scale_delta_comparison")
    class_heatmaps(args.out_dir)
    synthetic_real_gap_plot(args.out_dir, FIG_DIR / "synthetic_to_real_gap")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
