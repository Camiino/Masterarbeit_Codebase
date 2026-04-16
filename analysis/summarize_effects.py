#!/usr/bin/env python3
"""Create descriptive effect summaries aligned with the thesis questions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
REGIME_ORDER = ["real_only", "synthetic_only", "hybrid_70_30", "hybrid_50_50", "hybrid_30_70"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize descriptive effects for thesis interpretation.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def load_clean_long(out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(out_dir / "results_long.csv")
    df.loc[df["value"] < 0, "value"] = pd.NA
    return df


def effect_summary(out_dir: Path) -> pd.DataFrame:
    real = pd.read_csv(out_dir / "delta_vs_real.csv")
    synth = pd.read_csv(out_dir / "delta_vs_synth.csv")
    arch = pd.read_csv(out_dir / "architecture_gap.csv")

    filters = {"metric": "AP_50_95", "class": "all"}
    real = real[(real["metric"] == filters["metric"]) & (real["class"] == filters["class"])].copy()
    synth = synth[(synth["metric"] == filters["metric"]) & (synth["class"] == filters["class"])].copy()
    arch = arch[(arch["metric"] == filters["metric"]) & (arch["class"] == filters["class"])].copy()

    keys = [
        "scenario",
        "scale",
        "architecture",
        "regime",
        "eval_domain",
        "eval_dataset",
        "summary_value",
    ]
    out = real[keys + ["delta"]].rename(columns={"delta": "delta_vs_real"})
    out = out.merge(
        synth[keys[:-1] + ["delta"]].rename(columns={"delta": "delta_vs_synthetic"}),
        on=keys[:-1],
        how="left",
    )
    out = out.merge(
        arch[
            [
                "scenario",
                "scale",
                "regime",
                "eval_domain",
                "eval_dataset",
                "architecture_gap",
            ]
        ],
        on=["scenario", "scale", "regime", "eval_domain", "eval_dataset"],
        how="left",
    )
    return out


def consistency_summary(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        (df["eval_domain"].isin(["real_internal", "synthetic_internal"]))
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
    ].copy()
    rows: list[dict] = []
    compare_keys = ["scenario", "scale", "architecture", "eval_domain", "eval_dataset"]
    for keys, group in subset.groupby(compare_keys, dropna=False):
        pivot = group.pivot_table(index="seed", columns="regime", values="value", aggfunc="first")
        for baseline in ["real_only", "synthetic_only"]:
            if baseline not in pivot.columns:
                continue
            for regime in REGIME_ORDER:
                if regime == baseline or regime not in pivot.columns:
                    continue
                paired = pivot[[baseline, regime]].dropna()
                if paired.empty:
                    continue
                diff = paired[regime] - paired[baseline]
                rows.append(
                    {
                        **dict(zip(compare_keys, keys)),
                        "regime": regime,
                        "baseline_regime": baseline,
                        "n_paired_seeds": int(len(diff)),
                        "improved_seed_count": int((diff > 0).sum()),
                        "declined_seed_count": int((diff < 0).sum()),
                        "tied_seed_count": int((diff == 0).sum()),
                        "mean_paired_delta": float(diff.mean()),
                        "min_paired_delta": float(diff.min()),
                        "max_paired_delta": float(diff.max()),
                    }
                )
    return pd.DataFrame(rows)


def rankings(out_dir: Path) -> pd.DataFrame:
    internal = pd.read_csv(out_dir / "internal_summary.csv")
    kitti = pd.read_csv(out_dir / "kitti_summary.csv")

    rows = []
    internal = internal[(internal["metric"] == "AP_50_95") & (internal["class"] == "all")].copy()
    internal["score"] = internal["mean"]
    internal["seed_note"] = "mean_over_seeds"
    rows.append(
        internal[
            [
                "scenario",
                "scale",
                "architecture",
                "regime",
                "eval_domain",
                "eval_dataset",
                "score",
                "seed_note",
            ]
        ]
    )

    kitti = kitti[(kitti["metric"] == "AP_50_95") & (kitti["class"] == "all")].copy()
    kitti["score"] = kitti["value"]
    kitti["seed_note"] = "selected_seed_" + kitti["seed"].astype("Int64").astype(str)
    rows.append(
        kitti[
            [
                "scenario",
                "scale",
                "architecture",
                "regime",
                "eval_domain",
                "eval_dataset",
                "score",
                "seed_note",
            ]
        ]
    )
    out = pd.concat(rows, ignore_index=True)
    out["rank"] = out.groupby(["scenario", "architecture", "eval_domain", "eval_dataset"])["score"].rank(
        ascending=False, method="dense"
    )
    return out.sort_values(["scenario", "eval_domain", "architecture", "rank", "regime"])


def main() -> None:
    args = parse_args()
    df = load_clean_long(args.out_dir)
    effects = effect_summary(args.out_dir)
    consistency = consistency_summary(df)
    ranks = rankings(args.out_dir)

    effects.to_csv(args.out_dir / "effect_summary.csv", index=False)
    consistency.to_csv(args.out_dir / "consistency_summary.csv", index=False)
    ranks.to_csv(args.out_dir / "rankings.csv", index=False)
    print(f"Wrote effect summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
