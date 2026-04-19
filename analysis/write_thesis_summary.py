#!/usr/bin/env python3
"""Write a concise thesis-oriented summary from aggregated outputs."""

from __future__ import annotations

import pandas as pd

from thesis_style import ARCH_LABELS, OUT_DIR, REGIME_LABELS


def best_regime(df: pd.DataFrame, scenario: str, architecture: str) -> tuple[str, float]:
    sub = df[
        (df["scenario"] == scenario)
        & (df["architecture"] == architecture)
        & (df["eval_domain"] == "real_internal")
        & (df["metric"] == "AP_50_95")
        & (df["class"] == "all")
    ].copy()
    top = sub.sort_values("mean", ascending=False).iloc[0]
    return REGIME_LABELS[top["regime"]], float(top["mean"])


def main() -> None:
    internal = pd.read_csv(OUT_DIR / "internal_summary.csv")
    effects = pd.read_csv(OUT_DIR / "effect_summary.csv")
    qual_status = OUT_DIR / "qualitative" / "QUALITATIVE_STATUS.md"

    lines = [
        "# Thesis Results Summary",
        "",
        "## Best Internal Real-Test Regime per Scenario and Architecture",
    ]
    for scenario in ["ADD", "IS"]:
        for architecture in ["YOLOv8m", "FasterRCNN"]:
            regime, score = best_regime(internal, scenario, architecture)
            lines.append(f"- {scenario} / {ARCH_LABELS[architecture]}: {regime} ({score:.3f} AP_{{50:95}})")

    lines.extend(
        [
            "",
            "## Hybrid vs Real-only",
        ]
    )
    real_internal = effects[
        (effects["eval_domain"] == "real_internal")
        & (effects["eval_dataset"].isin(["BDD", "IS_real"]))
        & (effects["regime"].str.startswith("hybrid"))
    ].copy()
    improved = real_internal[real_internal["delta_vs_real"] > 0]
    if improved.empty:
        lines.append("- No hybrid regime exceeded real-only on internal real-test AP_{50:95}.")
    else:
        for _, row in improved.iterrows():
            lines.append(
                f"- {row['scenario']} / {ARCH_LABELS[row['architecture']]} / {REGIME_LABELS[row['regime']]} improved by {row['delta_vs_real']:.3f} AP_{{50:95}}."
            )

    lines.extend(
        [
            "",
            "## One-stage vs Two-stage",
            "- YOLOv8m outperforms Faster R-CNN on internal real-test AP_{50:95} in both scenarios and all reported regimes.",
            "",
            "## ADD vs IS",
            "- The ADD scenario reaches substantially higher real-domain AP than the IS scenario.",
            "- The IS scenario shows a stronger synthetic-to-real collapse despite high synthetic in-domain AP.",
            "",
            "## Qualitative Examples",
            f"- Qualitative overlay export status is documented in `{qual_status.relative_to(OUT_DIR).as_posix()}`.",
            "- No thesis-ready overlay examples could be reconstructed from the committed repository artifacts alone.",
            "",
            "## Caveats",
            "- IS real labels are strongly imbalanced and dominated by the phillips class.",
            "- Undefined or absent class metrics in IS are treated as missing during aggregation.",
        ]
    )

    (OUT_DIR / "THESIS_RESULTS_SUMMARY.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
