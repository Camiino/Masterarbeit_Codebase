#!/usr/bin/env python3
"""Write a master inventory of thesis analysis outputs."""

from __future__ import annotations

from pathlib import Path

from thesis_style import FIG_DIR, OUT_DIR, QUAL_DIR, TABLE_DIR


SECTION_MAP = [
    ("5.3.1 Internal Evaluation Results (ADD)", ["figures/add_internal_real_regime_comparison.png", "tables/table_add_internal.tex"]),
    ("5.3.2 External Evaluation on KITTI", ["figures/add_kitti_external_comparison.png", "tables/table_add_kitti.tex"]),
    ("5.3.3 Internal Evaluation Results (IS)", ["figures/is_internal_real_regime_comparison.png", "tables/table_is_internal.tex"]),
    ("5.3.4 Regime Deltas Relative to Real-only", ["figures/delta_vs_real_internal.png", "tables/table_effect_deltas.tex"]),
    ("5.3.5 Large-scale vs Small-scale Comparison", ["figures/scale_delta_comparison.png", "tables/table_scale_delta_comparison.tex"]),
    ("5.3.6 Class-wise Results", ["figures/class_heatmap_add_real_internal.png", "figures/class_heatmap_is_real_internal.png", "tables/table_class_ap_add.tex", "tables/table_class_ap_is.tex"]),
    ("5.3.7 Synthetic-to-Real Transfer Gap", ["figures/synthetic_to_real_gap.png"]),
    ("5.3.8 Seed Stability and Result Robustness", ["figures/seed_stability_internal_ap50_95.png", "tables/table_seed_stability_internal.tex"]),
    ("5.3.9 Qualitative Examples", ["qualitative/QUALITATIVE_STATUS.md"]),
]


def relpaths(root: Path, base: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(base)).replace("\\", "/") for path in root.rglob("*") if path.is_file())


def main() -> None:
    csvs = relpaths(OUT_DIR, OUT_DIR)
    figures = relpaths(FIG_DIR, OUT_DIR)
    tables = relpaths(TABLE_DIR, OUT_DIR)
    qualitative = relpaths(QUAL_DIR, OUT_DIR)

    lines = [
        "# Results Inventory",
        "",
        "## CSV Outputs",
    ]
    lines.extend(f"- `{path}`" for path in csvs if path.endswith(".csv"))
    lines.extend(
        [
            "",
            "## Tables",
        ]
    )
    lines.extend(f"- `{path}`" for path in tables)
    lines.extend(
        [
            "",
            "## Figures",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    lines.extend(
        [
            "",
            "## Qualitative Outputs",
        ]
    )
    lines.extend(f"- `{path}`" for path in qualitative)
    lines.extend(
        [
            "",
            "## Thesis Subsection Mapping",
        ]
    )
    for section, assets in SECTION_MAP:
        lines.append(f"- {section} -> {', '.join(f'`{asset}`' for asset in assets)}")

    (OUT_DIR / "RESULTS_INVENTORY.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
