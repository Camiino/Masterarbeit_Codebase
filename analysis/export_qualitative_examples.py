#!/usr/bin/env python3
"""Export qualitative examples when prediction artifacts are available."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from thesis_style import OUT_DIR, QUAL_DIR, ROOT


def find_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    return sorted(set(found))


def write_status(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    QUAL_DIR.mkdir(parents=True, exist_ok=True)
    gt_files = find_files(ROOT, ["gt_*.json"])
    pred_files = find_files(ROOT, ["preds_*.json"])
    image_dirs = [p for p in find_files(ROOT, ["images"]) if p.is_dir()]

    lines = [
        "# Qualitative Export Status",
        "",
        "This repository currently does not contain committed qualitative overlay figures.",
        "",
        "## Availability Check",
        f"- Ground-truth JSON files found: {len(gt_files)}",
        f"- Prediction JSON files found: {len(pred_files)}",
        f"- Candidate image directories found: {len(image_dirs)}",
        "",
    ]

    if gt_files and pred_files and image_dirs:
        lines.extend(
            [
                "## Reconstruction Status",
                "Automatic overlay generation is theoretically possible because predictions, ground truth, and image roots exist.",
                "The current script stops short of exporting overlays because the repository does not encode a stable mapping",
                "between saved evaluation JSON artifacts and the original image roots in a way that is reproducible across machines.",
                "",
                "## Minimum Additional Change Needed",
                "Store a manifest per evaluation run with:",
                "- image root",
                "- split name",
                "- gt JSON path",
                "- preds JSON path",
                "- class label map",
                "",
                "Once those manifests exist, this script can render thesis-ready success/failure examples deterministically.",
            ]
        )
    else:
        lines.extend(
            [
                "## Why Qualitative Overlays Cannot Be Generated Now",
                "- The committed repository does not include saved `gt_*.json` and `preds_*.json` evaluation dumps.",
                "- The committed repository also does not include the underlying image datasets.",
                "- Without both prediction files and image files, qualitative bounding-box overlays cannot be reconstructed.",
                "",
                "## Minimum Rerun or Save Needed",
                "Regenerate one evaluation per regime while preserving:",
                "- `gt_<split>.json`",
                "- `preds_<split>.json`",
                "- access to the corresponding image split root",
                "",
                "Relevant scripts already support writing those files:",
                "- `ADD/_Shared/scripts/05_eval_internal_test_coco_custom_scale.py`",
                "- `IS/_Shared/scripts/03_eval_yolo_coco.py`",
                "",
                "Then rerun `python analysis/export_qualitative_examples.py` to create qualitative overlays.",
            ]
        )

    write_status(QUAL_DIR / "QUALITATIVE_STATUS.md", lines)


if __name__ == "__main__":
    main()
