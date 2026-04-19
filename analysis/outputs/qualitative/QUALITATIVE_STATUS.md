# Qualitative Export Status

This repository currently does not contain committed qualitative overlay figures.

## Availability Check
- Ground-truth JSON files found: 0
- Prediction JSON files found: 0
- Candidate image directories found: 0

## Why Qualitative Overlays Cannot Be Generated Now
- The committed repository does not include saved `gt_*.json` and `preds_*.json` evaluation dumps.
- The committed repository also does not include the underlying image datasets.
- Without both prediction files and image files, qualitative bounding-box overlays cannot be reconstructed.

## Minimum Rerun or Save Needed
Regenerate one evaluation per regime while preserving:
- `gt_<split>.json`
- `preds_<split>.json`
- access to the corresponding image split root

Relevant scripts already support writing those files:
- `ADD/_Shared/scripts/05_eval_internal_test_coco_custom_scale.py`
- `IS/_Shared/scripts/03_eval_yolo_coco.py`

Then rerun `python analysis/export_qualitative_examples.py` to create qualitative overlays.
