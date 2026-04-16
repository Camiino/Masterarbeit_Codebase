# IS Scripts

Industrial-screw pipeline scripts live here. Run them from the scenario root:

```bash
cd IS
```

## Main Entry Points

```text
run_real_all.sh            # real-only YOLO and Faster R-CNN, seeds 1-3
run_synthetic_all.sh       # synthetic-only YOLO and Faster R-CNN, seeds 1-3
run_mixed_all.sh           # mixed real/synthetic YOLO and Faster R-CNN, seeds 1-3
run_real_only_prepare.sh   # prepare real 488-image split
run_synthetic_prepare.sh   # prepare synthetic 488-image controlled split
run_synthetic_evals_only.sh
```

Historical entry points are kept as wrappers/targets for reproducibility:

```text
run_budgeted_real.sh
run_budgeted_synthetic.sh
run_budgeted_mixed.sh
```

## Python Scripts

```text
00_build_label_mapping.py          # validate class mapping and label distribution
01_prepare_real_yolo_splits.py     # materialize real or synthetic YOLO split roots
02_train_yolov8m_real.py           # YOLOv8m training helper; name is historical
03_eval_yolo_coco.py               # YOLO COCO-style evaluation
04_train_frcnn_real.py             # Faster R-CNN training helper; name is historical
05_eval_frcnn_yolo.py              # Faster R-CNN COCO-style evaluation on YOLO splits
11_prepare_mixed_yolo_dataset.py   # build mixed real/synthetic train and val roots
```

The training helper names still include `real` because they were first created for
real-only runs. The scripts are now parameterized and are used for real, synthetic,
and mixed datasets.
