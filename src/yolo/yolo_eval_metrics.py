"""
Refactored YOLO evaluation orchestrator.
Runs predictions, builds ground truth, applies multiple collapse strategies (top1/presence/strict),
then computes classifier-style metrics and writes comparable confusion matrices and a preds_debug.csv.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import os
import json
import pandas as pd
from ultralytics import YOLO

from .evaluation.ground_truth import build_ground_truth_dataframe
from .evaluation.strategies import predict_top1, predict_presence, predict_strict_set_match
from .evaluation.metrics import compute_confusion, compute_paper_metrics, confusion_df
from .evaluation.reporting import write_preds_debug, write_summary


def _collect_yolo_predictions(model: YOLO, test_images_dir: Path, conf_thresh: float) -> pd.DataFrame:
    results = model.predict(source=str(test_images_dir), conf=conf_thresh, save=False, verbose=False, workers=4, stream=True)
    preds = []
    for result in results:
        img_path = Path(result.path)
        if result.boxes is not None and len(result.boxes) > 0:
            # Sort boxes by confidence descending and keep class order accordingly
            boxes = sorted(list(result.boxes), key=lambda b: float(b.conf.item()), reverse=True)
            classes = [int(b.cls.item()) for b in boxes]
        else:
            classes = []
        preds.append({'path': str(img_path), 'all_pred_classes': classes})
    return pd.DataFrame(preds)


def _save_confusion(cm, out_dir: Path, model_name: str, noise: float, int_to_class: Dict[int, str], tag: str) -> str:
    df = confusion_df(cm, int_to_class)
    suffix = f"{model_name}_noise{noise}_{tag}"
    path = os.path.join(out_dir, f"confusion_{suffix}.csv")
    df.to_csv(path)
    return path


def run_yolo_predictions_and_metrics(model, test_images_dir, labels_dir, label_list, output_dir, model_name, noise, conf_thresh=0.05):
    print(f"[YOLO] Running predictions on {test_images_dir}")
    test_images_dir = Path(test_images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    class_to_int: Dict[str, int] = label_list
    int_to_class: Dict[int, str] = {i: v for v, i in class_to_int.items()}
    noise_id = class_to_int.get('Noise', len(class_to_int) - 1)

    # 1) Ground truth table (single-label gt_primary + multi-label sets)
    gt_df = build_ground_truth_dataframe(test_images_dir, labels_dir, class_to_int)

    # 2) Predictions (ordered classes per image by confidence)
    preds_df = _collect_yolo_predictions(model, test_images_dir, conf_thresh)

    # 3) Merge on path and ensure alignment
    merged_df = pd.merge(gt_df, preds_df, on='path', how='left')
    merged_df['all_pred_classes'] = merged_df['all_pred_classes'].apply(lambda x: x if isinstance(x, list) else [])

    # 4) Collapse strategies
    y_top1 = predict_top1(merged_df, noise_id)
    y_presence = predict_presence(merged_df, merged_df['gt_primary'].tolist(), noise_id)
    y_strict = predict_strict_set_match(merged_df, merged_df['gt_set'].tolist(), merged_df['gt_primary'].tolist(), noise_id)

    strategies = {
        'top1': y_top1,
        'presence': y_presence,
        'strict': y_strict,
    }

    # 5) Compute confusion + metrics for each strategy (vs. single-label gt_primary)
    y_true = merged_df['gt_primary'].to_numpy()
    metrics_summary = {}
    for name, y_pred in strategies.items():
        cm = compute_confusion(y_true, pd.Series(y_pred).to_numpy(), n_classes=len(class_to_int))
        mets = compute_paper_metrics(cm, class_to_int)
        cm_path = _save_confusion(cm, output_dir, model_name, noise, int_to_class, tag=name)
        metrics_summary[name] = {**mets, 'confusion_path': cm_path}

    # 6) Preds debug
    write_preds_debug(output_dir, merged_df, strategies, int_to_class)
    write_summary(output_dir, metrics_summary)

    # Return the primary comparator's metrics (top1)
    return metrics_summary['top1'], metrics_summary['top1']['confusion_path']
