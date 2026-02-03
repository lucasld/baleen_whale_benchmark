"""
Refactored YOLO evaluation orchestrator.
Runs predictions, builds ground truth, applies Top-1 collapse strategy,
then computes classifier-style metrics and writes comparable confusion matrices and a preds_debug.csv.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO

from .evaluation.ground_truth import build_ground_truth_dataframe
from .evaluation.strategies import predict_top1
from .evaluation.metrics import compute_confusion, compute_paper_metrics, confusion_df
from .evaluation.reporting import write_preds_debug, write_summary, plot_tcr_vs_nmr_curves, write_raw_predictions


def _collect_yolo_predictions(model: YOLO, test_images_dir: Path, conf_thresh: float, imgsz: int = 640, device: str = None) -> pd.DataFrame:
    """
    Runs inference on a directory and collects raw bounding box data.
    Explicitly handles CPU movement and type conversion to avoid silent failures.
    """
    # Ensure conf is a float
    conf_thresh = float(conf_thresh)
    
    print(f"[YOLO] Collecting predictions... (imgsz={imgsz}, conf={conf_thresh}, device={device})")

    # Run inference
    results = model.predict(
        source=str(test_images_dir),
        conf=conf_thresh,
        imgsz=imgsz,
        device=device,
        save=False,
        verbose=False,
        workers=4,
        stream=True
    )
    
    preds = []
    for result in results:
        path = str(Path(result.path).resolve())
        
        # Extract boxes
        boxes = result.boxes
        
        classes = []
        confidences = []
        
        if boxes is not None:
            # Check if we have any detections
            if boxes.cls is not None and len(boxes.cls) > 0:
                # Move to CPU, convert to numpy, then to list
                # This chain is the safest way to extract values from Ultralytics tensors
                try:
                    cls_list = boxes.cls.cpu().numpy().tolist()
                    conf_list = boxes.conf.cpu().numpy().tolist()
                    
                    # Ensure correct types
                    classes = [int(c) for c in cls_list]
                    confidences = [float(c) for c in conf_list]
                except Exception as e:
                    print(f"[YOLO] Warning: Error extracting boxes for {path}: {e}")
                    classes = []
                    confidences = []

        preds.append({
            'path': str(path), 
            'all_pred_classes': classes, 
            'all_pred_confidences': confidences
        })
        
    return pd.DataFrame(preds)


def _save_confusion(cm, out_dir: Path, model_name: str, noise: float, int_to_class: Dict[int, str], tag: str) -> str:
    df = confusion_df(cm, int_to_class)
    suffix = f"{model_name}_noise{noise}_{tag}"
    path = os.path.join(out_dir, f"confusion_{suffix}.csv")
    df.to_csv(path)
    return path


def run_yolo_predictions_and_metrics(
    model,
    test_images_dir,
    labels_dir,
    label_list,
    output_dir,
    model_name,
    noise,
    conf_thresh: float = 0.05,
    conf_thresholds: Optional[List[float]] = None,
    imgsz: int = 640,
    device: str = None,
):
    print(f"[YOLO] Running predictions on {test_images_dir} (imgsz={imgsz})")
    test_images_dir = Path(test_images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    class_to_int: Dict[str, int] = label_list
    int_to_class: Dict[int, str] = {i: v for v, i in class_to_int.items()}
    noise_id = class_to_int.get('Noise', len(class_to_int) - 1)

    # Build list of thresholds (ensure baseline conf_thresh included)
    thresholds: List[float] = []
    if conf_thresholds is not None:
        thresholds.extend(conf_thresholds)
    thresholds.append(conf_thresh)
    thresholds = sorted({round(float(t), 6) for t in thresholds if t is not None and 0.0 <= t <= 1.0})
    if not thresholds:
        thresholds = [conf_thresh]
    collect_conf = max(min(thresholds), 0.0)

    # 1) Ground truth table (single-label gt_primary + multi-label sets)
    gt_df = build_ground_truth_dataframe(test_images_dir, labels_dir, class_to_int)

    # 2) Predictions (ordered classes + confidences per image)
    preds_df = _collect_yolo_predictions(model, test_images_dir, collect_conf, imgsz=imgsz, device=device)

    # 3) Merge on path and ensure alignment
    merged_df = pd.merge(gt_df, preds_df, on='path', how='left')
    merged_df['all_pred_classes'] = merged_df['all_pred_classes'].apply(lambda x: x if isinstance(x, list) else [])
    merged_df['all_pred_confidences'] = merged_df['all_pred_confidences'].apply(lambda x: x if isinstance(x, list) else [])

    # Save raw predictions (unfiltered) for safety/re-analysis
    write_raw_predictions(output_dir, merged_df, int_to_class)

    y_true = merged_df['gt_primary'].to_numpy()

    print(f"[YOLO] Evaluating {len(thresholds)} confidence thresholds for 1 strategy on {len(y_true)} samples")

    metrics_by_threshold: Dict[float, Dict[str, Dict[str, float]]] = {}
    metrics_rows: List[Dict[str, float]] = []
    baseline_threshold = conf_thresh if conf_thresh in thresholds else thresholds[0]
    baseline_strategy_preds: Optional[Dict[str, List[int]]] = None
    baseline_filtered_classes: Optional[List[List[int]]] = None

    for i, thr in enumerate(thresholds):
        if i % 5 == 0 or i == len(thresholds) - 1:  # Log progress every 5 thresholds or at the end
            print(f"[YOLO] Processing confidence threshold {thr:.3f} ({i+1}/{len(thresholds)})")

        filtered_classes: List[List[int]] = []
        for classes, confidences in zip(merged_df['all_pred_classes'], merged_df['all_pred_confidences']):
            kept = [cls for cls, conf in zip(classes, confidences) if conf >= thr]
            filtered_classes.append(kept)

        preds_df_thr = pd.DataFrame({'all_pred_classes': filtered_classes})

        y_top1 = predict_top1(preds_df_thr, noise_id)

        strategies = {
            'top1': y_top1,
        }

        strategy_outputs: Dict[str, Dict[str, float]] = {}
        for name, y_pred in strategies.items():
            cm = compute_confusion(y_true, np.asarray(y_pred), n_classes=len(class_to_int))
            mets = compute_paper_metrics(cm, class_to_int)
            cm_path = _save_confusion(cm, output_dir, model_name, noise, int_to_class, tag=f"{name}_conf{thr:.2f}")
            record = {**mets, 'confusion_path': cm_path, 'threshold': thr}
            strategy_outputs[name] = record
            metrics_rows.append({'threshold': thr, 'strategy': name, **mets, 'confusion_path': cm_path})

        metrics_by_threshold[thr] = strategy_outputs

        if abs(thr - baseline_threshold) < 1e-6:
            baseline_strategy_preds = {k: list(v) for k, v in strategies.items()}
            baseline_filtered_classes = [list(cls_list) for cls_list in filtered_classes]

    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(['threshold', 'strategy'])
        sweep_csv = output_dir / f"{model_name}_metrics_confidence_sweep.csv"
        metrics_df.to_csv(sweep_csv, index=False)
        # Auto-plot curves per strategy and combined
        try:
            plot_tcr_vs_nmr_curves(metrics_df, output_dir)
        except Exception:
            pass
        try:
            from .evaluation.reporting import plot_f_vs_confidence_curves  # type: ignore
            plot_f_vs_confidence_curves(metrics_df, output_dir)
        except Exception:
            pass

    # Prepare baseline outputs for summary/debug (fall back to first threshold if needed)
    if baseline_strategy_preds is None or baseline_filtered_classes is None:
        first_thr = thresholds[0]
        baseline_filtered_classes = [
            [cls for cls, conf in zip(classes, confidences) if conf >= first_thr]
            for classes, confidences in zip(merged_df['all_pred_classes'], merged_df['all_pred_confidences'])
        ]
        preds_df_first = pd.DataFrame({'all_pred_classes': baseline_filtered_classes})
        baseline_strategy_preds = {
            'top1': predict_top1(preds_df_first, noise_id),
        }
        baseline_threshold = first_thr

    # Write debug artifacts for the baseline threshold
    merged_for_debug = merged_df.copy()
    merged_for_debug['all_pred_classes'] = baseline_filtered_classes
    merged_for_debug['all_pred_confidences'] = [
        [conf for conf in confidences if conf >= baseline_threshold]
        for confidences in merged_df['all_pred_confidences']
    ]
    write_preds_debug(output_dir, merged_for_debug, baseline_strategy_preds, int_to_class)

    baseline_summary = metrics_by_threshold[baseline_threshold]
    write_summary(output_dir, baseline_summary)

    baseline_metrics = baseline_summary['top1']
    print(f"[YOLO] Evaluation complete. Baseline Top-1 metrics at conf={baseline_threshold:.3f}:")
    print(f"       TCR={baseline_metrics['TCR']:.3f}, NMR={baseline_metrics['NMR']:.3f}, CMR={baseline_metrics['CMR']:.3f}, F={baseline_metrics['F']:.3f}")
    return baseline_metrics, baseline_metrics['confusion_path'], metrics_df
