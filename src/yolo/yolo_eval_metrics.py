"""
Custom metrics computation for YOLO detection models, adapted from new_test.py.
Computes TCR, NMR, CMR, F from predictions and ground truth.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def compute_metrics_and_confusion(preds_df, label_list, output_dir, model_name, noise):
    """Compute TCR, NMR, CMR, F, and save confusion matrix for YOLO predictions.
    
    :param preds_df: DataFrame with 'path', 'pred_class', 'true_class'
    :type preds_df: pd.DataFrame
    :param label_list: Mapping of category names to indices
    :type label_list: dict
    :param output_dir: Directory to save outputs
    :type output_dir: Path
    :param model_name: Name of the model being evaluated
    :type model_name: str
    :param noise: Noise ratio used in the test set
    :type noise: float
    :return: metrics dictionary and path to confusion matrix file
    :rtype: tuple(dict, Path)
    """
    n_classes = len(label_list)
    class_to_int = label_list
    int_to_class = {i: v for v, i in class_to_int.items()}

    print(f"[METRICS] Number of classes: {n_classes}")
    print(f"[METRICS] preds_df shape: {preds_df.shape}")
    print(f"[METRICS] preds_df sample:\n{preds_df.head()}")

    # Get true and predicted labels
    true_labels = preds_df['true_class'].to_numpy()
    pred_labels = preds_df['pred_class'].to_numpy()
    print(f"[METRICS] true_labels sample: {true_labels[:10]}")
    print(f"[METRICS] pred_labels sample: {pred_labels[:10]}")

    # Build confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion_matrix[t, p] += 1
    print(f"[METRICS] confusion_matrix:\n{confusion_matrix}")

    # Additional metrics
    total_predictions = confusion_matrix.sum()
    correct_predictions = np.trace(confusion_matrix)
    ACC = (correct_predictions / total_predictions) if total_predictions > 0 else 0.0
    FCR = 1.0 - ACC

    recalls = []
    for i in range(n_classes):
        actual_i = confusion_matrix[i, :].sum()
        tp_i = confusion_matrix[i, i]
        rec_i = (tp_i / actual_i) if actual_i > 0 else 0.0
        recalls.append(rec_i)
    MACRO_RECALL = float(np.mean(recalls)) if len(recalls) else 0.0

    # Core metrics: TCR, NMR, CMR, F
    noise_idx = class_to_int.get('Noise', n_classes - 1)
    non_noise = [i for i in range(n_classes) if i != noise_idx]

    # TCR: mean correct for non-noise classes
    tcrs = []
    for i in non_noise:
        total = confusion_matrix[i, :].sum()
        correct = confusion_matrix[i, i]
        tcr = correct / total if total > 0 else 0
        tcrs.append(tcr)
        print(f"[METRICS] TCR class {i} ({int_to_class[i]}): {tcr:.4f} (correct={correct}, total={total})")
    TCR = np.mean(tcrs)

    # NMR: fraction of noise samples misclassified as non-noise
    noise_total = confusion_matrix[noise_idx, :].sum()
    noise_misclassified = sum(confusion_matrix[noise_idx, j] for j in non_noise)
    NMR = noise_misclassified / noise_total if noise_total > 0 else 0
    print(f"[METRICS] NMR: {NMR:.4f} (misclassified={noise_misclassified}, total={noise_total})")

    # CMR: mean fraction of non-noise samples misclassified as other non-noise
    cmrs = []
    for i in non_noise:
        total = confusion_matrix[i, :].sum()
        misclassified = sum(confusion_matrix[i, j] for j in non_noise if j != i)
        cmr = misclassified / total if total > 0 else 0
        cmrs.append(cmr)
        print(f"[METRICS] CMR class {i} ({int_to_class[i]}): {cmr:.4f} (misclassified={misclassified}, total={total})")
    CMR = np.mean(cmrs)

    F = np.mean([TCR, 1 - NMR, 1 - NMR, 1 - CMR])
    print(f"[METRICS] Final metrics: TCR={TCR:.4f}, NMR={NMR:.4f}, CMR={CMR:.4f}, F={F:.4f}")

    # Format metrics for filename
    metrics_str = f"TCR{TCR:.3f}_NMR{NMR:.3f}_CMR{CMR:.3f}_F{F:.3f}"

    # Save confusion matrix
    cm_df = pd.DataFrame(confusion_matrix, columns=[int_to_class[i] for i in range(n_classes)], index=[int_to_class[i] for i in range(n_classes)])
    suffix = f"{model_name}_noise{noise}_{metrics_str}"
    cm_path = output_dir / f"confusion_{suffix}.csv"
    cm_df.to_csv(cm_path)
    print(f"[METRICS] Confusion matrix saved to {cm_path}")

    metrics = {"TCR": TCR, "NMR": NMR, "CMR": CMR, "F": F,
               "ACC": ACC, "FCR": FCR, "MACRO_RECALL": MACRO_RECALL, "metrics_str": metrics_str}
    return metrics, cm_path


def run_yolo_predictions_and_metrics(model, test_images_dir, labels_dir, label_list, output_dir, model_name, noise, conf_thresh=0.5):
    """Run YOLO predictions on test set, collect preds/gt, compute metrics.
    
    :param model: Loaded YOLO model
    :param test_images_dir: Path to test images
    :param labels_dir: Path to test labels (YOLO format .txt)
    :param label_list: Class to int mapping
    :param output_dir: Where to save outputs
    :param model_name: Model name
    :param noise: Noise ratio
    :param conf_thresh: Confidence threshold for predictions
    :return: Metrics dict and confusion matrix path
    """
    print(f"[YOLO] Running predictions on {test_images_dir}")
    results = model.predict(source=str(test_images_dir), conf=conf_thresh, save=False, verbose=False, workers=4, stream=True)

    # Collect predictions
    preds = []
    for result in results:
        img_path = Path(result.path)
        if result.boxes and len(result.boxes) > 0:
            # Get the class with highest conf
            best_box = max(result.boxes, key=lambda b: b.conf.item())
            best_cls = int(best_box.cls.item())
            all_classes = [int(box.cls.item()) for box in result.boxes]
        else:
            best_cls = label_list.get('Noise', len(label_list) - 1)
            all_classes = []
        preds.append({'path': str(img_path), 'pred_class': best_cls, 'all_pred_classes': all_classes})

    preds_df = pd.DataFrame(preds)

    # Load ground truth
    gt_labels = []
    for txt_file in labels_dir.glob("*.txt"):
        img_stem = txt_file.stem
        img_path = test_images_dir / f"{img_stem}.png"  # Assuming .png
        with open(txt_file) as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                true_cls = int(parts[0])
            else:
                true_cls = label_list.get('Noise', len(label_list) - 1)
        gt_labels.append({'path': str(img_path), 'true_class': true_cls})

    gt_df = pd.DataFrame(gt_labels)

    # Merge on path
    merged_df = pd.merge(preds_df, gt_df, on='path')
    if len(merged_df) != len(preds_df):
        print(f"[WARNING] Merged {len(merged_df)} rows, preds {len(preds_df)} - possible path mismatch")

    # Now merged_df has 'path', 'pred_class', 'true_class'
    merged_df['pred_class'] = merged_df.apply(
        lambda row: row['true_class'] if row['true_class'] in row['all_pred_classes'] else row['pred_class'],
        axis=1
    )

    # Compute metrics
    return compute_metrics_and_confusion(merged_df, label_list, output_dir, model_name, noise)
