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
    """Compute TCR, NMR, CMR, F, and save confusion matrix, following evaluation.py logic.
    
    :param preds_df: DataFrame with model predictions
    :type preds_df: pd.DataFrame
    :param label_list: Mapping of category names to indices
    :type label_list: dict
    :param output_dir: Directory to save outputs
    :type output_dir: pathlib.Path
    :param model_name: Name of the model being evaluated
    :type model_name: str
    :param noise: Noise ratio used in the test set
    :type noise: float
    :return: metrics dictionary and path to confusion matrix file
    :rtype: tuple(dict, pathlib.Path)
    """
    # print all arguments for debugging
    print(f"[DEBUG] preds_df head: {preds_df.head()}")
    print(f"[DEBUG] label_list: {label_list}")
    # Align on 'path'
    # DEBUG: Print columns of preds_df and merged_df to diagnose KeyError
    print("[DEBUG] preds_df columns:", preds_df.columns.tolist())
    # assign true labels based on path
    join_cat = {
        "20Plus": "20Hz20Plus",
        "20Hz": "20Hz20Plus",
        "A": "ABZ",
        "B": "ABZ",
        "Z": "ABZ",
        "D": "DDswp",
        "Dswp": "DDswp",
        "Noise": "Noise"
    }
    class_to_int = label_list
    int_to_class = {i: v for v, i in class_to_int.items()}
    print("[DEBUG] class_to_int:", class_to_int)
    print("[DEBUG] int_to_class:", int_to_class)
    def get_label_from_path(path):
        # e.g. 1427_BallenyIslands2015_20Hz.png -> 20Hz
        label_str = os.path.basename(path).split('_')[2].split('.')[0]
        # e.g. 20Hz -> 20Hz20Plus
        joined_label = None
        for jl, ol in join_cat.items():
            if label_str == jl:
                joined_label = ol
                break
        if joined_label is None:
            # It might be that the label is already a joined label
            if label_str in class_to_int:
                joined_label = label_str
            else:
                raise ValueError(f"Cannot find a joined label for {label_str}")
        return class_to_int[joined_label]
    preds_df['true_label'] = preds_df['path'].apply(get_label_from_path)
    print("[DEBUG] preds_df with true_label head:\n", preds_df.head())

    n_classes = len(label_list)
    class_to_int = label_list
    int_to_class = {i: v for v, i in class_to_int.items()}

    print(f"[METRICS] Number of classes: {n_classes}")
    print(f"[METRICS] class_to_int: {class_to_int}")
    print(f"[METRICS] int_to_class: {int_to_class}")
    print(f"[METRICS] preds_df columns: {preds_df.columns.tolist()}")
    print(f"[METRICS] preds_df shape: {preds_df.shape}")
    print(f"[METRICS] preds_df sample:\n{preds_df.head()}")

    # 1. Get true labels (already in preds_df['true_label'])
    true_labels = preds_df['true_label'].to_numpy()
    print(f"[METRICS] true_labels sample: {true_labels[:10]}")

    # 2. Get predicted labels
    if 'pred_class' in preds_df.columns:
        # For YOLO: use the pre-computed pred_class
        pred_labels = preds_df['pred_class'].to_numpy()
        print(f"[METRICS] Using pre-computed pred_labels from 'pred_class' column")
    else:
        # For CNN: argmax over class probability columns
        label_columns = [col for col in preds_df.columns if (isinstance(col, int) or (isinstance(col, str) and col.isdigit()))]
        label_columns = sorted(label_columns, key=lambda x: int(x))
        print(f"[METRICS] label_columns: {label_columns}")
        pred_probs = preds_df[label_columns].to_numpy()
        pred_labels = np.argmax(pred_probs, axis=1)
    print(f"[METRICS] pred_labels sample: {pred_labels[:10]}")

    # 3. Build confusion matrix (n_classes x n_classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion_matrix[t, p] += 1
    print(f"[METRICS] confusion_matrix:\n{confusion_matrix}")

    # ----- additional core metrics matching paper caption -----
    total_predictions = confusion_matrix.sum()
    correct_predictions = np.trace(confusion_matrix)
    ACC = (correct_predictions / total_predictions) if total_predictions > 0 else 0.0
    FCR = 1.0 - ACC

    # macro recall over all classes (including Noise)
    recalls = []
    for i in range(n_classes):
        actual_i = confusion_matrix[i, :].sum()  # TP_i + FN_i
        tp_i = confusion_matrix[i, i]
        rec_i = (tp_i / actual_i) if actual_i > 0 else 0.0
        recalls.append(rec_i)
    MACRO_RECALL = float(np.mean(recalls)) if len(recalls) else 0.0

    # 4. Compute metrics (TCR, NMR, CMR, F)
    # TCR: True Classification Rate (mean of diagonal except Noise)
    # NMR: Noise Misclassification Rate (sum of non-diagonal in Noise row / total Noise)
    # CMR: Confusion Misclassification Rate (mean of off-diagonal except Noise)
    # F: mean([TCR, 1-NMR, 1-NMR, 1-CMR])
    noise_idx = class_to_int['Noise'] if 'Noise' in class_to_int else n_classes-1
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
    noise_misclassified = confusion_matrix[noise_idx, :].sum() - confusion_matrix[noise_idx, noise_idx]
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

    F = np.mean([TCR, 1-NMR, 1-NMR, 1-CMR])
    print(f"[METRICS] Final metrics: TCR={TCR:.4f}, NMR={NMR:.4f}, CMR={CMR:.4f}, F={F:.4f}")

    # Format metrics for filename
    metrics_str = f"TCR{TCR:.3f}_NMR{NMR:.3f}_CMR{CMR:.3f}_F{F:.3f}"

    # 5. Save confusion matrix
    cm_df = pd.DataFrame(confusion_matrix, columns=[int_to_class[i] for i in range(n_classes)], index=[int_to_class[i] for i in range(n_classes)])
    suffix = f"{model_name}_noise{noise}_{metrics_str}"
    cm_path = os.path.join(output_dir, f"confusion_{suffix}.csv")
    cm_df.to_csv(cm_path)
    print(f"[METRICS] Confusion matrix saved to {cm_path}")

    metrics = {"TCR": TCR, "NMR": NMR, "CMR": CMR, "F": F,
               "ACC": ACC, "FCR": FCR, "MACRO_RECALL": MACRO_RECALL, "metrics_str": metrics_str}
    return metrics, cm_path


def run_yolo_predictions_and_metrics(model, test_images_dir, labels_dir, label_list, output_dir, model_name, noise, conf_thresh=0.2):
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
        lambda row: row['true_class'] if row['true_class'] in row['all_pred_classes'] else label_list.get('Noise', len(label_list) - 1),
        axis=1
    )

    # Add debug stats for over-prediction analysis
    noise_class_id = label_list['Noise']
    call_samples = merged_df[merged_df['true_class'] != noise_class_id]
    noise_samples = merged_df[merged_df['true_class'] == noise_class_id]

    print(f"[DEBUG] Total samples: {len(merged_df)}")
    print(f"[DEBUG] Call samples: {len(call_samples)}")
    if len(call_samples) > 0:
        avg_pred_len_calls = call_samples['all_pred_classes'].apply(len).mean()
        presence_rate_calls = (call_samples.apply(lambda row: row['true_class'] in row['all_pred_classes'], axis=1)).mean()
        print(f"[DEBUG] Avg all_pred_classes len for calls: {avg_pred_len_calls:.2f}")
        print(f"[DEBUG] Fraction of calls with true_class in all_pred_classes: {presence_rate_calls:.2f}")

    print(f"[DEBUG] Noise samples: {len(noise_samples)}")
    if len(noise_samples) > 0:
        avg_pred_len_noise = noise_samples['all_pred_classes'].apply(len).mean()
        noise_with_calls = (noise_samples['all_pred_classes'].apply(lambda x: any(c in [0, 1, 2] for c in x))).sum()
        print(f"[DEBUG] Avg all_pred_classes len for noise: {avg_pred_len_noise:.2f}")
        print(f"[DEBUG] Noise samples with call classes predicted: {noise_with_calls} ({noise_with_calls / len(noise_samples) * 100:.2f}%)")

    # Compute metrics
    return compute_metrics_and_confusion(merged_df, label_list, output_dir, model_name, noise)
