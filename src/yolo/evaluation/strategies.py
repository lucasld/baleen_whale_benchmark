from typing import List, Dict
import pandas as pd


NOISE_KEY = 'Noise'


def predict_top1(preds_df: pd.DataFrame, noise_id: int) -> List[int]:
    """
    Single-label prediction per image using the class of the highest-confidence box if any; else Noise.
    Assumes preds_df contains columns: 'path', 'all_pred_classes' (List[int]) and optionally confidences in future.
    """
    pred: List[int] = []
    # We do not yet track confidences per class list; use first class if present (caller should supply best order)
    for classes in preds_df['all_pred_classes']:
        if classes and len(classes) > 0:
            pred.append(classes[0])
        else:
            pred.append(noise_id)
    return pred


def predict_presence(preds_df: pd.DataFrame, gt_primary: List[int], noise_id: int) -> List[int]:
    """
    Presence-based rule: if gt_primary is among predicted classes, predict gt_primary; else Noise.
    Note: This is lenient and intended for analysis, not headline metrics.
    """
    out: List[int] = []
    for classes, gt in zip(preds_df['all_pred_classes'], gt_primary):
        if gt in set(classes):
            out.append(gt)
        else:
            out.append(noise_id)
    return out


def predict_strict_set_match(preds_df: pd.DataFrame, gt_sets: List[set], gt_primary: List[int], noise_id: int) -> List[int]:
    """
    Very strict rule: predict gt_primary only if predicted set equals gt_set (ignoring duplicates).
    Else predict Noise. Intended as an analysis bound.
    """
    out: List[int] = []
    for classes, gt_set, gt in zip(preds_df['all_pred_classes'], gt_sets, gt_primary):
        if set(classes) == set(gt_set):
            out.append(gt)
        else:
            out.append(noise_id)
    return out


