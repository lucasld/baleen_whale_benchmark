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




