from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_confusion(true_labels: np.ndarray, pred_labels: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1
    return cm


def compute_paper_metrics(cm: np.ndarray, class_to_int: Dict[str, int]) -> Dict[str, float]:
    n_classes = cm.shape[0]
    noise_idx = class_to_int['Noise'] if 'Noise' in class_to_int else n_classes - 1
    non_noise = [i for i in range(n_classes) if i != noise_idx]

    # TCR: mean recall across non-noise classes
    tcrs = []
    for i in non_noise:
        total = cm[i, :].sum()
        correct = cm[i, i]
        tcrs.append((correct / total) if total > 0 else 0.0)
    TCR = float(np.mean(tcrs)) if tcrs else 0.0

    # NMR: fraction of noise samples predicted as non-noise
    noise_total = cm[noise_idx, :].sum()
    noise_mis = cm[noise_idx, :].sum() - cm[noise_idx, noise_idx]
    NMR = (noise_mis / noise_total) if noise_total > 0 else 0.0

    # CMR: mean fraction of non-noise samples misclassified as other non-noise
    cmrs = []
    for i in non_noise:
        total = cm[i, :].sum()
        mis = sum(cm[i, j] for j in non_noise if j != i)
        cmrs.append((mis / total) if total > 0 else 0.0)
    CMR = float(np.mean(cmrs)) if cmrs else 0.0

    F = float(np.mean([TCR, 1 - NMR, 1 - NMR, 1 - CMR]))

    # Additional classifier-style metrics for context
    total = cm.sum()
    ACC = (np.trace(cm) / total) if total > 0 else 0.0
    FCR = 1.0 - ACC
    return {"TCR": TCR, "NMR": NMR, "CMR": CMR, "F": F, "ACC": ACC, "FCR": FCR}


def confusion_df(cm: np.ndarray, int_to_class: Dict[int, str]) -> pd.DataFrame:
    labels = [int_to_class[i] for i in range(len(int_to_class))]
    return pd.DataFrame(cm, columns=labels, index=labels)


