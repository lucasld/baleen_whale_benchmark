def compute_metrics_and_confusion(preds_df, label_list, output_dir, model_name, noise, include_noise_in_macro_recall=True):
    """
    Compute evaluation metrics and save the confusion matrix.

    The function computes:
      - ACC: overall accuracy = correct / total
      - FCR: false classification rate = 1 - ACC
      - MACRO_RECALL: average of per-class recalls TP_k / (TP_k + FN_k)
      - TCR, NMR, CMR, F: kept from your previous logic for backward compatibility

    :param preds_df: DataFrame with per-class probabilities and a 'path' column
    :type preds_df: pd.DataFrame
    :param label_list: mapping {class_name: int_index}
    :type label_list: dict
    :param output_dir: directory where artifacts are written
    :type output_dir: pathlib.Path
    :param model_name: name of the evaluated model (used in filenames)
    :type model_name: str
    :param noise: noise ratio in the test set (used in filenames)
    :type noise: float
    :param include_noise_in_macro_recall: include "Noise" class in MACRO_RECALL computation
    :type include_noise_in_macro_recall: bool
    :return: (metrics dict, path to saved confusion-matrix CSV)
    :rtype: tuple(dict, pathlib.Path)
    """
    # map original labels to joined labels used in training/eval
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

    # derive true labels from file path
    def get_label_from_path(path):
        # e.g., "..._BallenyIslands2015_20Hz.png" -> "20Hz"
        label_str = os.path.basename(path).split('_')[2].split('.')[0]
        # join if needed
        if label_str in join_cat:
            joined = join_cat[label_str]
        else:
            joined = label_str
        if joined not in class_to_int:
            raise ValueError(f"Cannot map label '{label_str}' (joined '{joined}') to class_to_int.")
        return class_to_int[joined]

    preds_df = preds_df.copy()
    preds_df['true_label'] = preds_df['path'].apply(get_label_from_path)

    n_classes = len(class_to_int)

    # find the class-probability columns (they are numeric strings or ints)
    label_columns = [c for c in preds_df.columns if (isinstance(c, int) or (isinstance(c, str) and c.isdigit()))]
    label_columns = sorted(label_columns, key=lambda x: int(x))

    # predicted labels = argmax over the class-probability columns
    pred_probs = preds_df[label_columns].to_numpy()
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = preds_df['true_label'].to_numpy()

    # build confusion matrix (rows = true, cols = pred)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion_matrix[t, p] += 1

    # ----- core metrics requested in the caption -----

    # overall accuracy
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix)
    ACC = correct / total if total > 0 else 0.0

    # false classification rate = 1 - ACC
    FCR = 1.0 - ACC

    # macro recall (average over classes of TP_k / actual_k)
    # optionally exclude the "Noise" class
    if 'Noise' in class_to_int and not include_noise_in_macro_recall:
        noise_idx = class_to_int['Noise']
        class_idxs = [i for i in range(n_classes) if i != noise_idx]
    else:
        class_idxs = list(range(n_classes))

    recalls = []
    for i in class_idxs:
        actual_i = confusion_matrix[i, :].sum()       # TP_i + FN_i
        tp_i = confusion_matrix[i, i]
        rec_i = tp_i / actual_i if actual_i > 0 else 0.0
        recalls.append(rec_i)
    MACRO_RECALL = float(np.mean(recalls)) if len(recalls) else 0.0

    # ----- keep your domain-specific metrics (unchanged) -----
    noise_idx = class_to_int['Noise'] if 'Noise' in class_to_int else (n_classes - 1)
    non_noise = [i for i in range(n_classes) if i != noise_idx]

    # tcr: mean correct for non-noise classes
    tcrs = []
    for i in non_noise:
        total_i = confusion_matrix[i, :].sum()
        correct_i = confusion_matrix[i, i]
        tcrs.append(correct_i / total_i if total_i > 0 else 0.0)
    TCR = float(np.mean(tcrs)) if len(tcrs) else 0.0

    # nmr: fraction of noise samples misclassified as non-noise
    noise_total = confusion_matrix[noise_idx, :].sum()
    noise_misclassified = noise_total - confusion_matrix[noise_idx, noise_idx]
    NMR = (noise_misclassified / noise_total) if noise_total > 0 else 0.0

    # cmr: mean fraction of non-noise samples misclassified as other non-noise
    cmrs = []
    for i in non_noise:
        total_i = confusion_matrix[i, :].sum()
        mis_i = sum(confusion_matrix[i, j] for j in non_noise if j != i)
        cmrs.append(mis_i / total_i if total_i > 0 else 0.0)
    CMR = float(np.mean(cmrs)) if len(cmrs) else 0.0

    # F from your prior code
    F = float(np.mean([TCR, 1 - NMR, 1 - NMR, 1 - CMR]))

    # save confusion matrix
    int_to_class = {i: v for v, i in class_to_int.items()}
    cm_df = pd.DataFrame(confusion_matrix,
                         columns=[int_to_class[i] for i in range(n_classes)],
                         index=[int_to_class[i] for i in range(n_classes)])
    suffix = f"{model_name}_noise{noise}"
    cm_path = os.path.join(output_dir, f"confusion_{suffix}.csv")
    cm_df.to_csv(cm_path, index=True)

    metrics = {
        "ACC": ACC,
        "FCR": FCR,
        "MACRO_RECALL": MACRO_RECALL,
        # legacy metrics kept:
        "TCR": TCR,
        "NMR": NMR,
        "CMR": CMR,
        "F": F
    }
    return metrics, cm_path