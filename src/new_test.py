import argparse
import pathlib
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from dataset import SpectrogramDataSet
from training import select_more_noise
from model import Model


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

    # 2. Get predicted labels (argmax over class columns)
    # Find the columns that are the class probabilities
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

    # 5. Save confusion matrix
    cm_df = pd.DataFrame(confusion_matrix, columns=[int_to_class[i] for i in range(n_classes)], index=[int_to_class[i] for i in range(n_classes)])
    suffix = f"{model_name}_noise{noise}"
    cm_path = os.path.join(output_dir, f"confusion_{suffix}.csv")
    cm_df.to_csv(cm_path)
    print(f"[METRICS] Confusion matrix saved to {cm_path}")

    metrics = {"TCR": TCR, "NMR": NMR, "CMR": CMR, "F": F,
               "ACC": ACC, "FCR": FCR, "MACRO_RECALL": MACRO_RECALL}
    return metrics, cm_path
    

def evaluate_models(model_folder, ds, config, output_dir):
    """Evaluate all models in the specified folder on test sets with varying noise ratios.

    :param model_folder: Path to the folder containing trained models
    :type model_folder: pathlib.Path
    :param ds: Dataset object
    :type ds: SpectrogramDataSet
    :param config: Configuration dictionary
    :type config: dict
    :param output_dir: Directory to save evaluation results
    :type output_dir: pathlib.Path
    """
    noise_ratios = config['NOISE_RATIO_TEST']
    if not isinstance(noise_ratios, list):
        noise_ratios = [noise_ratios]
    label_list_path = model_folder / "labels.json"
    with open(label_list_path) as f:
        label_list = json.load(f)
    metrics_records = []  # To store metrics for each fold and noise
    for model_subfolder in sorted(model_folder.iterdir()):
        if not model_subfolder.is_dir():
            continue
        model_name = model_subfolder.name
        model_dir = model_subfolder / "model"
        model_file = model_dir / "saved_model.pb"
        if not model_dir.exists() or not model_file.exists():
            print(f"  Skipping {model_name}: no model found at {model_file}")
            continue
        print(f"\nEvaluating model: {model_name}")
        cnn_model = Model(save_path=model_subfolder.parent, categories=ds.categories, model_name=model_name)
        cnn_model.load_existing()
        # Load the true data_used file for this fold/model
        data_used_path = model_subfolder / f"data_used_{model_name}.csv"
        if not data_used_path.exists():
            print(f"  Skipping {model_name}: no data_used file found at {data_used_path}")
            continue
        all_df = pd.read_csv(data_used_path)
        base_test_df = all_df[all_df['set'] == 'test'].copy()
        last_noise = 0  #TODO: changed from noise_ratios[0]
        for noise in noise_ratios:
            print(f"  Noise ratio: {noise}")
            test_df, _ = select_more_noise(base_test_df.copy(), 'test', last_noise, noise, config, ds)
            last_noise = noise
            ds.print_sample_counts(test_df, partition_name="test")
            scores_df, con_mat_df, preds_df = cnn_model.new_test(ds, test_df)
            suffix = f"{model_name}_noise{noise}"
            os.makedirs(output_dir, exist_ok=True)
            preds_path = pathlib.Path(output_dir) / f"predictions_{suffix}.csv"
            dataused_path = pathlib.Path(output_dir) / f"dataused_{suffix}.csv"
            preds_df.to_csv(preds_path, index=False)
            test_df.to_csv(dataused_path, index=False)
            print("Label list:", label_list)
            metrics, cm_path = compute_metrics_and_confusion(preds_df, label_list, output_dir, model_name, noise)
            print(f"  Metrics: {metrics}")
            print(f"  Confusion matrix saved to: {cm_path}")
            # Store metrics for later aggregation
            metrics_record = {'fold': model_name, 'noise': noise}
            metrics_record.update(metrics)
            metrics_records.append(metrics_record)
    # After all evaluations, aggregate metrics
    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        metrics_df.to_csv(os.path.join(output_dir, 'all_metrics_by_fold_and_noise.csv'), index=False)
        print(f"[SUMMARY] Saved all metrics to {os.path.join(output_dir, 'all_metrics_by_fold_and_noise.csv')}")
        # Compute mean and std by noise
        summary = metrics_df.groupby('noise').agg(['mean', 'std'])
        summary.to_csv(os.path.join(output_dir, 'metrics_summary_by_noise.csv'))
        print(f"[SUMMARY] Saved summary metrics to {os.path.join(output_dir, 'metrics_summary_by_noise.csv')}")
        print("[SUMMARY] Example summary:\n", summary.head())
        # create side-by-side boxplots for ACC and FCR across noise levels
        try:
            plot_path = os.path.join(output_dir, 'metrics_boxplots.png')
            plot_metrics_boxplots(os.path.join(output_dir, 'all_metrics_by_fold_and_noise.csv'), plot_path)
        except Exception as e:
            print(f"[PLOT] Skipped plotting due to error: {e}")
    else:
        print("[SUMMARY] No metrics were collected.")


# Insert plotting function after evaluate_models, before main
def plot_metrics_boxplots(metrics_csv_path, output_path=None):
    """
    Plot boxplots of overall Accuracy (ACC) and False Classification Rate (FCR) vs. noise percentage.

    :param metrics_csv_path: path to all_metrics_by_fold_and_noise.csv
    :type metrics_csv_path: str
    :param output_path: if provided, saves the figure to this path; otherwise shows it
    :type output_path: str, optional
    :return: None
    :rtype: None
    """
    df = pd.read_csv(metrics_csv_path)
    # ensure noise is numeric and sorted
    if 'noise' not in df.columns:
        raise ValueError("'noise' column not found in metrics CSV")
    df['noise'] = df['noise'].astype(float)
    noise_levels = sorted(df['noise'].unique().tolist())

    # derive ACC and FCR columns
    if 'ACC' not in df.columns:
        raise ValueError("'ACC' column not found in metrics CSV; rerun evaluation to compute ACC")
    if 'FCR' not in df.columns:
        df['FCR'] = 1.0 - df['ACC']

    # collect data for each noise level
    acc_data = [df.loc[df['noise'] == n, 'ACC'].dropna().values for n in noise_levels]
    fcr_data = [df.loc[df['noise'] == n, 'FCR'].dropna().values for n in noise_levels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # left: ACC
    b1 = axes[0].boxplot(acc_data, patch_artist=True, widths=0.6, showfliers=False, whis=[0, 100])
    for patch in b1['boxes']:
        patch.set(facecolor='#b6eab6', edgecolor='#228B22', linewidth=2)
    for k in ('whiskers', 'caps', 'medians'):
        for artist in b1[k]:
            artist.set(color='#228B22', linewidth=2)
    axes[0].set_xticks(np.arange(1, len(noise_levels) + 1))
    axes[0].set_xticklabels([f"{n:.1f}" for n in noise_levels])
    axes[0].set_xlabel("Noise percentage in test dataset")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.70, 0.95)

    # right: FCR
    b2 = axes[1].boxplot(fcr_data, patch_artist=True, widths=0.6, showfliers=False, whis=[0, 100])
    for patch in b2['boxes']:
        patch.set(facecolor='#b6eaf7', edgecolor='#0077b6', linewidth=2)
    for k in ('whiskers', 'caps', 'medians'):
        for artist in b2[k]:
            artist.set(color='#0077b6', linewidth=2)
    axes[1].set_xticks(np.arange(1, len(noise_levels) + 1))
    axes[1].set_xticklabels([f"{n:.1f}" for n in noise_levels])
    axes[1].set_xlabel("Noise percentage in test dataset")
    axes[1].set_ylabel("False classification rate")
    axes[1].set_ylim(0.05, 0.30)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"[PLOT] Saved plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test sets with varying noise ratios.")
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    model_folder = pathlib.Path(args.model_folder)
    output_dir = pathlib.Path(args.output_dir)
    with open(args.config) as f:
        config = json.load(f)
    ds = SpectrogramDataSet(
        data_dir=config['DATA_DIR'],
        categories=config['CATEGORIES'],
        join_cat=config['CATEGORIES_TO_JOIN'],
        locations=config['LOCATIONS'],
        corrected=config['USE_CORRECTED_DATASET'],
        samples_per_class=config['SAMPLES_PER_CLASS']
    )
    evaluate_models(model_folder, ds, config, output_dir)

if __name__ == "__main__":
    main()