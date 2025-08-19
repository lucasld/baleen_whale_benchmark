#!/usr/bin/env python3
"""
Plot confusion matrices from CSV files produced by src/evaluation.py.

Usage:
  python src/utils/plot_confusions_for_run.py --eval_dir <EVAL_DIR> [--label_list <LABELS_JSON>] [--include_totals]

Defaults:
  - If --eval_dir ends with 'evaluation', the script looks for labels.json one level up
  - Otherwise, provide --label_list

Notes:
  - The CSVs produced by evaluation.py contain an extra last row and last column with totals.
    By default, we drop that summary row/col when plotting. Use --include_totals to keep them.
"""

import argparse
import json
import os
from typing import List

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for batch environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_label_names(labels_json_path: str) -> List[str]:
    with open(labels_json_path, 'r') as f:
        mapping = json.load(f)  # expects {class_name: index}
    index_to_name = {index: name for name, index in mapping.items()}
    names = [index_to_name[i] for i in range(len(index_to_name))]
    return names


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, output_png: str) -> None:
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot confusion matrices from evaluation CSVs.')
    parser.add_argument('--eval_dir', required=True, help='Path to the evaluation folder containing *_confusion_*.csv files')
    parser.add_argument('--label_list', required=False, help='Path to labels.json; defaults to parent of eval_dir if its name is evaluation')
    parser.add_argument('--include_totals', action='store_true', help='Include the last row/column of totals when plotting')
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    if not os.path.isdir(eval_dir):
        raise SystemExit(f"Evaluation directory not found: {eval_dir}")

    # Determine labels.json path
    if args.label_list:
        labels_json = os.path.abspath(args.label_list)
    else:
        parent = os.path.dirname(eval_dir)
        if os.path.basename(eval_dir).lower() == 'evaluation' and os.path.isfile(os.path.join(parent, 'labels.json')):
            labels_json = os.path.join(parent, 'labels.json')
        else:
            raise SystemExit('labels.json not provided and could not infer it from the parent run folder.')

    class_names = load_label_names(labels_json)

    # Find confusion CSVs
    csvs = [f for f in os.listdir(eval_dir) if f.endswith('.csv') and '_confusion_' in f]
    if not csvs:
        raise SystemExit(f"No confusion CSVs found in: {eval_dir}")

    for csv_file in sorted(csvs):
        csv_path = os.path.join(eval_dir, csv_file)
        df = pd.read_csv(csv_path, index_col=0)
        cm = df.to_numpy()

        # Drop last row/col (totals) by default
        if not args.include_totals:
            if cm.shape[0] >= 2 and cm.shape[1] >= 2:
                cm = cm[:-1, :-1]

        # Align labels to matrix size; when dropping totals the size should equal number of classes
        if cm.shape[0] == len(class_names):
            labels_for_plot = class_names
        elif cm.shape[0] == len(class_names) + 1 and args.include_totals:
            labels_for_plot = class_names + ['Total']
        else:
            # Fallback to index labels
            labels_for_plot = [str(i) for i in range(cm.shape[0])]

        title = os.path.splitext(csv_file)[0]
        png_path = os.path.join(eval_dir, os.path.splitext(csv_file)[0] + '.png')
        plot_confusion_matrix(cm, labels_for_plot, title, png_path)
        print(f"Saved: {png_path}")

    print('All plots saved.')


if __name__ == '__main__':
    main()


