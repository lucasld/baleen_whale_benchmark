#!/usr/bin/env python3
"""Aggregate per-fold YOLO evaluation outputs into summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

SUMMARY_FILENAME = "selected_threshold_summary.json"


def find_folds(run_dir: Path) -> List[Path]:
    folds = sorted(p for p in run_dir.glob("fold_*") if p.is_dir())
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found under {run_dir}")
    return folds


def load_fold_summary(fold_dir: Path, run_name: str) -> Dict:
    summary_path = fold_dir / "yolo" / "runs_det" / run_name / SUMMARY_FILENAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    data = json.loads(summary_path.read_text())
    data["fold"] = fold_dir.name
    return data


def flatten_row(data: Dict) -> Dict:
    row = {
        "fold": data["fold"],
        "selected_threshold": data["selected_threshold"],
        "test_best_threshold": data["test_best_threshold"],
        "threshold_gap": abs(data["selected_threshold"] - data["test_best_threshold"]),
    }
    prefix_map = {
        "validation_metrics": "val",
        "test_metrics_at_selected_threshold": "test_selected",
        "test_best_metrics": "test_best",
    }
    for key, prefix in prefix_map.items():
        metrics = data.get(key, {})
        for metric_name, value in metrics.items():
            row[f"{prefix}_{metric_name}"] = value
    return row


def aggregate(run_dir: Path, run_name: str, folds: List[str] | None, output_dir: Path) -> Path:
    rows: List[Dict] = []
    target_folds = [f for f in find_folds(run_dir) if (not folds or f.name in folds)]
    for fold_dir in target_folds:
        summary = load_fold_summary(fold_dir, run_name)
        rows.append(flatten_row(summary))
    df = pd.DataFrame(rows).sort_values("fold")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_selected_threshold_metrics.csv"
    df.to_csv(csv_path, index=False)
    stats = df.describe().loc[["mean", "std"]]
    stats.to_csv(output_dir / f"{run_name}_selected_threshold_metrics_summary.csv")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate YOLO fold results.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/cnn_results/<run_id>.")
    parser.add_argument("--run-name", type=str, required=True, help="YOLO detector run name under fold/yolo/runs_det/.")
    parser.add_argument("--folds", type=str, default="", help="Comma-separated subset of folds (default: all).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write aggregated CSV (default: <run-dir>/aggregates).")
    args = parser.parse_args()

    fold_subset = [f.strip() for f in args.folds.split(",") if f.strip()] or None
    output_dir = args.output_dir or (args.run_dir / "aggregates")
    csv_path = aggregate(args.run_dir, args.run_name, fold_subset, output_dir)
    print(f"Aggregated metrics written to {csv_path}")


if __name__ == "__main__":
    main()
