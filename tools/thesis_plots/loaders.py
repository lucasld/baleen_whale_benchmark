from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from thesis_plots.config import DEFAULT_STYLE_FILE, EXPLORATORY_HARMONIZED_THRESHOLDS, RUN_ORDER, format_fold_label


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path


def load_run_colors(style_file: Path | None = None) -> Dict[str, str]:
    style_path = style_file or DEFAULT_STYLE_FILE
    payload = json.loads(_require(style_path).read_text())
    colors = payload.get("run_colors", {})
    colors.setdefault("CNN", "#222222")
    return colors


def sort_runs(values: Iterable[str]) -> List[str]:
    order = {run: idx for idx, run in enumerate(RUN_ORDER)}
    return sorted(values, key=lambda x: (order.get(x, 999), x))


def load_exploratory_points(run_dir: Path, exploratory_fold: str) -> pd.DataFrame:
    path = _require(run_dir / exploratory_fold / "yolo" / "comparisons" / "latest" / "comparison_runs.csv")
    df = pd.read_csv(path)
    return df.sort_values("run_name").reset_index(drop=True)


def load_exploratory_sweeps(run_dir: Path, exploratory_fold: str, run_names: Iterable[str]) -> pd.DataFrame:
    rows = []
    class_to_int = _load_class_to_int(run_dir)
    noise_id = class_to_int.get("Noise", len(class_to_int) - 1)
    n_classes = len(class_to_int)
    for run_name in run_names:
        raw_path = _require(run_dir / exploratory_fold / "yolo" / "runs_det" / run_name / "raw_predictions.csv")
        df = pd.read_csv(raw_path, usecols=["gt_primary", "all_pred_classes", "all_pred_confidences"])
        if df.empty:
            continue
        df["gt_primary"] = df["gt_primary"].astype(int)
        df["all_pred_classes"] = df["all_pred_classes"].apply(_parse_int_list)
        df["all_pred_confidences"] = df["all_pred_confidences"].apply(_parse_float_list)
        rows.extend(_compute_harmonized_sweep_rows(df, run_name, noise_id, n_classes, class_to_int))
    if not rows:
        raise ValueError("No exploratory sweep rows found.")
    out = pd.DataFrame(rows)
    return out.sort_values(["run_name", "threshold"]).reset_index(drop=True)


def load_final_selected_summary(run_dir: Path, run_name: str) -> pd.DataFrame:
    path = _require(
        run_dir / "yolo" / "analysis" / "manual_compare" / f"02_compare_{run_name}_aggregated" / "tables" / "summary_mean_std.csv"
    )
    df = pd.read_csv(path)
    return df.sort_values("run_name").reset_index(drop=True)


def load_final_sweep_summary(run_dir: Path, run_name: str) -> pd.DataFrame:
    fold_rows = load_final_sweep_rows(run_dir, run_name)
    return summarize_final_sweep_rows(fold_rows, run_name)


def summarize_final_sweep_rows(fold_rows: pd.DataFrame, run_name: str) -> pd.DataFrame:
    metric_cols = ["TCR", "NMR", "CMR", "F", "ACC", "FCR"]
    grouped = fold_rows.groupby("threshold", as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = ["threshold"] + [f"{metric}_{stat}" for metric, stat in grouped.columns.tolist()[1:]]
    grouped["run_name"] = run_name
    return grouped.sort_values("threshold").reset_index(drop=True)


def load_final_selected_thresholds(run_dir: Path, run_name: str) -> pd.DataFrame:
    path = _require(
        run_dir
        / "yolo"
        / "analysis"
        / "manual_compare"
        / f"05_aggregate_{run_name}_all_folds"
        / "tables"
        / f"{run_name}_selected_threshold_metrics.csv"
    )
    df = pd.read_csv(path)
    df["fold_label"] = df["fold"].map(format_fold_label)
    return df.sort_values("fold_label").reset_index(drop=True)


def load_final_by_fold_comparison(run_dir: Path, run_name: str) -> pd.DataFrame:
    path = _require(
        run_dir
        / "yolo"
        / "analysis"
        / "manual_compare"
        / f"01_compare_{run_name}_by_fold"
        / "tables"
        / "comparison_runs.csv"
    )
    df = pd.read_csv(path)
    df = df[df["run_name"].isin([run_name, "CNN"])].copy()
    df["fold_label"] = df["fold"].map(format_fold_label)
    return df.sort_values(["fold_label", "run_name"]).reset_index(drop=True)


def load_final_sweep_rows(run_dir: Path, run_name: str) -> pd.DataFrame:
    class_to_int = _load_class_to_int(run_dir)
    noise_id = class_to_int.get("Noise", len(class_to_int) - 1)
    n_classes = len(class_to_int)
    rows = []
    for raw_path in sorted(run_dir.glob(f"fold_*/yolo/runs_det/{run_name}/raw_predictions.csv")):
        fold = raw_path.parents[3].name
        df = pd.read_csv(_require(raw_path), usecols=["gt_primary", "all_pred_classes", "all_pred_confidences"])
        if df.empty:
            continue
        df["gt_primary"] = df["gt_primary"].astype(int)
        df["all_pred_classes"] = df["all_pred_classes"].apply(_parse_int_list)
        df["all_pred_confidences"] = df["all_pred_confidences"].apply(_parse_float_list)
        fold_rows = _compute_harmonized_sweep_rows(df, run_name, noise_id, n_classes, class_to_int)
        for row in fold_rows:
            row["fold"] = fold
            row["fold_label"] = format_fold_label(fold)
            rows.append(row)
    if not rows:
        raise ValueError(f"No final sweep rows found for run {run_name}.")
    out = pd.DataFrame(rows)
    return out.sort_values(["fold_label", "threshold"]).reset_index(drop=True)


def load_cnn_metrics(run_dir: Path) -> pd.DataFrame:
    path = _require(run_dir / "evaluation" / "all_metrics_by_fold_and_noise.csv")
    df = pd.read_csv(path)
    df["fold_label"] = df["fold"].map(format_fold_label)
    return df.sort_values("fold_label").reset_index(drop=True)


def load_multiclass_summary(run_dir: Path, run_name: str) -> dict:
    path = _require(run_dir / "yolo" / "multiclass_analysis" / run_name / "latest" / "aggregate_summary.json")
    return json.loads(path.read_text())


def load_multiclass_per_fold(run_dir: Path, run_name: str) -> pd.DataFrame:
    path = _require(
        run_dir / "yolo" / "multiclass_analysis" / run_name / "latest" / "tables" / "per_fold_summary.csv"
    )
    df = pd.read_csv(path)
    df["fold_label"] = df["fold"].map(format_fold_label)
    return df.sort_values("fold_label").reset_index(drop=True)


def _load_class_to_int(run_dir: Path) -> Dict[str, int]:
    labels_path = _require(run_dir / "labels.json")
    payload = json.loads(labels_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected labels.json format: {labels_path}")
    return {str(k): int(v) for k, v in payload.items()}


def _parse_int_list(value) -> List[int]:
    return _parse_numeric_list(value, dtype=int)


def _parse_float_list(value) -> List[float]:
    return _parse_numeric_list(value, dtype=float)


def _parse_numeric_list(value, dtype) -> List:
    if isinstance(value, list):
        return [dtype(v) for v in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text in {"", "[]"}:
        return []
    if text[0] == "[" and text[-1] == "]":
        text = text[1:-1].strip()
    if not text:
        return []
    arr = np.fromstring(text, sep=",", dtype=dtype)
    return arr.tolist()


def _predict_top1(classes_per_sample: List[List[int]], noise_id: int) -> List[int]:
    preds: List[int] = []
    for classes in classes_per_sample:
        preds.append(int(classes[0]) if classes else noise_id)
    return preds


def _compute_confusion(true_labels: List[int], pred_labels: List[int], n_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for t, p in zip(true_labels, pred_labels):
        cm[int(t)][int(p)] += 1
    return cm


def _compute_paper_metrics(cm: List[List[int]], class_to_int: Dict[str, int]) -> Dict[str, float]:
    n_classes = len(cm)
    noise_idx = class_to_int.get("Noise", n_classes - 1)
    non_noise = [i for i in range(n_classes) if i != noise_idx]

    tcrs = []
    for i in non_noise:
        total = sum(cm[i])
        correct = cm[i][i]
        tcrs.append((correct / total) if total > 0 else 0.0)
    tcr = sum(tcrs) / len(tcrs) if tcrs else 0.0

    noise_total = sum(cm[noise_idx])
    noise_mis = noise_total - cm[noise_idx][noise_idx]
    nmr = (noise_mis / noise_total) if noise_total > 0 else 0.0

    cmrs = []
    for i in non_noise:
        total = sum(cm[i])
        mis = sum(cm[i][j] for j in non_noise if j != i)
        cmrs.append((mis / total) if total > 0 else 0.0)
    cmr = sum(cmrs) / len(cmrs) if cmrs else 0.0

    f = (tcr + (1 - nmr) + (1 - nmr) + (1 - cmr)) / 4.0
    total = sum(sum(row) for row in cm)
    acc = (sum(cm[i][i] for i in range(n_classes)) / total) if total > 0 else 0.0
    fcr = 1.0 - acc
    return {"TCR": tcr, "NMR": nmr, "CMR": cmr, "F": f, "ACC": acc, "FCR": fcr}


def _compute_harmonized_sweep_rows(
    df: pd.DataFrame,
    run_name: str,
    noise_id: int,
    n_classes: int,
    class_to_int: Dict[str, int],
) -> List[dict]:
    y_true = df["gt_primary"].astype(int).tolist()
    classes_series = df["all_pred_classes"].tolist()
    conf_series = df["all_pred_confidences"].tolist()
    rows: List[dict] = []
    for thr in EXPLORATORY_HARMONIZED_THRESHOLDS:
        filtered = [
            [cls for cls, conf in zip(classes, confidences) if float(conf) >= thr]
            for classes, confidences in zip(classes_series, conf_series)
        ]
        y_pred = _predict_top1(filtered, noise_id)
        cm = _compute_confusion(y_true, y_pred, n_classes)
        metrics = _compute_paper_metrics(cm, class_to_int)
        rows.append(
            {
                "run_name": run_name,
                "threshold": float(thr),
                "strategy": "top1",
                **metrics,
            }
        )
    return rows
