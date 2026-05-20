from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


csv.field_size_limit(sys.maxsize)

DEFAULT_LABELS = {
    0: "20Hz20Plus",
    1: "ABZ",
    2: "DDswp",
    3: "Noise",
}

LIST_COLUMNS = ["gt_set", "gt_multiset", "all_pred_classes", "all_pred_confidences"]
SUMMARY_FILENAME = "selected_threshold_summary.json"
RAW_PREDICTIONS_FILENAME = "raw_predictions.csv"
CNN_METRICS_REL = Path("evaluation/all_metrics_by_fold_and_noise.csv")


def safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def load_label_map(run_dir: Path) -> Dict[int, str]:
    labels_path = run_dir / "labels.json"
    if not labels_path.exists():
        return dict(DEFAULT_LABELS)
    payload = json.loads(labels_path.read_text())
    return {int(idx): name for name, idx in payload.items()}


def discover_folds(run_dir: Path, run_name: str, fold_subset: List[str] | None) -> List[Path]:
    folds = sorted(p for p in run_dir.glob("fold_*") if p.is_dir())
    if fold_subset:
        wanted = set(fold_subset)
        folds = [p for p in folds if p.name in wanted]
    return [fold_dir for fold_dir in folds if (fold_dir / "yolo" / "runs_det" / run_name).exists()]


def load_raw_predictions(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, engine="python")
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(json.loads)
    return df


def combo_label(class_ids: Iterable[int], label_map: Dict[int, str]) -> str:
    names = [label_map.get(int(class_id), f"class_{class_id}") for class_id in sorted(set(class_ids))]
    return " + ".join(names) if names else "Noise-only"


def class_set(class_ids: Iterable[int]) -> Set[int]:
    return {int(class_id) for class_id in class_ids}


def class_set_relation(gt_classes: Set[int], pred_classes: Set[int]) -> str:
    if pred_classes == gt_classes:
        return "exact"
    if not pred_classes:
        return "no_prediction"
    overlap = gt_classes & pred_classes
    if not overlap:
        return "no_overlap"
    if pred_classes < gt_classes:
        return "missing_only"
    if pred_classes > gt_classes:
        return "extra_only"
    return "partial_overlap"


def short_fold_label(fold_name: str) -> str:
    label = fold_name.replace("fold_", "").replace("_noise_0.25", "")
    match = re.match(r"^(.*?)(\d{4})$", label)
    if not match:
        return label
    stem, year = match.groups()
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem).strip()
    if stem:
        stem = stem[0].upper() + stem[1:]
        return f"{stem}\n{year}"
    return year


def prepare_prediction_columns(df: pd.DataFrame, threshold: float, noise_id: int) -> pd.DataFrame:
    df = df.copy()

    def filter_predictions(row: pd.Series) -> List[int]:
        return [
            int(class_id)
            for class_id, conf in zip(row["all_pred_classes"], row["all_pred_confidences"])
            if float(conf) >= threshold
        ]

    df["pred_filt"] = df.apply(filter_predictions, axis=1)
    df["pred_set"] = df["pred_filt"].apply(class_set)
    df["gt_class_set"] = df["gt_set"].apply(class_set)
    df["pred_box_count_selected"] = df["pred_filt"].apply(len)
    df["pred_class_count_selected"] = df["pred_set"].apply(len)
    df["top1_pred"] = df["pred_filt"].apply(lambda xs: xs[0] if xs else noise_id)
    df["gt_unique_count"] = df["gt_class_set"].apply(len)
    df["gt_box_count"] = df["gt_multiset"].apply(len)
    df["is_noise_only_gt"] = df["gt_box_count"].eq(0)
    df["is_multiclass_gt"] = df["gt_unique_count"].gt(1)
    df["is_multibox_gt"] = df["gt_multiset"].apply(lambda xs: len(xs) > 1)
    df["gt_primary_in_pred"] = df.apply(lambda row: int(row["gt_primary"]) in row["pred_filt"], axis=1)
    df["any_gtset_in_pred"] = df.apply(
        lambda row: bool(row["gt_class_set"] & row["pred_set"]),
        axis=1,
    )
    df["pred_set_relation"] = df.apply(
        lambda row: class_set_relation(row["gt_class_set"], row["pred_set"]),
        axis=1,
    )
    df["pred_set_exact"] = df["pred_set_relation"].eq("exact")
    df["pred_set_missing_only"] = df["pred_set_relation"].eq("missing_only")
    df["pred_set_extra_only"] = df["pred_set_relation"].eq("extra_only")
    df["pred_set_partial_overlap"] = df["pred_set_relation"].eq("partial_overlap")
    df["pred_set_no_overlap"] = df["pred_set_relation"].eq("no_overlap")
    df["pred_set_no_prediction"] = df["pred_set_relation"].eq("no_prediction")
    df["top1_correct"] = df["top1_pred"].eq(df["gt_primary"])
    df["top1_wrong"] = ~df["top1_correct"]
    df["rescued_primary"] = df["top1_wrong"] & df["gt_primary_in_pred"]
    df["rescued_any_gtset"] = df["top1_wrong"] & df["any_gtset_in_pred"]
    df["rescued_other_gt_only"] = df["rescued_any_gtset"] & ~df["gt_primary_in_pred"]
    df["wrong_no_relevant_prediction"] = df["top1_wrong"] & ~df["any_gtset_in_pred"]
    return df


def build_fold_summary(
    fold: str,
    df: pd.DataFrame,
    threshold: float,
    selected_metrics: Dict[str, float],
) -> Dict[str, float | int | str]:
    n = len(df)
    top1_wrong = int(df["top1_wrong"].sum())
    multiclass = int(df["is_multiclass_gt"].sum())
    multibox = int(df["is_multibox_gt"].sum())
    any_gt_box = int(df["gt_box_count"].gt(0).sum())
    noise_only = int(df["is_noise_only_gt"].sum())
    rescued_primary = int(df["rescued_primary"].sum())
    rescued_any = int(df["rescued_any_gtset"].sum())
    rescued_other_only = int(df["rescued_other_gt_only"].sum())
    rescued_any_on_multiclass = int((df["rescued_any_gtset"] & df["is_multiclass_gt"]).sum())
    wrong_no_relevant = int(df["wrong_no_relevant_prediction"].sum())
    multiclass_df = df[df["is_multiclass_gt"]]
    multiclass_top1_wrong_df = multiclass_df[multiclass_df["top1_wrong"]]
    multiclass_top1_wrong = len(multiclass_top1_wrong_df)

    def count_in(frame: pd.DataFrame, column: str) -> int:
        return int(frame[column].sum()) if not frame.empty else 0

    multiclass_set_exact = count_in(multiclass_df, "pred_set_exact")
    multiclass_set_missing_only = count_in(multiclass_df, "pred_set_missing_only")
    multiclass_set_extra_only = count_in(multiclass_df, "pred_set_extra_only")
    multiclass_set_partial_overlap = count_in(multiclass_df, "pred_set_partial_overlap")
    multiclass_set_no_overlap = count_in(multiclass_df, "pred_set_no_overlap")
    multiclass_set_no_prediction = count_in(multiclass_df, "pred_set_no_prediction")
    multiclass_top1_wrong_set_exact = count_in(multiclass_top1_wrong_df, "pred_set_exact")
    multiclass_top1_wrong_set_missing_only = count_in(multiclass_top1_wrong_df, "pred_set_missing_only")
    multiclass_top1_wrong_set_extra_only = count_in(multiclass_top1_wrong_df, "pred_set_extra_only")
    multiclass_top1_wrong_set_partial_overlap = count_in(multiclass_top1_wrong_df, "pred_set_partial_overlap")
    multiclass_top1_wrong_set_no_overlap = count_in(multiclass_top1_wrong_df, "pred_set_no_overlap")
    multiclass_top1_wrong_set_no_prediction = count_in(multiclass_top1_wrong_df, "pred_set_no_prediction")

    return {
        "fold": fold,
        "selected_threshold": threshold,
        "n_samples": n,
        "gt_noise_only": noise_only,
        "gt_any_box": any_gt_box,
        "gt_multibox": multibox,
        "gt_multiclass": multiclass,
        "pred_any_box_at_selected": int(df["pred_box_count_selected"].gt(0).sum()),
        "pred_multi_box_at_selected": int(df["pred_box_count_selected"].gt(1).sum()),
        "pred_multi_class_at_selected": int(df["pred_class_count_selected"].gt(1).sum()),
        "top1_correct": int(df["top1_correct"].sum()),
        "top1_wrong": top1_wrong,
        "wrong_no_relevant_prediction": wrong_no_relevant,
        "rescued_primary": rescued_primary,
        "rescued_any_gtset": rescued_any,
        "rescued_other_gt_only": rescued_other_only,
        "rescued_any_gtset_on_multiclass": rescued_any_on_multiclass,
        "multiclass_top1_wrong": multiclass_top1_wrong,
        "multiclass_set_exact": multiclass_set_exact,
        "multiclass_set_missing_only": multiclass_set_missing_only,
        "multiclass_set_extra_only": multiclass_set_extra_only,
        "multiclass_set_partial_overlap": multiclass_set_partial_overlap,
        "multiclass_set_no_overlap": multiclass_set_no_overlap,
        "multiclass_set_no_prediction": multiclass_set_no_prediction,
        "multiclass_top1_wrong_set_exact": multiclass_top1_wrong_set_exact,
        "multiclass_top1_wrong_set_missing_only": multiclass_top1_wrong_set_missing_only,
        "multiclass_top1_wrong_set_extra_only": multiclass_top1_wrong_set_extra_only,
        "multiclass_top1_wrong_set_partial_overlap": multiclass_top1_wrong_set_partial_overlap,
        "multiclass_top1_wrong_set_no_overlap": multiclass_top1_wrong_set_no_overlap,
        "multiclass_top1_wrong_set_no_prediction": multiclass_top1_wrong_set_no_prediction,
        "gt_multiclass_rate": safe_rate(multiclass, n),
        "gt_multibox_rate": safe_rate(multibox, n),
        "top1_wrong_rate": safe_rate(top1_wrong, n),
        "rescued_primary_rate_all": safe_rate(rescued_primary, n),
        "rescued_primary_rate_wrong": safe_rate(rescued_primary, top1_wrong),
        "rescued_any_gtset_rate_all": safe_rate(rescued_any, n),
        "rescued_any_gtset_rate_wrong": safe_rate(rescued_any, top1_wrong),
        "rescued_other_gt_only_rate_wrong": safe_rate(rescued_other_only, top1_wrong),
        "wrong_no_relevant_prediction_rate_wrong": safe_rate(wrong_no_relevant, top1_wrong),
        "rescued_any_gtset_rate_multiclass": safe_rate(rescued_any_on_multiclass, multiclass),
        "multiclass_top1_wrong_rate": safe_rate(multiclass_top1_wrong, multiclass),
        "multiclass_set_exact_rate": safe_rate(multiclass_set_exact, multiclass),
        "multiclass_set_missing_only_rate": safe_rate(multiclass_set_missing_only, multiclass),
        "multiclass_set_extra_only_rate": safe_rate(multiclass_set_extra_only, multiclass),
        "multiclass_set_partial_overlap_rate": safe_rate(multiclass_set_partial_overlap, multiclass),
        "multiclass_set_no_overlap_rate": safe_rate(multiclass_set_no_overlap, multiclass),
        "multiclass_set_no_prediction_rate": safe_rate(multiclass_set_no_prediction, multiclass),
        "multiclass_top1_wrong_set_exact_rate": safe_rate(
            multiclass_top1_wrong_set_exact, multiclass_top1_wrong
        ),
        "multiclass_top1_wrong_set_missing_only_rate": safe_rate(
            multiclass_top1_wrong_set_missing_only, multiclass_top1_wrong
        ),
        "multiclass_top1_wrong_set_extra_only_rate": safe_rate(
            multiclass_top1_wrong_set_extra_only, multiclass_top1_wrong
        ),
        "multiclass_top1_wrong_set_partial_overlap_rate": safe_rate(
            multiclass_top1_wrong_set_partial_overlap, multiclass_top1_wrong
        ),
        "multiclass_top1_wrong_set_no_overlap_rate": safe_rate(
            multiclass_top1_wrong_set_no_overlap, multiclass_top1_wrong
        ),
        "multiclass_top1_wrong_set_no_prediction_rate": safe_rate(
            multiclass_top1_wrong_set_no_prediction, multiclass_top1_wrong
        ),
        "selected_test_TCR": float(selected_metrics["TCR"]),
        "selected_test_NMR": float(selected_metrics["NMR"]),
        "selected_test_CMR": float(selected_metrics["CMR"]),
        "selected_test_F": float(selected_metrics["F"]),
        "selected_test_ACC": float(selected_metrics.get("ACC", 0.0)),
    }


def build_combo_rows(fold: str, df: pd.DataFrame, label_map: Dict[int, str]) -> List[Dict[str, object]]:
    combo_counter: Counter[Tuple[int, ...]] = Counter()
    for gt_set in df["gt_set"]:
        combo = tuple(sorted(set(int(class_id) for class_id in gt_set)))
        if len(combo) > 1:
            combo_counter[combo] += 1

    rows: List[Dict[str, object]] = []
    for combo, count in sorted(combo_counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "fold": fold,
                "combo_ids": json.dumps(list(combo)),
                "combo_label": combo_label(combo, label_map),
                "count": int(count),
            }
        )
    return rows


def build_per_primary_rows(fold: str, df: pd.DataFrame, label_map: Dict[int, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for primary_id, sdf in sorted(df.groupby("gt_primary"), key=lambda item: int(item[0])):
        n = len(sdf)
        top1_wrong = int(sdf["top1_wrong"].sum())
        rescued_any = int(sdf["rescued_any_gtset"].sum())
        rows.append(
            {
                "fold": fold,
                "gt_primary": int(primary_id),
                "gt_primary_name": label_map.get(int(primary_id), f"class_{primary_id}"),
                "n_samples": n,
                "gt_multiclass": int(sdf["is_multiclass_gt"].sum()),
                "gt_multibox": int(sdf["is_multibox_gt"].sum()),
                "top1_wrong": top1_wrong,
                "rescued_primary": int(sdf["rescued_primary"].sum()),
                "rescued_any_gtset": rescued_any,
                "rescued_other_gt_only": int(sdf["rescued_other_gt_only"].sum()),
                "gt_multiclass_rate": safe_rate(int(sdf["is_multiclass_gt"].sum()), n),
                "top1_wrong_rate": safe_rate(top1_wrong, n),
                "rescued_any_gtset_rate_wrong": safe_rate(rescued_any, top1_wrong),
            }
        )
    return rows


def build_examples_df(df: pd.DataFrame, label_map: Dict[int, str], limit: int) -> pd.DataFrame:
    cols = [
        "path",
        "gt_primary",
        "gt_primary_name",
        "gt_set",
        "gt_multiset",
        "pred_filt",
        "pred_set",
        "pred_set_relation",
        "top1_pred",
        "pred_box_count_selected",
        "pred_class_count_selected",
        "rescued_primary",
        "rescued_any_gtset",
        "rescued_other_gt_only",
        "is_multiclass_gt",
        "is_multibox_gt",
    ]
    examples = df[df["rescued_any_gtset"]].copy()
    examples["top1_pred_name"] = examples["top1_pred"].map(
        lambda idx: label_map.get(int(idx), f"class_{idx}")
    )
    examples["gt_set_label"] = examples["gt_set"].apply(lambda xs: combo_label(xs, label_map))
    examples["pred_filt_label"] = examples["pred_filt"].apply(lambda xs: combo_label(xs, label_map))
    examples["pred_set_label"] = examples["pred_set"].apply(lambda xs: combo_label(xs, label_map))
    examples = examples.sort_values(
        ["rescued_other_gt_only", "is_multiclass_gt", "pred_box_count_selected"],
        ascending=[False, False, False],
    )
    return examples[cols + ["top1_pred_name", "gt_set_label", "pred_filt_label", "pred_set_label"]].head(limit)


def build_per_primary_aggregate(per_primary_df: pd.DataFrame) -> pd.DataFrame:
    if per_primary_df.empty:
        return pd.DataFrame()
    grouped = (
        per_primary_df.groupby(["gt_primary", "gt_primary_name"], as_index=False)[
            [
                "n_samples",
                "gt_multiclass",
                "gt_multibox",
                "top1_wrong",
                "rescued_primary",
                "rescued_any_gtset",
                "rescued_other_gt_only",
            ]
        ]
        .sum()
        .sort_values("gt_primary")
        .reset_index(drop=True)
    )
    grouped["gt_multiclass_rate"] = grouped.apply(
        lambda row: safe_rate(int(row["gt_multiclass"]), int(row["n_samples"])), axis=1
    )
    grouped["top1_wrong_rate"] = grouped.apply(
        lambda row: safe_rate(int(row["top1_wrong"]), int(row["n_samples"])), axis=1
    )
    grouped["rescued_any_gtset_rate_wrong"] = grouped.apply(
        lambda row: safe_rate(int(row["rescued_any_gtset"]), int(row["top1_wrong"])), axis=1
    )
    return grouped


def build_aggregate_summary(per_fold_df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "n_folds": int(len(per_fold_df)),
        "n_samples": int(per_fold_df["n_samples"].sum()),
        "gt_any_box": int(per_fold_df["gt_any_box"].sum()),
        "gt_multibox": int(per_fold_df["gt_multibox"].sum()),
        "gt_multiclass": int(per_fold_df["gt_multiclass"].sum()),
        "top1_wrong": int(per_fold_df["top1_wrong"].sum()),
        "wrong_no_relevant_prediction": int(per_fold_df["wrong_no_relevant_prediction"].sum()),
        "rescued_primary": int(per_fold_df["rescued_primary"].sum()),
        "rescued_any_gtset": int(per_fold_df["rescued_any_gtset"].sum()),
        "rescued_other_gt_only": int(per_fold_df["rescued_other_gt_only"].sum()),
        "rescued_any_gtset_on_multiclass": int(per_fold_df["rescued_any_gtset_on_multiclass"].sum()),
        "multiclass_top1_wrong": int(per_fold_df["multiclass_top1_wrong"].sum()),
        "multiclass_set_exact": int(per_fold_df["multiclass_set_exact"].sum()),
        "multiclass_set_missing_only": int(per_fold_df["multiclass_set_missing_only"].sum()),
        "multiclass_set_extra_only": int(per_fold_df["multiclass_set_extra_only"].sum()),
        "multiclass_set_partial_overlap": int(per_fold_df["multiclass_set_partial_overlap"].sum()),
        "multiclass_set_no_overlap": int(per_fold_df["multiclass_set_no_overlap"].sum()),
        "multiclass_set_no_prediction": int(per_fold_df["multiclass_set_no_prediction"].sum()),
        "multiclass_top1_wrong_set_exact": int(per_fold_df["multiclass_top1_wrong_set_exact"].sum()),
        "multiclass_top1_wrong_set_missing_only": int(
            per_fold_df["multiclass_top1_wrong_set_missing_only"].sum()
        ),
        "multiclass_top1_wrong_set_extra_only": int(per_fold_df["multiclass_top1_wrong_set_extra_only"].sum()),
        "multiclass_top1_wrong_set_partial_overlap": int(
            per_fold_df["multiclass_top1_wrong_set_partial_overlap"].sum()
        ),
        "multiclass_top1_wrong_set_no_overlap": int(per_fold_df["multiclass_top1_wrong_set_no_overlap"].sum()),
        "multiclass_top1_wrong_set_no_prediction": int(
            per_fold_df["multiclass_top1_wrong_set_no_prediction"].sum()
        ),
    }
    summary["gt_multibox_rate"] = safe_rate(int(summary["gt_multibox"]), int(summary["n_samples"]))
    summary["gt_multiclass_rate"] = safe_rate(int(summary["gt_multiclass"]), int(summary["n_samples"]))
    summary["top1_wrong_rate"] = safe_rate(int(summary["top1_wrong"]), int(summary["n_samples"]))
    summary["rescued_primary_rate_all"] = safe_rate(int(summary["rescued_primary"]), int(summary["n_samples"]))
    summary["rescued_primary_rate_wrong"] = safe_rate(int(summary["rescued_primary"]), int(summary["top1_wrong"]))
    summary["rescued_any_gtset_rate_all"] = safe_rate(int(summary["rescued_any_gtset"]), int(summary["n_samples"]))
    summary["rescued_any_gtset_rate_wrong"] = safe_rate(int(summary["rescued_any_gtset"]), int(summary["top1_wrong"]))
    summary["rescued_other_gt_only_rate_wrong"] = safe_rate(
        int(summary["rescued_other_gt_only"]), int(summary["top1_wrong"])
    )
    summary["wrong_no_relevant_prediction_rate_wrong"] = safe_rate(
        int(summary["wrong_no_relevant_prediction"]), int(summary["top1_wrong"])
    )
    summary["rescued_any_gtset_rate_multiclass"] = safe_rate(
        int(summary["rescued_any_gtset_on_multiclass"]), int(summary["gt_multiclass"])
    )
    summary["multiclass_top1_wrong_rate"] = safe_rate(
        int(summary["multiclass_top1_wrong"]), int(summary["gt_multiclass"])
    )
    for relation in [
        "exact",
        "missing_only",
        "extra_only",
        "partial_overlap",
        "no_overlap",
        "no_prediction",
    ]:
        summary[f"multiclass_set_{relation}_rate"] = safe_rate(
            int(summary[f"multiclass_set_{relation}"]), int(summary["gt_multiclass"])
        )
        summary[f"multiclass_top1_wrong_set_{relation}_rate"] = safe_rate(
            int(summary[f"multiclass_top1_wrong_set_{relation}"]),
            int(summary["multiclass_top1_wrong"]),
        )
    return summary


def load_cnn_metrics(run_dir: Path) -> pd.DataFrame:
    cnn_path = run_dir / CNN_METRICS_REL
    if not cnn_path.exists():
        return pd.DataFrame()
    cnn_df = pd.read_csv(cnn_path).copy()
    cnn_df = cnn_df.rename(
        columns={
            "fold": "fold_name",
            "TCR": "cnn_TCR",
            "NMR": "cnn_NMR",
            "CMR": "cnn_CMR",
            "F": "cnn_F",
            "ACC": "cnn_ACC",
        }
    )
    return cnn_df


def build_fold_delta_table(run_dir: Path, per_fold_df: pd.DataFrame) -> pd.DataFrame:
    cnn_df = load_cnn_metrics(run_dir)
    if cnn_df.empty:
        return pd.DataFrame()
    yolo_df = per_fold_df.rename(columns={"fold": "fold_name"}).copy()
    merged = cnn_df.merge(yolo_df, on="fold_name", how="inner")
    merged["delta_F"] = merged["selected_test_F"] - merged["cnn_F"]
    merged["delta_TCR"] = merged["selected_test_TCR"] - merged["cnn_TCR"]
    merged["delta_NMR"] = merged["selected_test_NMR"] - merged["cnn_NMR"]
    merged["delta_CMR"] = merged["selected_test_CMR"] - merged["cnn_CMR"]
    merged["delta_ACC"] = merged["selected_test_ACC"] - merged["cnn_ACC"]
    return merged.sort_values("fold_name").reset_index(drop=True)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_summary_text(
    out_path: Path,
    run_dir: Path,
    run_name: str,
    per_fold_df: pd.DataFrame,
    aggregate_summary: Dict[str, object],
    combo_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("YOLO multiclass / Top-1 collapse analysis")
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"run_name: {run_name}")
    lines.append("")
    lines.append("Aggregate summary")
    for key in [
        "n_folds",
        "n_samples",
        "gt_any_box",
        "gt_multibox",
        "gt_multiclass",
        "top1_wrong",
        "wrong_no_relevant_prediction",
        "rescued_primary",
        "rescued_any_gtset",
        "rescued_other_gt_only",
        "rescued_any_gtset_on_multiclass",
        "multiclass_top1_wrong",
        "multiclass_set_exact",
        "multiclass_set_missing_only",
        "multiclass_set_extra_only",
        "multiclass_set_partial_overlap",
        "multiclass_set_no_overlap",
        "multiclass_set_no_prediction",
        "multiclass_top1_wrong_set_exact",
        "multiclass_top1_wrong_set_missing_only",
        "multiclass_top1_wrong_set_extra_only",
        "multiclass_top1_wrong_set_partial_overlap",
        "multiclass_top1_wrong_set_no_overlap",
        "multiclass_top1_wrong_set_no_prediction",
    ]:
        lines.append(f"- {key}: {aggregate_summary[key]}")
    for key in [
        "gt_multibox_rate",
        "gt_multiclass_rate",
        "top1_wrong_rate",
        "rescued_primary_rate_all",
        "rescued_primary_rate_wrong",
        "rescued_any_gtset_rate_all",
        "rescued_any_gtset_rate_wrong",
        "rescued_other_gt_only_rate_wrong",
        "wrong_no_relevant_prediction_rate_wrong",
        "rescued_any_gtset_rate_multiclass",
        "multiclass_top1_wrong_rate",
        "multiclass_set_exact_rate",
        "multiclass_set_missing_only_rate",
        "multiclass_set_extra_only_rate",
        "multiclass_set_partial_overlap_rate",
        "multiclass_set_no_overlap_rate",
        "multiclass_set_no_prediction_rate",
        "multiclass_top1_wrong_set_exact_rate",
        "multiclass_top1_wrong_set_missing_only_rate",
        "multiclass_top1_wrong_set_extra_only_rate",
        "multiclass_top1_wrong_set_partial_overlap_rate",
        "multiclass_top1_wrong_set_no_overlap_rate",
        "multiclass_top1_wrong_set_no_prediction_rate",
    ]:
        lines.append(f"- {key}: {aggregate_summary[key]:.4f}")

    lines.append("")
    lines.append("Per-fold selected-threshold summary")
    for row in per_fold_df.sort_values("fold").to_dict(orient="records"):
        lines.append(
            (
                f"- {row['fold']}: thr={row['selected_threshold']:.3f}, "
                f"F={row['selected_test_F']:.3f}, "
                f"gt_multiclass={row['gt_multiclass']} ({row['gt_multiclass_rate']:.2%}), "
                f"top1_wrong={row['top1_wrong']} ({row['top1_wrong_rate']:.2%}), "
                f"rescued_any={row['rescued_any_gtset']} ({row['rescued_any_gtset_rate_wrong']:.2%} of wrong)"
            )
        )

    if not combo_df.empty:
        lines.append("")
        lines.append("Most common multiclass combinations (aggregate)")
        for row in combo_df.head(10).to_dict(orient="records"):
            lines.append(f"- {row['combo_label']}: {row['count']}")

    if not delta_df.empty:
        lines.append("")
        lines.append("Fold-level YOLO vs CNN deltas (means)")
        lines.append(f"- delta_F_mean: {delta_df['delta_F'].mean():.4f}")
        lines.append(f"- delta_TCR_mean: {delta_df['delta_TCR'].mean():.4f}")
        lines.append(f"- delta_NMR_mean: {delta_df['delta_NMR'].mean():.4f}")
        lines.append(f"- delta_CMR_mean: {delta_df['delta_CMR'].mean():.4f}")

    out_path.write_text("\n".join(lines) + "\n")


def plot_multiclass_rate_by_fold(per_fold_df: pd.DataFrame, out_path: Path) -> None:
    if per_fold_df.empty:
        return
    sdf = per_fold_df.sort_values("gt_multiclass_rate", ascending=False)
    labels = [short_fold_label(fold) for fold in sdf["fold"]]
    x = list(range(len(sdf)))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(x, sdf["gt_multiclass_rate"], color="#2C7FB8")
    ax.set_ylabel("Multiclass snippet rate")
    ax.set_title("Ground-truth multiclass rate by fold")
    ax.set_ylim(0.0, max(0.05, float(sdf["gt_multiclass_rate"].max()) * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, sdf["gt_multiclass_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.004,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rescued_error_breakdown(per_fold_df: pd.DataFrame, out_path: Path) -> None:
    if per_fold_df.empty:
        return
    sdf = per_fold_df.sort_values("rescued_any_gtset_rate_wrong", ascending=False).copy()
    labels = [short_fold_label(fold) for fold in sdf["fold"]]
    x = list(range(len(sdf)))
    no_rel = sdf["wrong_no_relevant_prediction_rate_wrong"].tolist()
    primary_not_top1 = sdf["rescued_primary_rate_wrong"].tolist()
    other_gt_only = sdf["rescued_other_gt_only_rate_wrong"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, no_rel, label="No relevant prediction", color="#BDBDBD")
    ax.bar(x, primary_not_top1, bottom=no_rel, label="Primary predicted but not Top-1", color="#4DAF4A")
    stacked_bottom = [a + b for a, b in zip(no_rel, primary_not_top1)]
    ax.bar(x, other_gt_only, bottom=stacked_bottom, label="Other GT class predicted but not primary", color="#E6550D")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Share within Top-1 wrong samples")
    ax.set_title("Top-1 wrong samples by fold: normalized breakdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_multiclass_vs_delta_f(delta_df: pd.DataFrame, out_path: Path) -> None:
    if delta_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(delta_df["gt_multiclass_rate"], delta_df["delta_F"], color="#6A51A3")
    for _, row in delta_df.iterrows():
        ax.annotate(
            row["fold_name"].replace("fold_", "").replace("_noise_0.25", ""),
            (row["gt_multiclass_rate"], row["delta_F"]),
            fontsize=8,
            alpha=0.8,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Multiclass snippet rate")
    ax.set_ylabel("YOLO - CNN delta F")
    ax.set_title("Fold multiclass rate vs YOLO-CNN delta F")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_multiclass_vs_rescued_rate(per_fold_df: pd.DataFrame, out_path: Path) -> None:
    if per_fold_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(per_fold_df["gt_multiclass_rate"], per_fold_df["rescued_any_gtset_rate_wrong"], color="#C51B8A")
    for _, row in per_fold_df.iterrows():
        ax.annotate(
            short_fold_label(row["fold"]).replace("\n", " "),
            (row["gt_multiclass_rate"], row["rescued_any_gtset_rate_wrong"]),
            fontsize=8,
            alpha=0.8,
        )
    ax.set_xlabel("Multiclass snippet rate")
    ax.set_ylabel("Share of Top-1 wrong samples rescued by any GT class")
    ax.set_title("Multiclass rate vs collapse-sensitive rescued-error rate")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_analysis(
    run_dir: Path,
    run_name: str,
    output_dir: Path,
    fold_subset: List[str] | None,
    examples_per_fold: int,
) -> Dict[str, pd.DataFrame | Dict[str, object]]:
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(run_dir)
    noise_id = next((class_id for class_id, name in label_map.items() if name == "Noise"), 3)

    fold_dirs = discover_folds(run_dir, run_name, fold_subset)
    if not fold_dirs:
        raise SystemExit(f"No folds found for run '{run_name}' under {run_dir}")

    per_fold_rows: List[Dict[str, object]] = []
    combo_rows: List[Dict[str, object]] = []
    per_primary_rows: List[Dict[str, object]] = []

    for fold_dir in fold_dirs:
        fold = fold_dir.name
        run_path = fold_dir / "yolo" / "runs_det" / run_name
        summary_path = run_path / SUMMARY_FILENAME
        raw_path = run_path / RAW_PREDICTIONS_FILENAME
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw predictions file: {raw_path}")

        summary_payload = json.loads(summary_path.read_text())
        selected_threshold = float(summary_payload["selected_threshold"])
        selected_metrics = summary_payload["test_metrics_at_selected_threshold"]

        df = load_raw_predictions(raw_path)
        df = prepare_prediction_columns(df, selected_threshold, noise_id=noise_id)

        per_fold_rows.append(
            build_fold_summary(
                fold=fold,
                df=df,
                threshold=selected_threshold,
                selected_metrics=selected_metrics,
            )
        )
        combo_rows.extend(build_combo_rows(fold=fold, df=df, label_map=label_map))
        per_primary_rows.extend(build_per_primary_rows(fold=fold, df=df, label_map=label_map))
        build_examples_df(df=df, label_map=label_map, limit=examples_per_fold).to_csv(
            examples_dir / f"{fold}_rescued_examples.csv",
            index=False,
        )

    per_fold_df = pd.DataFrame(per_fold_rows).sort_values("fold").reset_index(drop=True)
    combo_df = pd.DataFrame(combo_rows)
    per_primary_df = pd.DataFrame(per_primary_rows)

    combo_agg_df = (
        combo_df.groupby(["combo_ids", "combo_label"], as_index=False)["count"].sum().sort_values(
            ["count", "combo_label"], ascending=[False, True]
        )
        if not combo_df.empty
        else pd.DataFrame(columns=["combo_ids", "combo_label", "count"])
    )
    per_primary_agg_df = build_per_primary_aggregate(per_primary_df)
    aggregate_summary = build_aggregate_summary(per_fold_df)
    delta_df = build_fold_delta_table(run_dir, per_fold_df)

    per_fold_df.to_csv(tables_dir / "per_fold_summary.csv", index=False)
    combo_df.to_csv(tables_dir / "multiclass_combo_counts_by_fold.csv", index=False)
    combo_agg_df.to_csv(tables_dir / "multiclass_combo_counts_aggregate.csv", index=False)
    per_primary_df.to_csv(tables_dir / "per_primary_summary_by_fold.csv", index=False)
    per_primary_agg_df.to_csv(tables_dir / "per_primary_summary_aggregate.csv", index=False)
    delta_df.to_csv(tables_dir / "fold_join_with_cnn_yolo_delta.csv", index=False)

    write_json(output_dir / "aggregate_summary.json", aggregate_summary)
    write_summary_text(
        out_path=output_dir / "summary.txt",
        run_dir=run_dir,
        run_name=run_name,
        per_fold_df=per_fold_df,
        aggregate_summary=aggregate_summary,
        combo_df=combo_agg_df,
        delta_df=delta_df,
    )

    plot_multiclass_rate_by_fold(per_fold_df, plots_dir / "multiclass_rate_by_fold.png")
    plot_rescued_error_breakdown(per_fold_df, plots_dir / "rescued_error_breakdown_by_fold.png")
    plot_multiclass_vs_delta_f(delta_df, plots_dir / "multiclass_vs_delta_f.png")
    plot_multiclass_vs_rescued_rate(per_fold_df, plots_dir / "multiclass_vs_rescued_rate.png")

    return {
        "per_fold_df": per_fold_df,
        "combo_agg_df": combo_agg_df,
        "per_primary_agg_df": per_primary_agg_df,
        "delta_df": delta_df,
        "aggregate_summary": aggregate_summary,
    }
