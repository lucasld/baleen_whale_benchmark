#!/usr/bin/env python3
"""Compute final set-aware CNN/YOLO comparison from saved YOLO box tables.

This is a lightweight post-processing script. It does not run model inference.
It reads the per-fold tables produced by ``analyze_yolo_multiclass_iou.py``,
recomputes same-class IoU matching from saved boxes for arbitrary thresholds,
and writes final tables, plots, and method notes for thesis reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


csv.field_size_limit(sys.maxsize)

CALL_IDS = {0, 1, 2}
NOISE_ID = 3
METRICS = ["TCR", "NMR", "CMR", "F"]
DEFAULT_IOUS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def iou_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_boxes(value: str) -> list[Box]:
    rows = json.loads(value) if isinstance(value, str) and value else []
    return [
        Box(
            class_id=int(row["class_id"]),
            xyxy=tuple(float(v) for v in row["xyxy"]),
            confidence=float(row.get("confidence", 1.0)),
        )
        for row in rows
    ]


def parse_set(value: str) -> set[int]:
    return {int(v) for v in json.loads(value)} if isinstance(value, str) and value else set()


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_same_class(pred_boxes: list[Box], gt_boxes: list[Box], threshold: float) -> set[int]:
    used_gt: set[int] = set()
    matched_classes: set[int] = set()
    for pred in sorted(pred_boxes, key=lambda box: box.confidence, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt or gt.class_id != pred.class_id:
                continue
            overlap = box_iou(pred.xyxy, gt.xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx is not None and best_iou >= threshold:
            used_gt.add(best_idx)
            matched_classes.add(pred.class_id)
    return matched_classes


def prediction_set_from_cnn_row(row: pd.Series) -> set[int]:
    pred = int(np.argmax([float(row[str(i)]) for i in range(4)]))
    return set() if pred == NOISE_ID else {pred}


def compute_metrics(frame: pd.DataFrame, pred_col: str, recovered_col: str | None, nmr: float) -> dict[str, float]:
    recovered_col = recovered_col or pred_col
    tcrs: list[float] = []
    cmrs: list[float] = []
    for class_id in sorted(CALL_IDS):
        contains_class = frame["gt_set"].apply(lambda values: class_id in values)
        total = int(contains_class.sum())
        if total == 0:
            tcrs.append(0.0)
            cmrs.append(0.0)
            continue
        correct = frame.loc[contains_class, recovered_col].apply(lambda values: class_id in values).sum()
        extra = frame.loc[contains_class].apply(
            lambda row: bool((row[pred_col] - row["gt_set"]) & CALL_IDS),
            axis=1,
        ).sum()
        tcrs.append(float(correct) / total)
        cmrs.append(float(extra) / total)
    tcr = float(np.mean(tcrs))
    cmr = float(np.mean(cmrs))
    return {
        "TCR": tcr,
        "NMR": float(nmr),
        "CMR": cmr,
        "F": float(np.mean([tcr, 1.0 - nmr, 1.0 - nmr, 1.0 - cmr])),
    }


def compute_call_set_rates(frame: pd.DataFrame, pred_col: str, recovered_col: str | None) -> dict[str, float]:
    recovered_col = recovered_col or pred_col
    exact = frame.apply(
        lambda row: row[recovered_col] == row["gt_set"] and not bool((row[pred_col] - row["gt_set"]) & CALL_IDS),
        axis=1,
    ).mean()
    recall = frame.apply(lambda row: len(row[recovered_col] & row["gt_set"]) / len(row["gt_set"]), axis=1).mean()
    return {
        "exact_call_set_rate": float(exact),
        "mean_call_set_recall": float(recall),
    }


def load_cnn_predictions(benchmark_run_dir: Path, fold: str) -> pd.DataFrame:
    matches = sorted((benchmark_run_dir / "evaluation").glob(f"predictions_{fold}_noise*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CNN prediction CSV found for {fold}")
    df = pd.read_csv(matches[0])
    df["filename"] = df["path"].apply(lambda value: Path(str(value)).name)
    df["cnn_pred_set"] = df.apply(prediction_set_from_cnn_row, axis=1)
    return df


def load_benchmark_metrics(benchmark_run_dir: Path) -> pd.DataFrame:
    path = benchmark_run_dir / "evaluation" / "all_metrics_by_fold_and_noise.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark metrics CSV: {path}")
    return pd.read_csv(path).set_index("fold")


def load_fold_table(path: Path, iou_thresholds: list[float]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["gt_set"] = frame["gt_set"].apply(parse_set)
    frame["yolo_pred_set_class_only"] = frame["yolo_pred_set_class_only"].apply(parse_set)
    frame["gt_boxes"] = frame["gt_boxes_json"].apply(parse_boxes)
    frame["yolo_pred_boxes"] = frame["yolo_pred_boxes_json"].apply(parse_boxes)
    for threshold in iou_thresholds:
        tag = iou_tag(threshold)
        frame[f"yolo_pred_set_iou_{tag}"] = frame.apply(
            lambda row: match_same_class(row["yolo_pred_boxes"], row["gt_boxes"], threshold),
            axis=1,
        )
    return frame


def build_variant_rows(summary_by_fold: pd.DataFrame, iou_thresholds: list[float]) -> pd.DataFrame:
    variants = [("cnn", "CNN set-aware"), ("yolo_class_set", "YOLO class set")]
    variants.extend((f"yolo_iou_{iou_tag(thr)}", f"YOLO IoU >= {thr:.2f}") for thr in iou_thresholds)
    rows = []
    for key, label in variants:
        row = {"variant": key, "label": label}
        for metric in METRICS:
            col = f"{key}_{metric}"
            row[f"{metric}_mean"] = float(summary_by_fold[col].mean())
            row[f"{metric}_std"] = float(summary_by_fold[col].std(ddof=1))
        for col_suffix in ["exact_call_set_rate", "mean_call_set_recall"]:
            col = f"{key}_{col_suffix}"
            if col in summary_by_fold.columns:
                row[f"{col_suffix}_mean"] = float(summary_by_fold[col].mean())
                row[f"{col_suffix}_std"] = float(summary_by_fold[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_selected_metrics(summary: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.6,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )
    colors = {
        "cnn": "#222222",
        "yolo_class_set": "#5A2CA0",
        "yolo_iou_0p25": "#41AB5D",
        "yolo_iou_0p5": "#238B45",
        "yolo_iou_0p75": "#006D2C",
    }
    x = np.arange(len(METRICS), dtype=float)
    width = 0.12
    offsets = (np.arange(len(summary)) - (len(summary) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for offset, (_, row) in zip(offsets, summary.iterrows()):
        key = row["variant"]
        xs = x + offset
        vals = [row[f"{metric}_mean"] for metric in METRICS]
        errs = [row[f"{metric}_std"] for metric in METRICS]
        ax.errorbar(
            xs,
            vals,
            yerr=errs,
            fmt="o",
            capsize=3,
            lw=1.3,
            ms=5,
            color=colors.get(key, "#666666"),
            label=row["label"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    colors = ["#222222", "#5A2CA0", "#41AB5D", "#238B45", "#006D2C"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["F_mean"],
        yerr=summary["F_std"],
        color=colors[: len(summary)],
        edgecolor="black",
        linewidth=0.4,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(" ", "\n", 1) for label in summary["label"]])
    ax.set_ylabel("Set-aware F")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def write_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "% Auto-generated by tools/compute_set_aware_comparison.py",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scoring variant & TCR & NMR & CMR & F \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        values = " & ".join(format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"]) for metric in METRICS)
        lines.append(f"{row['label']} & {values} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines))


def write_method_notes(out_path: Path, args: argparse.Namespace, iou_thresholds: list[float]) -> None:
    out_path.write_text(
        "\n".join(
            [
                "# Set-Aware Comparison Method Notes",
                "",
                f"Analysis directory: `{args.analysis_dir}`",
                f"Benchmark run directory: `{args.benchmark_run_dir}`",
                f"IoU thresholds: `{', '.join(f'{v:.2f}' for v in iou_thresholds)}`",
                "",
                "Ground truth is the set of call classes present in each non-noise snippet, read from the saved YOLO fold tables.",
                "CNN call predictions are aligned to call snippets by filename; call filenames were verified to align exactly.",
                "CNN NMR is imported from the original benchmark fold metrics because sampled noise filenames are not consistently aligned across CNN and YOLO artifacts.",
                "YOLO class-set predictions use all retained detections after the validation-selected confidence threshold.",
                "YOLO IoU-aware rows count a class as recovered only when a same-class predicted box is greedily matched to a ground-truth box at the stated IoU threshold.",
                "YOLO NMR and CMR use the full retained predicted class set, so unmatched false positives and extra predicted classes remain counted.",
                "F uses the Schall-style formula: mean(TCR, 1 - NMR, 1 - NMR, 1 - CMR).",
                "",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute final set-aware CNN/YOLO comparison from saved IoU fold tables.")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Directory containing fold_tables/ from multiclass IoU analysis.")
    parser.add_argument("--benchmark-run-dir", type=Path, required=True, help="Path to outputs/cnn_results/<run_id> with CNN evaluation outputs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <analysis-dir>/set_aware_comparison_final.")
    parser.add_argument("--iou-thresholds", default="0.25,0.5,0.75", help="Comma-separated IoU thresholds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    benchmark_run_dir = args.benchmark_run_dir.resolve()
    out_dir = (args.output_dir or (analysis_dir / "set_aware_comparison_final")).resolve()
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    iou_thresholds = [float(value.strip()) for value in args.iou_thresholds.split(",") if value.strip()]

    fold_tables = sorted((analysis_dir / "fold_tables").glob("*_iou_predictions.csv"))
    if not fold_tables:
        raise FileNotFoundError(f"No fold tables found under {analysis_dir / 'fold_tables'}")

    benchmark_metrics = load_benchmark_metrics(benchmark_run_dir)
    rows: list[dict[str, float | int | str]] = []
    for fold_table in fold_tables:
        frame_all = load_fold_table(fold_table, iou_thresholds)
        fold = str(frame_all["fold"].iloc[0])
        frame_calls = frame_all[frame_all["gt_set"].apply(len).gt(0)].copy()

        cnn_predictions = load_cnn_predictions(benchmark_run_dir, fold)
        cnn_call_predictions = cnn_predictions[cnn_predictions["true_label"] != NOISE_ID]
        cnn_call_map = dict(zip(cnn_call_predictions["filename"], cnn_call_predictions["cnn_pred_set"]))
        frame_calls["cnn_pred_set"] = frame_calls["filename"].map(cnn_call_map)
        if frame_calls["cnn_pred_set"].isna().any():
            missing = int(frame_calls["cnn_pred_set"].isna().sum())
            raise RuntimeError(f"{fold}: {missing} call snippets missing CNN predictions")

        cnn_nmr = float(benchmark_metrics.loc[fold, "NMR"])
        yolo_nmr = float(
            frame_all[frame_all["gt_set"].apply(len).eq(0)]["yolo_pred_set_class_only"]
            .apply(lambda values: bool(values & CALL_IDS))
            .mean()
        )

        row: dict[str, float | int | str] = {
            "fold": fold,
            "n_all_snippets": int(len(frame_all)),
            "n_call_snippets": int(len(frame_calls)),
            "gt_multiclass": int(frame_calls["gt_class_count"].gt(1).sum()),
            "gt_multiclass_rate_calls": float(frame_calls["gt_class_count"].gt(1).mean()),
        }

        for prefix, pred_col, recovered_col, nmr in [
            ("cnn", "cnn_pred_set", None, cnn_nmr),
            ("yolo_class_set", "yolo_pred_set_class_only", None, yolo_nmr),
        ]:
            for key, value in compute_metrics(frame_calls, pred_col, recovered_col, nmr).items():
                row[f"{prefix}_{key}"] = value
            for key, value in compute_call_set_rates(frame_calls, pred_col, recovered_col).items():
                row[f"{prefix}_{key}"] = value

        multiclass = frame_calls[frame_calls["gt_class_count"].gt(1)]
        for threshold in iou_thresholds:
            tag = iou_tag(threshold)
            prefix = f"yolo_iou_{tag}"
            recovered_col = f"yolo_pred_set_iou_{tag}"
            for key, value in compute_metrics(frame_calls, "yolo_pred_set_class_only", recovered_col, yolo_nmr).items():
                row[f"{prefix}_{key}"] = value
            for key, value in compute_call_set_rates(frame_calls, "yolo_pred_set_class_only", recovered_col).items():
                row[f"{prefix}_{key}"] = value
            exact_multi = multiclass.apply(
                lambda r: r[recovered_col] == r["gt_set"] and not bool((r["yolo_pred_set_class_only"] - r["gt_set"]) & CALL_IDS),
                axis=1,
            ).sum()
            any_multi = multiclass.apply(lambda r: bool(r[recovered_col] & r["gt_set"]), axis=1).sum()
            row[f"{prefix}_exact_multiclass_rate"] = float(exact_multi / len(multiclass)) if len(multiclass) else 0.0
            row[f"{prefix}_any_multiclass_rate"] = float(any_multi / len(multiclass)) if len(multiclass) else 0.0

        rows.append(row)

    by_fold = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    by_fold.to_csv(tables_dir / "set_aware_comparison_by_fold.csv", index=False)
    summary = build_variant_rows(by_fold, iou_thresholds)
    summary.to_csv(tables_dir / "set_aware_comparison_summary.csv", index=False)

    payload = {
        "n_folds": int(len(by_fold)),
        "n_all_snippets": int(by_fold["n_all_snippets"].sum()),
        "n_call_snippets": int(by_fold["n_call_snippets"].sum()),
        "gt_multiclass": int(by_fold["gt_multiclass"].sum()),
        "gt_multiclass_rate_calls": float(by_fold["gt_multiclass"].sum() / by_fold["n_call_snippets"].sum()),
        "metrics": summary.to_dict(orient="records"),
    }
    write_json(out_dir / "aggregate_set_aware_comparison.json", payload)
    write_latex_table(summary, tables_dir / "set_aware_comparison_table.tex")
    write_method_notes(out_dir / "README.md", args, iou_thresholds)
    plot_selected_metrics(summary, plots_dir / "set_aware_selected_metrics.png")
    plot_f_comparison(summary, plots_dir / "set_aware_f_comparison.png")
    print(f"Wrote set-aware comparison to: {out_dir}")
    print(summary[["label", "TCR_mean", "NMR_mean", "CMR_mean", "F_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
