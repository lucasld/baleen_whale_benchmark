#!/usr/bin/env python3
"""
Compare YOLO experiment runs over time (single fold or across folds).

This repo already writes per-run artifacts under:
  outputs/cnn_results/<RUN_ID>/<fold_*>/yolo/runs_det/<RUN_NAME>/
    - selected_threshold_summary.json
    - training_config.json
    - YOLO_<RUN_NAME>_metrics_confidence_sweep.csv  (optional, if conf_sweep=True)
    - plots/f_vs_confidence_top1.png               (per-run; optional)
    - plots/tcr_vs_nmr_top1.png                    (per-run; optional)

This tool aggregates those artifacts across many runs and produces:
  - comparison_runs.csv (one row per run, or per run×fold in across-fold mode)
  - summary_mean_std.csv (across-fold mode only)
  - comparison plots (bar + scatter + curve overlays)

Typical usage (single fold exploratory):
  python tools/compare_yolo_runs.py \
    --run-dir outputs/cnn_results/251008_160341 \
    --fold fold_BallenyIslands2015_noise_0.25

Across folds (confirmatory; compare several run names):
  python tools/compare_yolo_runs.py \
    --run-dir outputs/cnn_results/251008_160341 \
    --mode across-folds \
    --runs BS1,BS2,R0b
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for servers/SLURM
import matplotlib.pyplot as plt


SUMMARY_FILENAME = "selected_threshold_summary.json"
DEFAULT_CNN_METRICS_REL = Path("evaluation/all_metrics_by_fold_and_noise.csv")


@dataclass(frozen=True)
class RunRecord:
    run_name: str
    run_dir: Path
    fold: str
    summary_path: Path
    config_path: Optional[Path]
    sweep_csv_path: Optional[Path]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _discover_folds(run_dir: Path) -> List[Path]:
    folds = sorted([p for p in run_dir.glob("fold_*") if p.is_dir()])
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found under: {run_dir}")
    return folds


def _parse_noise_from_fold_name(fold_name: str) -> float | None:
    """
    Extract noise ratio from fold directory name.
    Expected: fold_<Site>_noise_<float>, e.g. fold_BallenyIslands2015_noise_0.25
    """
    m = re.search(r"_noise_([0-9]+(?:\.[0-9]+)?)", fold_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _discover_runs_det_dir(run_dir: Path, fold_name: str) -> Path:
    runs_det_dir = run_dir / fold_name / "yolo" / "runs_det"
    if not runs_det_dir.exists():
        raise FileNotFoundError(f"Missing runs_det directory: {runs_det_dir}")
    return runs_det_dir


def _discover_run_records_single_fold(
    run_dir: Path,
    fold: str,
    runs: Optional[List[str]],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> List[RunRecord]:
    runs_det_dir = _discover_runs_det_dir(run_dir, fold)
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    candidates = sorted([p for p in runs_det_dir.iterdir() if p.is_dir()])
    out: List[RunRecord] = []

    wanted = set(runs or [])
    for d in candidates:
        run_name = d.name
        if wanted and run_name not in wanted:
            continue
        if include_re and not include_re.search(run_name):
            continue
        if exclude_re and exclude_re.search(run_name):
            continue

        summary_path = d / SUMMARY_FILENAME
        if not summary_path.exists():
            # partially-finished runs (e.g., training-only) are ignored
            continue

        config_path = d / "training_config.json"
        if not config_path.exists():
            config_path = None

        sweep_csv = d / f"YOLO_{run_name}_metrics_confidence_sweep.csv"
        if not sweep_csv.exists():
            sweep_csv = None

        out.append(
            RunRecord(
                run_name=run_name,
                run_dir=run_dir,
                fold=fold,
                summary_path=summary_path,
                config_path=config_path,
                sweep_csv_path=sweep_csv,
            )
        )

    return out


def _discover_run_records_across_folds(
    run_dir: Path,
    runs: List[str],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> List[RunRecord]:
    folds = _discover_folds(run_dir)
    all_records: List[RunRecord] = []
    for fold_dir in folds:
        fold = fold_dir.name
        recs = _discover_run_records_single_fold(
            run_dir=run_dir,
            fold=fold,
            runs=runs,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
        )
        all_records.extend(recs)
    return all_records


def _load_cnn_metrics_table(run_dir: Path, cnn_metrics_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Load CNN per-fold metrics table (paper-style metrics).

    Expected columns (as produced by this repo):
      fold, noise, TCR, NMR, CMR, F, ACC, ...
    """
    path = cnn_metrics_csv if cnn_metrics_csv is not None else (run_dir / DEFAULT_CNN_METRICS_REL)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"fold", "noise", "TCR", "NMR", "CMR", "F", "ACC"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"CNN metrics CSV missing columns: {missing}. Path: {path}")
    return df


def _cnn_row_for_fold(
    run_dir: Path,
    fold: str,
    noise: float | None,
    cnn_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Build a comparison table row for the CNN baseline for a specific fold/noise.
    """
    subset = cnn_df[cnn_df["fold"] == fold]
    if noise is not None and "noise" in subset.columns:
        # be tolerant to float formatting
        subset = subset[np.isclose(subset["noise"].astype(float), float(noise))]
    if subset.empty:
        return None
    r = subset.iloc[0].to_dict()
    row: Dict[str, Any] = {
        "run_id": run_dir.name,
        "fold": fold,
        "run_name": "CNN",
        "label": "CNN baseline",
        "summary_path": "",
        "training_config_path": "",
        "sweep_csv_path": "",
        "strategy": "single-label",
        "selected_threshold": float("nan"),
        "test_best_threshold": float("nan"),
        "threshold_gap": float("nan"),
        # Map CNN metrics into the same namespace used by plotting helpers
        "test_selected_TCR": _safe_float(r.get("TCR")),
        "test_selected_NMR": _safe_float(r.get("NMR")),
        "test_selected_CMR": _safe_float(r.get("CMR")),
        "test_selected_F": _safe_float(r.get("F")),
        "test_selected_ACC": _safe_float(r.get("ACC")),
    }
    # Keep some extra CNN-only diagnostics if present
    for extra in ("FCR", "MACRO_RECALL", "metrics_str"):
        if extra in r:
            row[f"cnn_{extra}"] = r.get(extra)
    return row


def _extract_config_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tp = cfg.get("training_params", {}) or {}
    augs = cfg.get("augmentations", {}) or {}
    evalp = cfg.get("evaluation_params", {}) or {}

    def g(d: Dict[str, Any], k: str) -> Any:
        return d.get(k, None)

    return {
        "model_size": g(tp, "model_size"),
        "imgsz": g(tp, "imgsz"),
        "batch": g(tp, "batch"),
        "epochs": g(tp, "epochs"),
        "rect": g(tp, "rect"),
        "pretrained": g(tp, "pretrained"),
        "augs_preset": g(tp, "augs_preset"),
        "seed": g(tp, "seed"),
        "deterministic": g(tp, "deterministic"),
        "weights": g(tp, "weights"),
        "eval_conf": g(evalp, "eval_conf"),
        "conf_sweep": g(evalp, "conf_sweep"),
        "conf_sweep_min": g(evalp, "conf_sweep_min"),
        "conf_sweep_max": g(evalp, "conf_sweep_max"),
        "conf_sweep_step": g(evalp, "conf_sweep_step"),
        # a few raw augmentation knobs for debugging (optional)
        "mosaic": g(augs, "mosaic"),
        "fliplr": g(augs, "fliplr"),
        "hsv_h": g(augs, "hsv_h"),
        "hsv_s": g(augs, "hsv_s"),
        "hsv_v": g(augs, "hsv_v"),
    }


def _row_from_summary(
    record: RunRecord,
    summary: Dict[str, Any],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_id": record.run_dir.name,
        "fold": record.fold,
        "run_name": record.run_name,
        "summary_path": str(record.summary_path),
        "training_config_path": str(record.config_path) if record.config_path else "",
        "sweep_csv_path": str(record.sweep_csv_path) if record.sweep_csv_path else "",
        "strategy": summary.get("strategy"),
        "selected_threshold": _safe_float(summary.get("selected_threshold")),
        "test_best_threshold": _safe_float(summary.get("test_best_threshold")),
    }
    row["threshold_gap"] = abs(row["selected_threshold"] - row["test_best_threshold"])

    def add_metrics(prefix: str, d: Dict[str, Any]) -> None:
        for k in ("TCR", "NMR", "CMR", "F", "ACC"):
            if k in d:
                row[f"{prefix}_{k}"] = _safe_float(d.get(k))

    add_metrics("val", summary.get("validation_metrics", {}) or {})
    add_metrics("test_selected", summary.get("test_metrics_at_selected_threshold", {}) or {})
    add_metrics("test_best", summary.get("test_best_metrics", {}) or {})

    if config is not None:
        row.update(_extract_config_fields(config))

    # human-friendly label used in plots
    if config is not None:
        tp = config.get("training_params", {}) or {}
        label = (
            f"{record.run_name}"
            f" | yolo12{tp.get('model_size', '?')}"
            f" imgsz={tp.get('imgsz', '?')}"
            f" bs={tp.get('batch', '?')}"
            f" rect={tp.get('rect', '?')}"
            f" augs={tp.get('augs_preset', '?')}"
            f" pre={tp.get('pretrained', '?')}"
        )
    else:
        label = record.run_name
    row["label"] = label
    return row


def _load_dataframe(records: List[RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        summary = _read_json(rec.summary_path)
        cfg = _read_json(rec.config_path) if rec.config_path else None
        rows.append(_row_from_summary(rec, summary, cfg))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # stable ordering: by run_name then fold
    df = df.sort_values(["run_name", "fold"]).reset_index(drop=True)
    return df


def _plot_bar_metrics(df: pd.DataFrame, out_dir: Path, metric_prefix: str) -> None:
    metrics = ["TCR", "NMR", "CMR", "F"]
    cols = [f"{metric_prefix}_{m}" for m in metrics]
    if any(c not in df.columns for c in cols):
        return

    # Use run_name on x; for single-fold mode this is unique. For across-fold summary, caller passes per-run summary.
    x = np.arange(len(df))
    labels = df["run_name"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, m in zip(axes, metrics):
        y = df[f"{metric_prefix}_{m}"].to_numpy(dtype=float)
        ax.bar(x, y, color="#4C78A8", alpha=0.9)
        ax.set_title(f"{metric_prefix}: {m}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_{metric_prefix}_TCR_NMR_CMR_F.png", dpi=200)
    plt.close(fig)


def _plot_scatter_tcr_vs_nmr(df: pd.DataFrame, out_dir: Path, metric_prefix: str) -> None:
    tcr_col = f"{metric_prefix}_TCR"
    nmr_col = f"{metric_prefix}_NMR"
    if tcr_col not in df.columns or nmr_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    x = df[nmr_col].to_numpy(dtype=float)
    y = df[tcr_col].to_numpy(dtype=float)
    ax.scatter(x, y, s=60, alpha=0.85, color="#F58518", edgecolors="black", linewidth=0.5)
    for _, r in df.iterrows():
        ax.annotate(
            str(r["run_name"]),
            (float(r[nmr_col]), float(r[tcr_col])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("NMR")
    ax.set_ylabel("TCR")
    ax.set_title(f"TCR vs NMR ({metric_prefix})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_tcr_vs_nmr_{metric_prefix}.png", dpi=200)
    plt.close(fig)


def _load_sweep_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # enforce expected columns (threshold, strategy, F, TCR, NMR)
    for c in ("threshold", "strategy"):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in sweep CSV: {path}")
    return df


def _plot_overlay_f_vs_conf(records: List[RunRecord], df_runs: pd.DataFrame, out_dir: Path) -> None:
    # map run_name -> selected_threshold for marker
    sel = {r["run_name"]: float(r["selected_threshold"]) for _, r in df_runs.iterrows() if "selected_threshold" in r}

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = 0
    for rec in records:
        if rec.sweep_csv_path is None:
            continue
        try:
            sdf = _load_sweep_df(rec.sweep_csv_path)
        except Exception:
            continue
        sdf = sdf[sdf["strategy"] == "top1"].sort_values("threshold")
        if sdf.empty or "F" not in sdf.columns:
            continue
        ax.plot(sdf["threshold"], sdf["F"], linewidth=1.6, label=rec.run_name)
        if rec.run_name in sel and np.isfinite(sel[rec.run_name]):
            thr = sel[rec.run_name]
            # mark selected point if present in sweep grid; otherwise just vline
            ax.axvline(thr, linestyle="--", alpha=0.15)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F-score")
    ax.set_title("F vs confidence (Top-1) — overlay")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_f_vs_confidence_top1.png", dpi=200)
    plt.close(fig)


def _plot_overlay_tcr_vs_nmr(records: List[RunRecord], df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    plotted = 0
    for rec in records:
        if rec.sweep_csv_path is None:
            continue
        try:
            sdf = _load_sweep_df(rec.sweep_csv_path)
        except Exception:
            continue
        sdf = sdf[sdf["strategy"] == "top1"].sort_values("threshold")
        if sdf.empty or "TCR" not in sdf.columns or "NMR" not in sdf.columns:
            continue
        ax.plot(sdf["NMR"], sdf["TCR"], marker="o", markersize=2.5, linewidth=1.2, label=rec.run_name)
        plotted += 1

    # Add CNN baseline as a single scatter point
    cnn_row = df[df["run_name"] == "CNN"]
    if not cnn_row.empty:
        cnn_tcr = cnn_row["test_selected_TCR"].iloc[0]
        cnn_nmr = cnn_row["test_selected_NMR"].iloc[0]
        if np.isfinite(cnn_tcr) and np.isfinite(cnn_nmr):
            # Academic star marker for baseline, reduced size for professional look
            ax.scatter(cnn_nmr, cnn_tcr, s=90, color="#E41A1C", marker="*", edgecolors="black", linewidth=0.5, label="CNN baseline", zorder=10)
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("NMR")
    ax.set_ylabel("TCR")
    ax.set_title("TCR vs NMR — operating curves (Top-1) overlay")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_tcr_vs_nmr_top1.png", dpi=200)
    plt.close(fig)


def _mean_std_summary(df: pd.DataFrame, by: str, metric_prefix: str) -> pd.DataFrame:
    metrics = ["TCR", "NMR", "CMR", "F", "ACC"]
    cols = [f"{metric_prefix}_{m}" for m in metrics if f"{metric_prefix}_{m}" in df.columns]
    if not cols:
        return pd.DataFrame()
    grp = df.groupby(by, dropna=False)
    out = grp[cols].agg(["mean", "std"]).reset_index()
    # flatten multi-index columns
    out.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in out.columns.to_flat_index()]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLO runs across experiments.")
    p.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/cnn_results/<run_id>.")
    p.add_argument(
        "--mode",
        choices=["single-fold", "across-folds"],
        default="single-fold",
        help="single-fold compares runs under one fold; across-folds aggregates runs across all folds.",
    )
    p.add_argument("--fold", type=str, default=None, help="Fold name for single-fold mode.")
    p.add_argument(
        "--runs",
        type=str,
        default="",
        help="Comma-separated run names to include (default: auto-discover all completed runs).",
    )
    p.add_argument(
        "--include-regex",
        type=str,
        default="",
        help="Only include run names matching this regex (applied after --runs filter).",
    )
    p.add_argument(
        "--exclude-regex",
        type=str,
        default="",
        help="Exclude run names matching this regex.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <fold>/yolo/comparisons/latest (single-fold) or <run-dir>/yolo/comparisons/latest (across-folds).",
    )
    p.add_argument(
        "--cnn-metrics-csv",
        type=Path,
        default=None,
        help="Optional path to CNN metrics CSV (default: <run-dir>/evaluation/all_metrics_by_fold_and_noise.csv).",
    )
    p.add_argument(
        "--no-cnn",
        action="store_true",
        help="Disable adding CNN baseline rows to the comparison outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    runs = [r.strip() for r in args.runs.split(",") if r.strip()] or None
    include_regex = args.include_regex.strip() or None
    exclude_regex = args.exclude_regex.strip() or None

    if args.mode == "single-fold":
        if not args.fold:
            raise SystemExit("--fold is required in single-fold mode")
        fold = args.fold
        records = _discover_run_records_single_fold(run_dir, fold, runs, include_regex, exclude_regex)
        if args.out_dir is None:
            out_dir = run_dir / fold / "yolo" / "comparisons" / "latest"
        else:
            out_dir = args.out_dir
    else:
        if runs is None:
            raise SystemExit("--runs is required in across-folds mode (list the run names/IDs to compare)")
        records = _discover_run_records_across_folds(run_dir, runs, include_regex, exclude_regex)
        if args.out_dir is None:
            out_dir = run_dir / "yolo" / "comparisons" / "latest"
        else:
            out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(records)
    if df.empty:
        print("[compare_yolo_runs] No completed runs found (missing selected_threshold_summary.json).")
        return

    # Optionally append CNN baseline rows (where they make sense)
    if not args.no_cnn:
        cnn_df = _load_cnn_metrics_table(run_dir, args.cnn_metrics_csv)
        if cnn_df is not None:
            cnn_rows: List[Dict[str, Any]] = []
            if args.mode == "single-fold":
                fold = args.fold  # type: ignore[assignment]
                noise = _parse_noise_from_fold_name(str(fold))
                row = _cnn_row_for_fold(run_dir, str(fold), noise, cnn_df)
                if row is not None:
                    cnn_rows.append(row)
            else:
                # across folds: add CNN row for every fold we touched (if present in CNN table)
                for fold_name in sorted({rec.fold for rec in records}):
                    noise = _parse_noise_from_fold_name(fold_name)
                    row = _cnn_row_for_fold(run_dir, fold_name, noise, cnn_df)
                    if row is not None:
                        cnn_rows.append(row)
            if cnn_rows:
                df = pd.concat([df, pd.DataFrame(cnn_rows)], ignore_index=True).sort_values(["run_name", "fold"]).reset_index(drop=True)

    # Persist raw comparison table
    csv_path = out_dir / "comparison_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"[compare_yolo_runs] Wrote: {csv_path}")

    # Single-fold plots: one row per run
    if args.mode == "single-fold":
        # de-dupe by run_name just in case
        df_one = df.drop_duplicates(subset=["run_name"]).reset_index(drop=True)
        _plot_bar_metrics(df_one, out_dir, metric_prefix="test_selected")
        _plot_scatter_tcr_vs_nmr(df_one, out_dir, metric_prefix="test_selected")
        # Confidence-sweep overlays only make sense for YOLO runs
        _plot_overlay_f_vs_conf(records, df_one[df_one["run_name"] != "CNN"], out_dir)
        _plot_overlay_tcr_vs_nmr(records, df_one, out_dir)
        return

    # Across-folds: compute mean/std across folds for each run_name
    summary = _mean_std_summary(df, by="run_name", metric_prefix="test_selected")
    if not summary.empty:
        summary_path = out_dir / "summary_mean_std.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[compare_yolo_runs] Wrote: {summary_path}")
        # plot mean bars for key metrics (mean only)
        df_mean = summary.copy()
        # rename columns into the same format expected by plotting helpers
        for m in ("TCR", "NMR", "CMR", "F", "ACC"):
            c = f"test_selected_{m}_mean"
            if c in df_mean.columns:
                df_mean[f"test_selected_{m}"] = df_mean[c]
        _plot_bar_metrics(df_mean, out_dir, metric_prefix="test_selected")
        _plot_scatter_tcr_vs_nmr(df_mean, out_dir, metric_prefix="test_selected")


if __name__ == "__main__":
    main()


