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
from yolo_aggregation import (
    aggregate_selected_threshold,
    aggregate_sweeps,
    default_output_layout,
    discover_fold_run_artifacts,
    ensure_output_layout,
    is_cache_valid,
    load_selected_threshold_rows,
    load_sweep_rows,
    write_cache_metadata,
)


SUMMARY_FILENAME = "selected_threshold_summary.json"
DEFAULT_CNN_METRICS_REL = Path("evaluation/all_metrics_by_fold_and_noise.csv")
DEFAULT_STYLE_REL = Path("experiments/plot_style.json")
DEFAULT_RUN_COLORS: Dict[str, str] = {
    "CNN": "#222222",
    "BS1": "#00441B",
    "BS2": "#5AAE61",
    "R0a": "#40004B",
    "R0b": "#9970AB",
    "A1": "#084594",
    "A2": "#4292C6",
    "A3": "#C6DBEF",
    "B1": "#01665E",
    "B2": "#5AB4AC",
    "B3": "#C7EAE5",
    "C1": "#8C510A",
    "C2": "#DFC27D",
    "D2": "#D73027",
}
FALLBACK_COLOR_CYCLE = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


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


def _normalize_color_hex(color: Any) -> Optional[str]:
    if not isinstance(color, str):
        return None
    c = color.strip()
    if not c:
        return None
    if not c.startswith("#"):
        c = f"#{c}"
    if len(c) != 7:
        return None
    try:
        int(c[1:], 16)
    except ValueError:
        return None
    return c.upper()


def _load_style_run_colors(style_file: Optional[Path]) -> Dict[str, str]:
    """
    Load run color mapping from JSON style file.

    Expected schema:
    {
      "run_colors": {"B1": "#01665E", ...}
    }
    """
    run_colors = dict(DEFAULT_RUN_COLORS)
    if style_file is None:
        return run_colors
    if not style_file.exists():
        return run_colors
    try:
        payload = json.loads(style_file.read_text())
    except Exception as e:
        print(f"[compare_yolo_runs] Warning: failed to parse style file {style_file}: {e}")
        return run_colors
    if not isinstance(payload, dict):
        return run_colors
    user_map = payload.get("run_colors", {})
    if not isinstance(user_map, dict):
        return run_colors
    for k, v in user_map.items():
        if not isinstance(k, str):
            continue
        color = _normalize_color_hex(v)
        if color is not None:
            run_colors[k] = color
    return run_colors


def _color_for_run(run_name: str, run_colors: Dict[str, str]) -> str:
    c = run_colors.get(run_name)
    if c:
        return c
    idx = abs(hash(run_name)) % len(FALLBACK_COLOR_CYCLE)
    return FALLBACK_COLOR_CYCLE[idx]


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


def _plot_bar_metrics(df: pd.DataFrame, out_dir: Path, metric_prefix: str, run_colors: Dict[str, str]) -> None:
    metrics = ["TCR", "NMR", "CMR", "F"]
    cols = [f"{metric_prefix}_{m}" for m in metrics]
    if any(c not in df.columns for c in cols):
        return

    # Use run_name on x; for single-fold mode this is unique. For across-fold summary, caller passes per-run summary.
    x = np.arange(len(df))
    labels = df["run_name"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    bar_colors = [_color_for_run(rn, run_colors) for rn in labels]
    for ax, m in zip(axes, metrics):
        y = df[f"{metric_prefix}_{m}"].to_numpy(dtype=float)
        ax.bar(x, y, color=bar_colors, alpha=0.9)
        ax.set_title(f"{metric_prefix}: {m}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_{metric_prefix}_TCR_NMR_CMR_F.png", dpi=200)
    plt.close(fig)


def _plot_scatter_tcr_vs_nmr(df: pd.DataFrame, out_dir: Path, metric_prefix: str, run_colors: Dict[str, str]) -> None:
    tcr_col = f"{metric_prefix}_TCR"
    nmr_col = f"{metric_prefix}_NMR"
    if tcr_col not in df.columns or nmr_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    for _, r in df.iterrows():
        run_name = str(r["run_name"])
        c = _color_for_run(run_name, run_colors)
        x = float(r[nmr_col])
        y = float(r[tcr_col])
        ax.scatter(x, y, s=60, alpha=0.9, color=c, edgecolors="black", linewidth=0.5)
        ax.annotate(
            run_name,
            (x, y),
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


def _plot_overlay_f_vs_conf(
    records: List[RunRecord],
    df_runs: pd.DataFrame,
    out_dir: Path,
    run_colors: Dict[str, str],
    annotate_confidence: bool = True,
    annotate_every: int = 1,
) -> None:
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
        color = _color_for_run(rec.run_name, run_colors)
        ax.plot(sdf["threshold"], sdf["F"], marker="o", markersize=2.8, linewidth=1.6, label=rec.run_name, color=color)
        if annotate_confidence:
            step = max(1, int(annotate_every))
            for i, (_, row) in enumerate(sdf.iterrows()):
                if i % step != 0:
                    continue
                thr = float(row["threshold"])
                f_val = float(row["F"])
                ax.annotate(
                    f"{thr:.2f}",
                    (thr, f_val),
                    textcoords="offset points",
                    xytext=(3, 2),
                    fontsize=6,
                    color=color,
                    alpha=0.9,
                )
        if rec.run_name in sel and np.isfinite(sel[rec.run_name]):
            thr = sel[rec.run_name]
            # mark selected point if present in sweep grid; otherwise just vline
            ax.axvline(thr, linestyle="--", alpha=0.2, color=color)
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
    cnn_row = df_runs[df_runs["run_name"] == "CNN"]
    if not cnn_row.empty and "test_selected_F" in cnn_row.columns:
        cnn_f = float(cnn_row["test_selected_F"].iloc[0])
        if np.isfinite(cnn_f):
            ax.axhline(
                cnn_f,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                color=_color_for_run("CNN", run_colors),
                label="CNN baseline",
            )
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_f_vs_confidence_top1.png", dpi=200)
    plt.close(fig)


def _plot_overlay_tcr_vs_nmr(
    records: List[RunRecord],
    df: pd.DataFrame,
    out_dir: Path,
    run_colors: Dict[str, str],
    annotate_confidence: bool = True,
    annotate_every: int = 1,
) -> None:
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
        color = _color_for_run(rec.run_name, run_colors)
        ax.plot(sdf["NMR"], sdf["TCR"], marker="o", markersize=2.5, linewidth=1.2, label=rec.run_name, color=color)
        if annotate_confidence:
            step = max(1, int(annotate_every))
            for i, (_, row) in enumerate(sdf.iterrows()):
                if i % step != 0:
                    continue
                nmr = float(row["NMR"])
                tcr = float(row["TCR"])
                thr = float(row["threshold"])
                ax.annotate(
                    f"{thr:.2f}",
                    (nmr, tcr),
                    textcoords="offset points",
                    xytext=(3, 2),
                    fontsize=6,
                    color=color,
                    alpha=0.9,
                )
        plotted += 1

    # Add CNN baseline as a single scatter point
    cnn_row = df[df["run_name"] == "CNN"]
    if not cnn_row.empty:
        cnn_tcr = cnn_row["test_selected_TCR"].iloc[0]
        cnn_nmr = cnn_row["test_selected_NMR"].iloc[0]
        if np.isfinite(cnn_tcr) and np.isfinite(cnn_nmr):
            # Academic star marker for baseline
            ax.scatter(
                cnn_nmr,
                cnn_tcr,
                s=90,
                color=_color_for_run("CNN", run_colors),
                marker="*",
                edgecolors="black",
                linewidth=0.5,
                label="CNN baseline",
                zorder=10,
            )
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
    p.add_argument(
        "--style-file",
        type=Path,
        default=DEFAULT_STYLE_REL,
        help=f"JSON style file with run colors. Default: {DEFAULT_STYLE_REL}",
    )
    p.add_argument(
        "--no-annotate-confidence",
        action="store_true",
        help="Disable confidence-threshold text annotation on overlay curve points.",
    )
    p.add_argument(
        "--annotate-every",
        type=int,
        default=1,
        help="Annotate every Nth point on overlays (default: 1 = every point).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    runs = [r.strip() for r in args.runs.split(",") if r.strip()] or None
    include_regex = args.include_regex.strip() or None
    exclude_regex = args.exclude_regex.strip() or None
    run_colors = _load_style_run_colors(args.style_file)
    annotate_confidence = not args.no_annotate_confidence
    annotate_every = max(1, int(args.annotate_every))

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
        _plot_bar_metrics(df_one, out_dir, metric_prefix="test_selected", run_colors=run_colors)
        _plot_scatter_tcr_vs_nmr(df_one, out_dir, metric_prefix="test_selected", run_colors=run_colors)
        # Confidence-sweep overlays only make sense for YOLO runs
        _plot_overlay_f_vs_conf(
            records,
            df_one[df_one["run_name"] != "CNN"],
            out_dir,
            run_colors=run_colors,
            annotate_confidence=annotate_confidence,
            annotate_every=annotate_every,
        )
        _plot_overlay_tcr_vs_nmr(
            records,
            df_one,
            out_dir,
            run_colors=run_colors,
            annotate_confidence=annotate_confidence,
            annotate_every=annotate_every,
        )
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
        _plot_bar_metrics(df_mean, out_dir, metric_prefix="test_selected", run_colors=run_colors)
        _plot_scatter_tcr_vs_nmr(df_mean, out_dir, metric_prefix="test_selected", run_colors=run_colors)


def _plot_point_metrics_with_uncertainty(
    df_agg_points: pd.DataFrame,
    out_dir: Path,
    run_colors: Dict[str, str],
    band_kind: str,
) -> None:
    if df_agg_points.empty:
        return
    metric_names = ["TCR", "NMR", "CMR", "F"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    x = np.arange(len(df_agg_points))
    labels = df_agg_points["run_name"].astype(str).tolist()
    colors = [_color_for_run(rn, run_colors) for rn in labels]
    for ax, m in zip(axes, metric_names):
        mean_col = f"test_selected_{m}_mean"
        low_col = f"test_selected_{m}_{band_kind}_low"
        high_col = f"test_selected_{m}_{band_kind}_high"
        if mean_col not in df_agg_points.columns:
            continue
        y = df_agg_points[mean_col].to_numpy(dtype=float)
        ax.bar(x, y, color=colors, alpha=0.85)
        if low_col in df_agg_points.columns and high_col in df_agg_points.columns:
            low = df_agg_points[low_col].to_numpy(dtype=float)
            high = df_agg_points[high_col].to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(y - low, 0.0), np.maximum(high - y, 0.0)])
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_title(f"{m} ({band_kind})")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_test_selected_metrics_{band_kind}.png", dpi=200)
    plt.close(fig)


def _plot_sweep_overlay_with_band(
    df_sweep_agg: pd.DataFrame,
    out_dir: Path,
    run_colors: Dict[str, str],
    metric: str,
    band_kind: str,
    cnn_summary_row: Optional[Dict[str, Any]] = None,
) -> None:
    if df_sweep_agg.empty:
        return
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_{band_kind}_low"
    high_col = f"{metric}_{band_kind}_high"
    if mean_col not in df_sweep_agg.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for run_name, sdf in df_sweep_agg.groupby("run_name"):
        sdf = sdf.sort_values("threshold")
        color = _color_for_run(str(run_name), run_colors)
        x = sdf["threshold"].to_numpy(dtype=float)
        y = sdf[mean_col].to_numpy(dtype=float)
        ax.plot(x, y, lw=1.8, color=color, label=str(run_name))
        if low_col in sdf.columns and high_col in sdf.columns:
            n = int(sdf["n"].max()) if "n" in sdf.columns else 1
            if n > 1:
                low = sdf[low_col].to_numpy(dtype=float)
                high = sdf[high_col].to_numpy(dtype=float)
                ax.fill_between(x, low, high, alpha=0.18, color=color)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs confidence ({band_kind})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if cnn_summary_row is not None:
        mean_key = f"test_selected_{metric}_mean"
        low_key = f"test_selected_{metric}_{band_kind}_low"
        high_key = f"test_selected_{metric}_{band_kind}_high"
        y = cnn_summary_row.get(mean_key)
        if y is not None and np.isfinite(float(y)):
            yv = float(y)
            ax.axhline(
                yv,
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
                color=_color_for_run("CNN", run_colors),
                label="CNN baseline",
            )
            n = int(cnn_summary_row.get("n", 1))
            yl = cnn_summary_row.get(low_key)
            yh = cnn_summary_row.get(high_key)
            if n > 1 and yl is not None and yh is not None and np.isfinite(float(yl)) and np.isfinite(float(yh)):
                ax.fill_between(
                    [0.0, 1.0],
                    [float(yl), float(yl)],
                    [float(yh), float(yh)],
                    alpha=0.12,
                    color=_color_for_run("CNN", run_colors),
                )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"overlay_{metric.lower()}_vs_confidence_{band_kind}.png", dpi=200)
    plt.close(fig)


def _plot_operating_curve_by_fold(
    df_sweep: pd.DataFrame,
    out_dir: Path,
    run_name: str,
    cnn_summary_row: Optional[Dict[str, Any]] = None,
) -> None:
    if df_sweep.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for fold, sdf in df_sweep.groupby("fold"):
        sdf = sdf.sort_values("threshold")
        if "NMR" not in sdf.columns or "TCR" not in sdf.columns:
            continue
        ax.plot(
            sdf["NMR"],
            sdf["TCR"],
            marker="o",
            markersize=2.6,
            linewidth=1.3,
            alpha=0.8,
            label=str(fold),
        )
    ax.set_xlabel("NMR")
    ax.set_ylabel("TCR")
    ax.set_title(f"{run_name}: per-fold operating curves")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if cnn_summary_row is not None:
        x = cnn_summary_row.get("test_selected_NMR_mean")
        y = cnn_summary_row.get("test_selected_TCR_mean")
        if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y)):
            xv = float(x)
            yv = float(y)
            ax.scatter(
                xv,
                yv,
                s=95,
                marker="*",
                color=_color_for_run("CNN", DEFAULT_RUN_COLORS),
                edgecolors="black",
                linewidth=0.5,
                label="CNN baseline",
                zorder=10,
            )
            n = int(cnn_summary_row.get("n", 1))
            if n > 1:
                xlo = cnn_summary_row.get("test_selected_NMR_std_low")
                xhi = cnn_summary_row.get("test_selected_NMR_std_high")
                ylo = cnn_summary_row.get("test_selected_TCR_std_low")
                yhi = cnn_summary_row.get("test_selected_TCR_std_high")
                if all(v is not None and np.isfinite(float(v)) for v in (xlo, xhi, ylo, yhi)):
                    ax.errorbar(
                        [xv],
                        [yv],
                        xerr=[[max(xv - float(xlo), 0.0)], [max(float(xhi) - xv, 0.0)]],
                        yerr=[[max(yv - float(ylo), 0.0)], [max(float(yhi) - yv, 0.0)]],
                        fmt="none",
                        ecolor=_color_for_run("CNN", DEFAULT_RUN_COLORS),
                        capsize=3,
                        lw=1.2,
                        alpha=0.8,
                    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_per_fold_tcr_vs_nmr.png", dpi=200)
    plt.close(fig)


def _plot_f_curve_by_fold(df_sweep: pd.DataFrame, out_dir: Path, run_name: str) -> None:
    if df_sweep.empty or "F" not in df_sweep.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for fold, sdf in df_sweep.groupby("fold"):
        sdf = sdf.sort_values("threshold")
        ax.plot(
            sdf["threshold"],
            sdf["F"],
            marker="o",
            markersize=2.6,
            linewidth=1.3,
            alpha=0.8,
            label=str(fold),
        )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F")
    ax.set_title(f"{run_name}: per-fold F vs confidence")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_per_fold_f_vs_confidence.png", dpi=200)
    plt.close(fig)


def _plot_metric_curve_by_fold(
    df_sweep: pd.DataFrame,
    out_dir: Path,
    run_name: str,
    metric: str,
    cnn_summary_row: Optional[Dict[str, Any]] = None,
) -> None:
    if df_sweep.empty or metric not in df_sweep.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for fold, sdf in df_sweep.groupby("fold"):
        sdf = sdf.sort_values("threshold")
        ax.plot(
            sdf["threshold"],
            sdf[metric],
            marker="o",
            markersize=2.6,
            linewidth=1.3,
            alpha=0.8,
            label=str(fold),
        )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(metric)
    ax.set_title(f"{run_name}: per-fold {metric} vs confidence")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if cnn_summary_row is not None:
        mean_key = f"test_selected_{metric}_mean"
        y = cnn_summary_row.get(mean_key)
        if y is not None and np.isfinite(float(y)):
            yv = float(y)
            ax.axhline(
                yv,
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
                color=_color_for_run("CNN", DEFAULT_RUN_COLORS),
                label="CNN baseline",
            )
            n = int(cnn_summary_row.get("n", 1))
            low_key = f"test_selected_{metric}_std_low"
            high_key = f"test_selected_{metric}_std_high"
            yl = cnn_summary_row.get(low_key)
            yh = cnn_summary_row.get(high_key)
            if n > 1 and yl is not None and yh is not None and np.isfinite(float(yl)) and np.isfinite(float(yh)):
                ax.fill_between(
                    [0.0, 1.0],
                    [float(yl), float(yl)],
                    [float(yh), float(yh)],
                    alpha=0.10,
                    color=_color_for_run("CNN", DEFAULT_RUN_COLORS),
                )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    ax_metric = metric.lower()
    fig.savefig(out_dir / f"{run_name}_per_fold_{ax_metric}_vs_confidence.png", dpi=200)
    plt.close(fig)


def _plot_point_metrics_by_fold(df_points: pd.DataFrame, out_dir: Path, run_name: str) -> None:
    if df_points.empty:
        return
    metric_names = ["TCR", "NMR", "CMR", "F"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    x = np.arange(len(df_points))
    labels = df_points["fold"].astype(str).tolist()
    for ax, m in zip(axes, metric_names):
        c = f"test_selected_{m}"
        if c not in df_points.columns:
            continue
        y = df_points[c].to_numpy(dtype=float)
        ax.bar(x, y, alpha=0.85)
        ax.set_title(f"{run_name}: selected {m} by fold")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_selected_metrics_by_fold.png", dpi=200)
    plt.close(fig)


def _plot_selected_scatter_by_fold(df_points: pd.DataFrame, out_dir: Path, run_name: str) -> None:
    tcr_col = "test_selected_TCR"
    nmr_col = "test_selected_NMR"
    if df_points.empty or tcr_col not in df_points.columns or nmr_col not in df_points.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for _, r in df_points.iterrows():
        x = float(r[nmr_col])
        y = float(r[tcr_col])
        fold = str(r.get("fold", "fold"))
        ax.scatter(x, y, s=55, alpha=0.9, edgecolors="black", linewidth=0.5)
        ax.annotate(fold, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("NMR")
    ax.set_ylabel("TCR")
    ax.set_title(f"{run_name}: selected-point TCR vs NMR by fold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_selected_scatter_tcr_vs_nmr_by_fold.png", dpi=200)
    plt.close(fig)


def _plot_operating_curve_agg_with_band(
    df_sweep_agg: pd.DataFrame,
    out_dir: Path,
    run_colors: Dict[str, str],
    band_kind: str,
    cnn_summary_row: Optional[Dict[str, Any]] = None,
    annotate_every: int = 2,
) -> None:
    if df_sweep_agg.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for run_name, sdf in df_sweep_agg.groupby("run_name"):
        sdf = sdf.sort_values("threshold")
        if "NMR_mean" not in sdf.columns or "TCR_mean" not in sdf.columns:
            continue
        color = _color_for_run(str(run_name), run_colors)
        x = sdf["NMR_mean"].to_numpy(dtype=float)
        y = sdf["TCR_mean"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=1.8, marker="o", markersize=3.0, color=color, label=str(run_name))
        step = max(1, int(annotate_every))
        for i, (_, row) in enumerate(sdf.iterrows()):
            if i % step != 0:
                continue
            nmr_val = float(row["NMR_mean"])
            tcr_val = float(row["TCR_mean"])
            thr = float(row["threshold"])
            ax.annotate(
                f"{thr:.2f}",
                (nmr_val, tcr_val),
                textcoords="offset points",
                xytext=(3, 2),
                fontsize=6,
                color=color,
                alpha=0.9,
            )
        n = int(sdf["n"].max()) if "n" in sdf.columns else 1
        if n > 1:
            x_low_col = f"NMR_{band_kind}_low"
            x_high_col = f"NMR_{band_kind}_high"
            y_low_col = f"TCR_{band_kind}_low"
            y_high_col = f"TCR_{band_kind}_high"
            if all(c in sdf.columns for c in (x_low_col, x_high_col, y_low_col, y_high_col)):
                x_low = sdf[x_low_col].to_numpy(dtype=float)
                x_high = sdf[x_high_col].to_numpy(dtype=float)
                y_low = sdf[y_low_col].to_numpy(dtype=float)
                y_high = sdf[y_high_col].to_numpy(dtype=float)
                ax.fill_betweenx(y, x_low, x_high, alpha=0.06, color=color)
                ax.fill_between(x, y_low, y_high, alpha=0.06, color=color)
    ax.set_xlabel("NMR")
    ax.set_ylabel("TCR")
    ax.set_title(f"TCR vs NMR operating curves ({band_kind})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if cnn_summary_row is not None:
        x = cnn_summary_row.get("test_selected_NMR_mean")
        y = cnn_summary_row.get("test_selected_TCR_mean")
        if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y)):
            xv = float(x)
            yv = float(y)
            ax.scatter(
                xv,
                yv,
                s=100,
                marker="*",
                color=_color_for_run("CNN", run_colors),
                edgecolors="black",
                linewidth=0.6,
                label="CNN baseline",
                zorder=11,
            )
            n = int(cnn_summary_row.get("n", 1))
            if n > 1:
                x_low = cnn_summary_row.get(f"test_selected_NMR_{band_kind}_low")
                x_high = cnn_summary_row.get(f"test_selected_NMR_{band_kind}_high")
                y_low = cnn_summary_row.get(f"test_selected_TCR_{band_kind}_low")
                y_high = cnn_summary_row.get(f"test_selected_TCR_{band_kind}_high")
                if all(v is not None and np.isfinite(float(v)) for v in (x_low, x_high, y_low, y_high)):
                    ax.errorbar(
                        [xv],
                        [yv],
                        xerr=[[max(xv - float(x_low), 0.0)], [max(float(x_high) - xv, 0.0)]],
                        yerr=[[max(yv - float(y_low), 0.0)], [max(float(y_high) - yv, 0.0)]],
                        fmt="none",
                        ecolor=_color_for_run("CNN", run_colors),
                        capsize=3,
                        lw=1.2,
                        alpha=0.85,
                    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"overlay_tcr_vs_nmr_{band_kind}.png", dpi=200)
    plt.close(fig)


def parse_args_v2() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLO runs across experiments.")
    p.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/cnn_results/<run_id>.")
    p.add_argument(
        "--mode",
        choices=["single-fold", "across-folds"],
        default="single-fold",
        help="single-fold compares within one fold OR one run across folds; across-folds computes per-run averages.",
    )
    p.add_argument("--fold", type=str, default=None, help="Fold name for explicit single-fold mode.")
    p.add_argument("--runs", type=str, default="", help="Comma-separated run names to include.")
    p.add_argument("--include-regex", type=str, default="", help="Include run names matching regex.")
    p.add_argument("--exclude-regex", type=str, default="", help="Exclude run names matching regex.")
    p.add_argument(
        "--aggregate-folds",
        action="store_true",
        help="Aggregate over folds even in single-fold mode when a single run has multiple folds.",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Root output directory.")
    p.add_argument("--cnn-metrics-csv", type=Path, default=None, help="Optional CNN metrics CSV path.")
    p.add_argument("--no-cnn", action="store_true", help="Disable adding CNN baseline rows.")
    p.add_argument(
        "--cnn-on-curves",
        action="store_true",
        help="Also render CNN baseline on confidence-sweep and operating-curve plots.",
    )
    p.add_argument("--style-file", type=Path, default=DEFAULT_STYLE_REL, help="JSON style file with run colors.")
    p.add_argument("--no-annotate-confidence", action="store_true", help="Disable confidence text annotations.")
    p.add_argument("--annotate-every", type=int, default=1, help="Annotate every Nth confidence point.")
    return p.parse_args()


def main_v2() -> None:
    args = parse_args_v2()
    run_dir = args.run_dir
    run_colors = _load_style_run_colors(args.style_file)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not runs:
        raise SystemExit("--runs is required for the enhanced workflow.")

    # Decide output root.
    if args.out_dir is not None:
        out_root = args.out_dir
    elif args.mode == "single-fold" and args.fold:
        out_root = run_dir / args.fold / "yolo" / "analysis" / "latest"
    elif args.mode == "single-fold" and len(runs) == 1:
        out_root = run_dir / "yolo" / "analysis" / runs[0] / "latest"
    else:
        out_root = run_dir / "yolo" / "analysis" / "latest"

    layout = default_output_layout(out_root)
    ensure_output_layout(layout)

    fold_subset = [args.fold] if args.fold else None
    artifacts = discover_fold_run_artifacts(run_dir=run_dir, run_names=runs, fold_subset=fold_subset)
    if not artifacts:
        raise SystemExit("No matching run artifacts found.")

    source_paths = []
    for a in artifacts:
        if a.summary_path is not None:
            source_paths.append(a.summary_path)
        if a.sweep_path is not None:
            source_paths.append(a.sweep_path)

    settings = {
        "script": "compare_yolo_runs",
        "mode": args.mode,
        "runs": runs,
        "fold": args.fold or "",
        "aggregate_folds": bool(args.aggregate_folds),
    }
    cache_meta = layout["cache"] / "compare_cache_meta.json"
    points_csv = layout["tables"] / "comparison_runs.csv"
    points_agg_csv = layout["tables"] / "summary_mean_std.csv"
    sweep_raw_csv = layout["tables"] / "sweep_rows.csv"
    sweep_agg_csv = layout["tables"] / "sweep_summary_mean_std.csv"

    if is_cache_valid(cache_meta, settings=settings, source_paths=source_paths) and points_csv.exists():
        df_points = pd.read_csv(points_csv)
        df_points_agg = pd.read_csv(points_agg_csv) if points_agg_csv.exists() else pd.DataFrame()
        df_sweep = pd.read_csv(sweep_raw_csv) if sweep_raw_csv.exists() else pd.DataFrame()
        df_sweep_agg = pd.read_csv(sweep_agg_csv) if sweep_agg_csv.exists() else pd.DataFrame()
    else:
        df_points = load_selected_threshold_rows(artifacts)
        df_sweep = load_sweep_rows(artifacts, strategy="top1")
        df_points_agg = aggregate_selected_threshold(df_points)
        df_sweep_agg = aggregate_sweeps(df_sweep, strategy="top1")
        df_points.to_csv(points_csv, index=False)
        df_points_agg.to_csv(points_agg_csv, index=False)
        df_sweep.to_csv(sweep_raw_csv, index=False)
        df_sweep_agg.to_csv(sweep_agg_csv, index=False)
        write_cache_metadata(cache_meta, settings=settings, source_paths=source_paths)

    # Optional CNN baseline row in aggregated point summary.
    if not args.no_cnn:
        cnn_df = _load_cnn_metrics_table(run_dir, args.cnn_metrics_csv)
        if cnn_df is not None and not df_points.empty:
            cnn_rows: List[Dict[str, Any]] = []
            touched_folds = sorted(set(df_points["fold"].astype(str).tolist()))
            for fold_name in touched_folds:
                noise = _parse_noise_from_fold_name(fold_name)
                row = _cnn_row_for_fold(run_dir, fold_name, noise, cnn_df)
                if row is not None:
                    row["run_name"] = "CNN"
                    cnn_rows.append(row)
            if cnn_rows:
                df_cnn = pd.DataFrame(cnn_rows)
                base = df_cnn.rename(
                    columns={
                        "test_selected_TCR": "test_selected_TCR",
                        "test_selected_NMR": "test_selected_NMR",
                        "test_selected_CMR": "test_selected_CMR",
                        "test_selected_F": "test_selected_F",
                        "test_selected_ACC": "test_selected_ACC",
                    }
                )
                df_points = pd.concat([df_points, base], ignore_index=True, sort=False)
                df_points_agg = aggregate_selected_threshold(df_points)
                df_points.to_csv(points_csv, index=False)
                df_points_agg.to_csv(points_agg_csv, index=False)

    cnn_summary_row: Optional[Dict[str, Any]] = None
    if not args.no_cnn:
        df_cnn_agg = df_points_agg[df_points_agg["run_name"] == "CNN"] if not df_points_agg.empty else pd.DataFrame()
        if not df_cnn_agg.empty:
            cnn_summary_row = df_cnn_agg.iloc[0].to_dict()

    # Behavior matrix:
    # - explicit single fold: compare runs within that fold (existing style)
    # - single run + no fold + no aggregate flag: show per-fold traces
    # - across-fold mode OR aggregate flag: show aggregated comparisons
    if args.mode == "single-fold" and args.fold:
        if df_points.empty:
            print("[compare_yolo_runs] No selected-threshold summaries found.")
            return
        df_one = df_points.drop_duplicates(subset=["run_name"]).reset_index(drop=True)
        _plot_bar_metrics(df_one, layout["plots"], metric_prefix="test_selected", run_colors=run_colors)
        _plot_scatter_tcr_vs_nmr(df_one, layout["plots"], metric_prefix="test_selected", run_colors=run_colors)
        recs = _discover_run_records_single_fold(run_dir, args.fold, runs, None, None)
        _plot_overlay_f_vs_conf(
            recs,
            df_one,
            layout["plots"],
            run_colors=run_colors,
            annotate_confidence=not args.no_annotate_confidence,
            annotate_every=max(1, int(args.annotate_every)),
        )
        _plot_overlay_tcr_vs_nmr(
            recs,
            df_one,
            layout["plots"],
            run_colors=run_colors,
            annotate_confidence=not args.no_annotate_confidence,
            annotate_every=max(1, int(args.annotate_every)),
        )
        return

    if args.mode == "single-fold" and len(runs) == 1 and not args.aggregate_folds:
        if df_sweep.empty:
            print("[compare_yolo_runs] No sweep CSVs found for by-fold view.")
            return
        run_name = runs[0]
        r_sweep = df_sweep[df_sweep["run_name"] == run_name].copy()
        r_points = df_points[df_points["run_name"] == run_name].copy()
        cnn_for_curves = cnn_summary_row if args.cnn_on_curves else None
        _plot_point_metrics_by_fold(r_points, layout["plots"], run_name=run_name)
        _plot_selected_scatter_by_fold(r_points, layout["plots"], run_name=run_name)
        _plot_f_curve_by_fold(r_sweep, layout["plots"], run_name=run_name)
        _plot_metric_curve_by_fold(r_sweep, layout["plots"], run_name=run_name, metric="TCR", cnn_summary_row=cnn_for_curves)
        _plot_metric_curve_by_fold(r_sweep, layout["plots"], run_name=run_name, metric="NMR", cnn_summary_row=cnn_for_curves)
        _plot_operating_curve_by_fold(r_sweep, layout["plots"], run_name=run_name, cnn_summary_row=cnn_for_curves)
        return

    # Aggregated view (across-fold mode, or single-run with --aggregate-folds)
    if df_points_agg.empty:
        print("[compare_yolo_runs] No aggregated point metrics available.")
        return
    _plot_point_metrics_with_uncertainty(df_points_agg, layout["plots"], run_colors, band_kind="std")
    _plot_point_metrics_with_uncertainty(df_points_agg, layout["plots"], run_colors, band_kind="ci95_t")
    _plot_scatter_tcr_vs_nmr(
        df_points_agg.rename(
            columns={
                "test_selected_TCR_mean": "test_selected_TCR",
                "test_selected_NMR_mean": "test_selected_NMR",
            }
        ),
        layout["plots"],
        metric_prefix="test_selected",
        run_colors=run_colors,
    )
    cnn_for_curves = cnn_summary_row if args.cnn_on_curves else None
    _plot_sweep_overlay_with_band(df_sweep_agg, layout["plots"], run_colors, metric="F", band_kind="std", cnn_summary_row=cnn_for_curves)
    _plot_sweep_overlay_with_band(df_sweep_agg, layout["plots"], run_colors, metric="F", band_kind="ci95_t", cnn_summary_row=cnn_for_curves)
    _plot_sweep_overlay_with_band(df_sweep_agg, layout["plots"], run_colors, metric="TCR", band_kind="std", cnn_summary_row=cnn_for_curves)
    _plot_sweep_overlay_with_band(df_sweep_agg, layout["plots"], run_colors, metric="TCR", band_kind="ci95_t", cnn_summary_row=cnn_for_curves)
    _plot_sweep_overlay_with_band(df_sweep_agg, layout["plots"], run_colors, metric="NMR", band_kind="std", cnn_summary_row=cnn_for_curves)
    _plot_sweep_overlay_with_band(df_sweep_agg, layout["plots"], run_colors, metric="NMR", band_kind="ci95_t", cnn_summary_row=cnn_for_curves)
    _plot_operating_curve_agg_with_band(
        df_sweep_agg,
        layout["plots"],
        run_colors,
        band_kind="std",
        cnn_summary_row=cnn_for_curves,
        annotate_every=2,
    )
    _plot_operating_curve_agg_with_band(
        df_sweep_agg,
        layout["plots"],
        run_colors,
        band_kind="ci95_t",
        cnn_summary_row=cnn_for_curves,
        annotate_every=2,
    )


if __name__ == "__main__":
    main_v2()


