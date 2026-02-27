#!/usr/bin/env python3
"""Aggregate one YOLO run across folds, including sweep-level statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_per_run_metric_points(df_points: pd.DataFrame, out_dir: Path, run_name: str) -> None:
    if df_points.empty:
        return
    metrics = ["test_selected_TCR", "test_selected_NMR", "test_selected_CMR", "test_selected_F"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    x = np.arange(len(df_points))
    labels = df_points["fold"].astype(str).tolist()
    for ax, c in zip(axes, metrics):
        if c not in df_points.columns:
            continue
        ax.bar(x, df_points[c].to_numpy(dtype=float), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(c.replace("test_selected_", ""))
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_selected_threshold_per_fold.png", dpi=200)
    plt.close(fig)


def _plot_metric_vs_confidence(df_sweep_agg: pd.DataFrame, out_dir: Path, run_name: str, metric: str, band: str) -> None:
    if df_sweep_agg.empty:
        return
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_{band}_low"
    high_col = f"{metric}_{band}_high"
    if mean_col not in df_sweep_agg.columns:
        return
    sdf = df_sweep_agg.sort_values("threshold")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sdf["threshold"], sdf[mean_col], lw=2, label=f"{run_name} mean")
    if low_col in sdf.columns and high_col in sdf.columns and int(sdf["n"].max()) > 1:
        ax.fill_between(sdf["threshold"], sdf[low_col], sdf[high_col], alpha=0.2, label=band)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(metric)
    ax.set_title(f"{run_name}: {metric} vs confidence ({band})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_{metric.lower()}_vs_confidence_{band}.png", dpi=200)
    plt.close(fig)


def _plot_tcr_vs_nmr(
    df_sweep_raw: pd.DataFrame,
    df_sweep_agg: pd.DataFrame,
    out_dir: Path,
    run_name: str,
    band: str,
) -> None:
    if df_sweep_raw.empty or df_sweep_agg.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for fold, sdf in df_sweep_raw.groupby("fold"):
        if "NMR" not in sdf.columns or "TCR" not in sdf.columns:
            continue
        sdf = sdf.sort_values("threshold")
        ax.plot(sdf["NMR"], sdf["TCR"], alpha=0.2, linewidth=1.0)
    if "NMR_mean" in df_sweep_agg.columns and "TCR_mean" in df_sweep_agg.columns:
        s = df_sweep_agg.sort_values("threshold")
        ax.plot(s["NMR_mean"], s["TCR_mean"], linewidth=2.0, marker="o", markersize=3.0, label="mean curve")
        n = int(s["n"].max()) if "n" in s.columns else 1
        if n > 1:
            x_low_col = f"NMR_{band}_low"
            x_high_col = f"NMR_{band}_high"
            y_low_col = f"TCR_{band}_low"
            y_high_col = f"TCR_{band}_high"
            if all(c in s.columns for c in (x_low_col, x_high_col, y_low_col, y_high_col)):
                ax.fill_betweenx(s["TCR_mean"], s[x_low_col], s[x_high_col], alpha=0.08)
                ax.fill_between(s["NMR_mean"], s[y_low_col], s[y_high_col], alpha=0.08)
    ax.set_xlabel("NMR")
    ax.set_ylabel("TCR")
    ax.set_title(f"{run_name}: TCR vs NMR (fold traces + mean, {band})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_tcr_vs_nmr_fold_overlay_{band}.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate YOLO fold results for one run.")
    p.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/cnn_results/<run_id>.")
    p.add_argument("--run-name", type=str, required=True, help="YOLO run name under fold/yolo/runs_det/.")
    p.add_argument("--folds", type=str, default="", help="Optional comma-separated subset of folds.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory (default: <run-dir>/yolo/analysis/<run-name>/latest).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_name = args.run_name
    fold_subset = [f.strip() for f in args.folds.split(",") if f.strip()] or None
    out_root = args.output_dir or (run_dir / "yolo" / "analysis" / run_name / "latest")
    layout = default_output_layout(out_root)
    ensure_output_layout(layout)

    artifacts = discover_fold_run_artifacts(run_dir=run_dir, run_names=[run_name], fold_subset=fold_subset)
    if not artifacts:
        raise SystemExit(f"No artifacts found for run '{run_name}'.")
    source_paths = []
    for a in artifacts:
        if a.summary_path is not None:
            source_paths.append(a.summary_path)
        if a.sweep_path is not None:
            source_paths.append(a.sweep_path)

    settings = {
        "script": "aggregate_yolo_results",
        "run_name": run_name,
        "fold_subset": fold_subset or [],
    }
    cache_meta = layout["cache"] / "aggregate_cache_meta.json"
    points_csv = layout["tables"] / f"{run_name}_selected_threshold_metrics.csv"
    points_summary_csv = layout["tables"] / f"{run_name}_selected_threshold_metrics_summary.csv"
    sweep_raw_csv = layout["tables"] / f"{run_name}_sweep_rows.csv"
    sweep_summary_csv = layout["tables"] / f"{run_name}_sweep_summary_mean_std.csv"

    if is_cache_valid(cache_meta, settings=settings, source_paths=source_paths) and points_csv.exists():
        df_points = pd.read_csv(points_csv)
        df_points_summary = pd.read_csv(points_summary_csv) if points_summary_csv.exists() else pd.DataFrame()
        df_sweep = pd.read_csv(sweep_raw_csv) if sweep_raw_csv.exists() else pd.DataFrame()
        df_sweep_summary = pd.read_csv(sweep_summary_csv) if sweep_summary_csv.exists() else pd.DataFrame()
    else:
        df_points = load_selected_threshold_rows(artifacts)
        df_points_summary = aggregate_selected_threshold(df_points)
        df_sweep = load_sweep_rows(artifacts, strategy="top1")
        df_sweep_summary = aggregate_sweeps(df_sweep, strategy="top1")
        df_points.to_csv(points_csv, index=False)
        df_points_summary.to_csv(points_summary_csv, index=False)
        df_sweep.to_csv(sweep_raw_csv, index=False)
        df_sweep_summary.to_csv(sweep_summary_csv, index=False)
        write_cache_metadata(cache_meta, settings=settings, source_paths=source_paths)

    _plot_per_run_metric_points(df_points, layout["plots"], run_name=run_name)
    _plot_metric_vs_confidence(df_sweep_summary, layout["plots"], run_name=run_name, metric="F", band="std")
    _plot_metric_vs_confidence(df_sweep_summary, layout["plots"], run_name=run_name, metric="F", band="ci95_t")
    _plot_metric_vs_confidence(df_sweep_summary, layout["plots"], run_name=run_name, metric="TCR", band="std")
    _plot_metric_vs_confidence(df_sweep_summary, layout["plots"], run_name=run_name, metric="TCR", band="ci95_t")
    _plot_metric_vs_confidence(df_sweep_summary, layout["plots"], run_name=run_name, metric="NMR", band="std")
    _plot_metric_vs_confidence(df_sweep_summary, layout["plots"], run_name=run_name, metric="NMR", band="ci95_t")
    _plot_tcr_vs_nmr(df_sweep, df_sweep_summary, layout["plots"], run_name=run_name, band="std")
    _plot_tcr_vs_nmr(df_sweep, df_sweep_summary, layout["plots"], run_name=run_name, band="ci95_t")

    print(f"Aggregated outputs written under: {out_root}")


if __name__ == "__main__":
    main()
