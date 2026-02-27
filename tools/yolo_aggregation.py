#!/usr/bin/env python3
"""Shared utilities for YOLO fold-level aggregation and caching."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SUMMARY_FILENAME = "selected_threshold_summary.json"
SWEEP_GLOB_TEMPLATE = "YOLO_{run_name}_metrics_confidence_sweep.csv"
POINT_METRICS = ("TCR", "NMR", "CMR", "F", "ACC")
CURVE_METRICS = ("TCR", "NMR", "CMR", "F", "ACC")

# Two-sided 95% t critical values (df=1..30), then asymptotic 1.96.
_T95_BY_DF: Dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class FoldRunArtifact:
    run_name: str
    fold: str
    summary_path: Optional[Path]
    sweep_path: Optional[Path]


def discover_folds(run_dir: Path) -> List[Path]:
    folds = sorted([p for p in run_dir.glob("fold_*") if p.is_dir()])
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found under: {run_dir}")
    return folds


def discover_fold_run_artifacts(
    run_dir: Path,
    run_names: Sequence[str],
    fold_subset: Optional[Sequence[str]] = None,
) -> List[FoldRunArtifact]:
    fold_set = set(fold_subset or [])
    out: List[FoldRunArtifact] = []
    for fold_dir in discover_folds(run_dir):
        fold_name = fold_dir.name
        if fold_set and fold_name not in fold_set:
            continue
        runs_det = fold_dir / "yolo" / "runs_det"
        if not runs_det.exists():
            continue
        for run_name in run_names:
            run_path = runs_det / run_name
            if not run_path.exists():
                continue
            summary = run_path / SUMMARY_FILENAME
            if not summary.exists():
                summary = None
            sweep = run_path / SWEEP_GLOB_TEMPLATE.format(run_name=run_name)
            if not sweep.exists():
                sweep = None
            out.append(
                FoldRunArtifact(
                    run_name=run_name,
                    fold=fold_name,
                    summary_path=summary,
                    sweep_path=sweep,
                )
            )
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_selected_threshold_rows(artifacts: Sequence[FoldRunArtifact]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for a in artifacts:
        if a.summary_path is None:
            continue
        try:
            payload = _read_json(a.summary_path)
        except Exception:
            continue
        row: Dict[str, Any] = {
            "run_name": a.run_name,
            "fold": a.fold,
            "summary_path": str(a.summary_path),
            "selected_threshold": _safe_float(payload.get("selected_threshold")),
            "test_best_threshold": _safe_float(payload.get("test_best_threshold")),
        }
        row["threshold_gap"] = abs(row["selected_threshold"] - row["test_best_threshold"])
        prefix_map = {
            "validation_metrics": "val",
            "test_metrics_at_selected_threshold": "test_selected",
            "test_best_metrics": "test_best",
        }
        for source_key, prefix in prefix_map.items():
            metric_map = payload.get(source_key, {}) or {}
            for m in POINT_METRICS:
                if m in metric_map:
                    row[f"{prefix}_{m}"] = _safe_float(metric_map.get(m))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["run_name", "fold"]).reset_index(drop=True)


def load_sweep_rows(artifacts: Sequence[FoldRunArtifact], strategy: str = "top1") -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for a in artifacts:
        if a.sweep_path is None:
            continue
        try:
            df = pd.read_csv(a.sweep_path)
        except Exception:
            continue
        if "threshold" not in df.columns or "strategy" not in df.columns:
            continue
        sdf = df[df["strategy"] == strategy].copy()
        if sdf.empty:
            continue
        sdf["run_name"] = a.run_name
        sdf["fold"] = a.fold
        sdf["sweep_csv_path"] = str(a.sweep_path)
        rows.append(sdf)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["threshold"] = out["threshold"].astype(float)
    return out.sort_values(["run_name", "fold", "threshold"]).reset_index(drop=True)


def t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    df = n - 1
    if df in _T95_BY_DF:
        return _T95_BY_DF[df]
    return 1.96


def _attach_uncertainty(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in metric_cols:
        mean_col = f"{c}_mean"
        std_col = f"{c}_std"
        sem_col = f"{c}_sem"
        low_std_col = f"{c}_std_low"
        high_std_col = f"{c}_std_high"
        low_ci_col = f"{c}_ci95_t_low"
        high_ci_col = f"{c}_ci95_t_high"

        out[sem_col] = out[std_col] / np.sqrt(out["n"].clip(lower=1).astype(float))
        tcrit = out["n"].astype(int).map(t_critical_95)
        margin = tcrit * out[sem_col]
        out[low_std_col] = out[mean_col] - out[std_col]
        out[high_std_col] = out[mean_col] + out[std_col]
        out[low_ci_col] = out[mean_col] - margin
        out[high_ci_col] = out[mean_col] + margin
    return out


def aggregate_selected_threshold(df_points: pd.DataFrame) -> pd.DataFrame:
    if df_points.empty:
        return pd.DataFrame()
    value_cols = [f"test_selected_{m}" for m in POINT_METRICS if f"test_selected_{m}" in df_points.columns]
    if not value_cols:
        return pd.DataFrame()
    agg = (
        df_points.groupby("run_name", dropna=False)[value_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    if "test_selected_F_count" in agg.columns:
        agg = agg.rename(columns={"test_selected_F_count": "n"})
    else:
        first_count = [c for c in agg.columns if c.endswith("_count")]
        agg["n"] = agg[first_count[0]] if first_count else 1
    for c in value_cols:
        std_col = f"{c}_std"
        if std_col in agg.columns:
            agg[std_col] = agg[std_col].fillna(0.0)
    out = _attach_uncertainty(agg, value_cols)
    return out.sort_values(["run_name"]).reset_index(drop=True)


def aggregate_sweeps(df_sweep: pd.DataFrame, strategy: str = "top1") -> pd.DataFrame:
    if df_sweep.empty:
        return pd.DataFrame()
    sdf = df_sweep[df_sweep["strategy"] == strategy].copy()
    if sdf.empty:
        return pd.DataFrame()
    metric_cols = [m for m in CURVE_METRICS if m in sdf.columns]
    if not metric_cols:
        return pd.DataFrame()
    agg = (
        sdf.groupby(["run_name", "threshold", "strategy"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    count_cols = [c for c in agg.columns if c.endswith("_count")]
    if count_cols:
        agg["n"] = agg[count_cols[0]]
    else:
        agg["n"] = 1
    for c in metric_cols:
        std_col = f"{c}_std"
        if std_col in agg.columns:
            agg[std_col] = agg[std_col].fillna(0.0)
    out = _attach_uncertainty(agg, metric_cols)
    return out.sort_values(["run_name", "threshold"]).reset_index(drop=True)


def default_output_layout(base_out_dir: Path) -> Dict[str, Path]:
    return {
        "root": base_out_dir,
        "tables": base_out_dir / "tables",
        "plots": base_out_dir / "plots",
        "cache": base_out_dir / "cache",
    }


def ensure_output_layout(layout: Dict[str, Path]) -> None:
    for p in layout.values():
        p.mkdir(parents=True, exist_ok=True)


def source_mtime_map(paths: Iterable[Path]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in paths:
        if p.exists():
            out[str(p)] = p.stat().st_mtime
    return out


def _cache_payload(
    settings: Dict[str, Any],
    source_mtimes: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "settings": settings,
        "source_mtimes": source_mtimes,
    }


def write_cache_metadata(cache_meta_path: Path, settings: Dict[str, Any], source_paths: Iterable[Path]) -> None:
    payload = _cache_payload(settings=settings, source_mtimes=source_mtime_map(source_paths))
    cache_meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def is_cache_valid(cache_meta_path: Path, settings: Dict[str, Any], source_paths: Iterable[Path]) -> bool:
    if not cache_meta_path.exists():
        return False
    try:
        payload = json.loads(cache_meta_path.read_text())
    except Exception:
        return False
    if payload.get("settings") != settings:
        return False
    expected = payload.get("source_mtimes", {})
    current = source_mtime_map(source_paths)
    return expected == current
