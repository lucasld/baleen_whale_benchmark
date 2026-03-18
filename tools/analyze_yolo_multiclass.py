#!/usr/bin/env python3
"""CLI entrypoint for YOLO multiclass / Top-1 collapse analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiclass_analysis_lib import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze YOLO raw prediction artifacts to quantify multiclass snippets, "
            "Top-1 collapse effects, fold-level error breakdowns, and YOLO-vs-CNN deltas."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to outputs/cnn_results/<run_id>.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="F1",
        help="YOLO run name under fold_*/yolo/runs_det/ (default: F1).",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="",
        help="Optional comma-separated subset of fold directory names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root (default: <run-dir>/yolo/multiclass_analysis/<run-name>/latest)."
        ),
    )
    parser.add_argument(
        "--examples-per-fold",
        type=int,
        default=200,
        help="How many rescued examples to export per fold (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    fold_subset = [f.strip() for f in args.folds.split(",") if f.strip()] or None
    out_root = args.output_dir or (run_dir / "yolo" / "multiclass_analysis" / args.run_name / "latest")

    run_analysis(
        run_dir=run_dir,
        run_name=args.run_name,
        output_dir=out_root,
        fold_subset=fold_subset,
        examples_per_fold=args.examples_per_fold,
    )

    print(f"Multiclass analysis written under: {out_root}")
    print(f"- summary: {out_root / 'summary.txt'}")
    print(f"- aggregate summary: {out_root / 'aggregate_summary.json'}")
    print(f"- tables: {out_root / 'tables'}")
    print(f"- plots: {out_root / 'plots'}")
    print(f"- examples: {out_root / 'examples'}")


if __name__ == "__main__":
    main()
