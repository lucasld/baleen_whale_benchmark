#!/usr/bin/env python3
"""Sanity checks for YOLO dataset folds (label mapping and noise counts)."""

from __future__ import annotations

import argparse
from pathlib import Path


def load_labels_map(run_dir: Path) -> dict:
    path = run_dir / "labels.json"
    if not path.exists():
        raise FileNotFoundError(f"labels.json not found under {run_dir}")
    return __import__("json").loads(path.read_text())


def mismatch_report(images_dir: Path, labels_dir: Path) -> list[str]:
    missing = []
    for img in images_dir.glob("*.png"):
        label = labels_dir / f"{img.stem}.txt"
        if not label.exists():
            missing.append(img.name)
    return missing


def check_fold(fold_dir: Path, lbl: dict) -> None:
    yolo_root = fold_dir / "yolo_dataset"
    if not yolo_root.exists():
        raise FileNotFoundError(f"Fold {fold_dir.name} missing yolo_dataset")
    labels_dir = yolo_root / "labels" / "train"
    images_dir = yolo_root / "images" / "train"
    missing = mismatch_report(images_dir, labels_dir)
    if missing:
        raise RuntimeError(f"Missing label files for {len(missing)} images in {fold_dir.name} (first: {missing[0]})")
    empty_count = sum(1 for f in labels_dir.glob("*.txt") if f.stat().st_size == 0)
    if empty_count == 0:
        raise RuntimeError(f"No empty (noise) label files found in {labels_dir}")
    print(f"[check] {fold_dir.name}: {len(list(images_dir.glob('*.png')))} train images, {empty_count} noise-only samples")
    if "Noise" not in lbl:
        raise RuntimeError("Noise class missing from labels.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset folds before running experiments.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/cnn_results/<run_id>.")
    parser.add_argument("--fold", type=str, default=None, help="Optional fold name (default: all folds found).")
    args = parser.parse_args()

    lbl = load_labels_map(args.run_dir)
    folds = [p for p in sorted(args.run_dir.glob("fold_*")) if p.is_dir()]
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found under {args.run_dir}")
    for fold in folds:
        if args.fold and fold.name != args.fold:
            continue
        check_fold(fold, lbl)


if __name__ == "__main__":
    main()
