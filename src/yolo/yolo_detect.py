#!/usr/bin/env python3
"""
YOLO detection training + evaluation helper.

- Uses the pre-cleaned YOLO dataset created during CNN training.
- Trains a YOLO detector (default yolo11n.pt) on the selected fold and evaluates on its test split.
- Stores all YOLO artifacts under each fold at yolo/runs_det/...

Assumptions / Layout:
run_root/
  config.json                         # contains "CATEGORIES" (ordered) and "CATEGORIES_TO_JOIN"
  labels.json                         # contains merged taxonomy IDs, e.g. {"20Hz20Plus":0,"ABZ":1,"DDswp":2}
  fold_*/                             # each fold has a yolo_dataset/ created during CNN training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

# ---- Environment: set BEFORE importing ultralytics ----
os.environ["RICH_PROGRESS_BAR"] = "0"   # clean logs on SLURM / non-TTY
os.environ["ULTRALYTICS_QUIET"] = "1"   # suppress batch tqdm spam; we'll print per-epoch

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Match logging style from yolo_test.py: concise, readable logs
LOGGER.setLevel(logging.WARNING)

# =========================
# Constants
# =========================

YOLO_BASE_DIRNAME = "yolo"
PROJECT_DIRNAME = "runs_det"
DEFAULT_RUN_NAME = "det"


def find_latest_run(cnn_results_root: Path) -> Path:
    runs = [p for p in cnn_results_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found under {cnn_results_root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def list_folds(run_root: Path) -> List[Path]:
    return sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])


def select_train_fold(fold_dirs: List[Path], requested_fold: Optional[str]) -> Path:
    if requested_fold:
        chosen = next((p for p in fold_dirs if p.name == requested_fold), None)
        if chosen is None:
            raise FileNotFoundError(f"Requested fold missing: {requested_fold}")
        return chosen
    # Default to the first fold
    return fold_dirs[0]


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def add_epoch_logger(model: YOLO):
    def _epoch(trainer):
        try:
            epoch = trainer.epoch + 1
            total = trainer.epochs
            try:
                lr = trainer.optimizer.param_groups[0].get("lr", None)
            except Exception:
                lr = None
            v = getattr(trainer.validator, "metrics", None)
            box = getattr(v, "box", None)
            m50 = getattr(box, "map50", None)
            m   = getattr(box, "map", None)
            P   = getattr(box, "P", None)
            R   = getattr(box, "R", None)

            def fmt(x): 
                try: return f"{float(x):.4f}"
                except: return "-"

            print(f"[YOLO][epoch {epoch}/{total}] mAP50={fmt(m50)} mAP50-95={fmt(m)} P={fmt(P)} R={fmt(R)} lr={fmt(lr)}")
        except Exception:
            pass
    try:
        model.add_callback("on_fit_epoch_end", _epoch)
    except Exception:
        pass


def train_detector(
    weights: str,
    data_yaml: Path,
    out_project: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> YOLO:
    model = YOLO(weights)
    add_epoch_logger(model)
    print(f"[YOLO] Starting DET training: epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_project),
        name=run_name,
        exist_ok=True,
        verbose=False,   # no per-batch spam
        plots=False,
    )
    return model


def best_weights_or(weights: str, out_project: Path, run_name: str) -> Path:
    best = out_project / run_name / "weights" / "best.pt"
    return best if best.exists() else Path(weights)


# =========================
# Main workflow
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO detector using pre-cleaned dataset from CNN training.")
    p.add_argument("--run_dir", type=str, default=None, help="Path to CNN run folder (outputs/cnn_results/YYMMDD_...).")
    p.add_argument("--train_fold", type=str, default=None, help="Fold name to train on (e.g., fold_XXX). Auto-pick if omitted.")
    p.add_argument("--weights", type=str, default="yolo11n.pt", help="Detector weights (init for training and/or eval).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", type=str, default="0", help="GPU index or 'cpu'.")
    p.add_argument("--skip_train", action="store_true", help="Skip training; only evaluate with provided/best weights.")
    p.add_argument("--name", type=str, default=None, help=f"Name for the detector run under {YOLO_BASE_DIRNAME}/{PROJECT_DIRNAME}/. Default '{DEFAULT_RUN_NAME}' (overwrite).")
    p.add_argument("--overwrite", action="store_true", help=f"Overwrite existing {YOLO_BASE_DIRNAME}/{PROJECT_DIRNAME}/<name> if it exists.")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = Path(os.getcwd())
    results_root = base_dir / "outputs" / "cnn_results"
    run_root = Path(args.run_dir) if args.run_dir else find_latest_run(results_root)
    print(f"[YOLO] Using run_dir: {run_root}")

    folds = list_folds(run_root)
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found in {run_root}")
    train_fold = select_train_fold(folds, args.train_fold)
    print(f"[YOLO] Using fold for train/val: {train_fold.name}")

    # Read merged taxonomy from labels.json
    lbl = read_json(run_root / "labels.json")
    merged_names = [name for name in lbl.keys() if name.lower() != "noise"]
    merged_names.sort(key=lambda x: lbl[x])  # sort by ID

    # Use the pre-cleaned YOLO dataset created during CNN training
    ds_root = train_fold / "yolo_dataset"
    if not ds_root.exists():
        raise FileNotFoundError(f"YOLO dataset not found at {ds_root}. Ensure CNN training has created it.")

    # Prepare detector output project inside the fold (fold/yolo/runs_det)
    yolo_base = train_fold / YOLO_BASE_DIRNAME
    out_project = yolo_base / PROJECT_DIRNAME
    out_project.mkdir(parents=True, exist_ok=True)
    run_name = args.name or DEFAULT_RUN_NAME
    out_dir = out_project / run_name
    if out_dir.exists() and args.overwrite:
        print(f"[YOLO] overwrite: removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training YAML (names/nc included for clarity)
    data_yaml = out_dir / "data_trainval.yaml"
    write_yaml(data_yaml, {
        "path": str(ds_root.resolve()),
        "train": str((ds_root / "images" / "train").resolve()),
        "val":   str((ds_root / "images" / "valid").resolve()),
        "test":  str((ds_root / "images" / "test").resolve()),
        "names": merged_names,
        "nc": len(merged_names),
    })

    # Train
    if not args.skip_train:
        model = train_detector(
            weights=args.weights,
            data_yaml=data_yaml,
            out_project=out_project,
            run_name=run_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    else:
        print("[YOLO] Skipping training (--skip_train)")
        model = YOLO(args.weights)

    # Load best weights if available
    eval_weights = best_weights_or(args.weights, out_project, run_name)
    model = YOLO(str(eval_weights))
    print(f"[YOLO] Loaded eval weights: {eval_weights}")

    # Evaluate once on this fold's test split
    print(f"[YOLO] Evaluating on test split for {train_fold.name}")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        plots=False,
        workers=4,
    )
    try:
        box = getattr(metrics, "box", None)
        m50 = getattr(box, "map50", None)
        m   = getattr(box, "map", None)
        P   = getattr(box, "P", None)
        R   = getattr(box, "R", None)
        def fmt(x):
            try: return f"{float(x):.4f}"
            except: return "-"
        print(f"[YOLO][eval] {train_fold.name} mAP50={fmt(m50)} mAP50-95={fmt(m)} P={fmt(P)} R={fmt(R)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
