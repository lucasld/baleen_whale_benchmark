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
import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ---- Environment: set BEFORE importing ultralytics ----
os.environ["RICH_PROGRESS_BAR"] = "0"   # clean logs on SLURM / non-TTY
os.environ["ULTRALYTICS_QUIET"] = "1"   # suppress batch tqdm spam; we'll print per-epoch

from ultralytics import YOLO, settings
from ultralytics.utils import LOGGER
from ultralytics.utils import DEFAULT_CFG as UL_DEFAULT_CFG

# Initialize Weights & Biases
import wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))
# Override WandB project name globally before enabling Ultralytics logging
os.environ["WANDB_PROJECT"] = "baleen-yolo"
# Enable W&B logging in Ultralytics
settings.update({"wandb": False})

# Monkey patch WandB callback to use short project name
from ultralytics.utils.callbacks import wb
_original_on_pretrain_routine_start = wb.on_pretrain_routine_start

def _patched_on_pretrain_routine_start(trainer):
    """Modified WandB callback that uses a short project name."""
    if not wb.wb.run:
        wb.wb.init(
            project="baleen-yolo",  # Short project name instead of path-derived name
            name=str(trainer.args.name).replace("/", "-"),
            config=vars(trainer.args),
        )

wb.on_pretrain_routine_start = _patched_on_pretrain_routine_start

# Match logging style from yolo_test.py: concise, readable logs
LOGGER.setLevel(logging.WARNING)

# Add base directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.yolo.yolo_eval_metrics import run_yolo_predictions_and_metrics

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


def save_training_config(out_dir: Path, args: argparse.Namespace, run_root: Path, train_fold_name: str):
    """Save complete training configuration to JSON file for reproducibility.

    This records both explicitly provided CLI values and effective values after
    filling in Ultralytics defaults for any unset options.
    """
    import datetime as _dt

    def effective_value(key: str, cli_value):
        return cli_value if cli_value is not None else UL_DEFAULT_CFG.get(key, None)

    training_params = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "weights": args.weights,
        "optimizer": effective_value("optimizer", getattr(args, "optimizer", None)) if hasattr(args, "optimizer") else UL_DEFAULT_CFG.get("optimizer"),
        "lr0": effective_value("lr0", getattr(args, "lr0", None)) if hasattr(args, "lr0") else UL_DEFAULT_CFG.get("lr0"),
        "close_mosaic": effective_value("close_mosaic", getattr(args, "close_mosaic", None)) if hasattr(args, "close_mosaic") else UL_DEFAULT_CFG.get("close_mosaic"),
        "freeze": getattr(args, "freeze", None),
    }

    augmentations = {
        "mosaic": effective_value("mosaic", getattr(args, "mosaic", None)) if hasattr(args, "mosaic") else UL_DEFAULT_CFG.get("mosaic"),
        "degrees": effective_value("degrees", getattr(args, "degrees", None)) if hasattr(args, "degrees") else UL_DEFAULT_CFG.get("degrees"),
        "translate": effective_value("translate", getattr(args, "translate", None)) if hasattr(args, "translate") else UL_DEFAULT_CFG.get("translate"),
        "scale": effective_value("scale", getattr(args, "scale", None)) if hasattr(args, "scale") else UL_DEFAULT_CFG.get("scale"),
        "shear": effective_value("shear", getattr(args, "shear", None)) if hasattr(args, "shear") else UL_DEFAULT_CFG.get("shear"),
        "perspective": effective_value("perspective", getattr(args, "perspective", None)) if hasattr(args, "perspective") else UL_DEFAULT_CFG.get("perspective"),
        "flipud": effective_value("flipud", getattr(args, "flipud", None)) if hasattr(args, "flipud") else UL_DEFAULT_CFG.get("flipud"),
        "fliplr": effective_value("fliplr", getattr(args, "fliplr", None)) if hasattr(args, "fliplr") else UL_DEFAULT_CFG.get("fliplr"),
        "hsv_h": effective_value("hsv_h", getattr(args, "hsv_h", None)) if hasattr(args, "hsv_h") else UL_DEFAULT_CFG.get("hsv_h"),
        "hsv_s": effective_value("hsv_s", getattr(args, "hsv_s", None)) if hasattr(args, "hsv_s") else UL_DEFAULT_CFG.get("hsv_s"),
        "hsv_v": effective_value("hsv_v", getattr(args, "hsv_v", None)) if hasattr(args, "hsv_v") else UL_DEFAULT_CFG.get("hsv_v"),
        "mixup": effective_value("mixup", getattr(args, "mixup", None)) if hasattr(args, "mixup") else UL_DEFAULT_CFG.get("mixup"),
        "cutmix": effective_value("cutmix", getattr(args, "cutmix", None)) if hasattr(args, "cutmix") else UL_DEFAULT_CFG.get("cutmix"),
        "copy_paste": effective_value("copy_paste", getattr(args, "copy_paste", None)) if hasattr(args, "copy_paste") else UL_DEFAULT_CFG.get("copy_paste"),
    }

    config = {
        "run_id": run_root.name,
        "fold": train_fold_name,
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "training_params": training_params,
        "augmentations": augmentations,
        "evaluation_params": {
            "eval_conf": args.eval_conf,
            "conf_sweep": args.conf_sweep,
            "conf_sweep_min": args.conf_sweep_min,
            "conf_sweep_max": args.conf_sweep_max,
            "conf_sweep_step": args.conf_sweep_step,
        },
        "other_settings": {
            "verbose": False,
            "plots": True,
        }
    }

    config_path = out_dir / "training_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    print(f"[YOLO] Saved training configuration to: {config_path}")
    return config_path

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
    fold_name: str,
    # Augmentation parameters (optional; None preserves Ultralytics defaults)
    mosaic: float | None = None,
    degrees: float | None = None,
    translate: float | None = None,
    scale: float | None = None,
    shear: float | None = None,
    perspective: float | None = None,
    flipud: float | None = None,
    fliplr: float | None = None,
    hsv_h: float | None = None,
    hsv_s: float | None = None,
    hsv_v: float | None = None,
    mixup: float | None = None,
    cutmix: float | None = None,
    copy_paste: float | None = None,
    # Additional training parameters (optional)
    optimizer: str | None = None,
    lr0: float | None = None,
    close_mosaic: int | None = None,
    freeze: int | None = None,
) -> YOLO:
    model = YOLO(weights)
    add_epoch_logger(model)
    print(f"[YOLO] Starting DET training: epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")
    # Build kwargs with required params first
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_project),  # Keep full path for output directory
        name=run_name,
        exist_ok=True,
        verbose=False,
        plots=True,
    )

    # Only set optional params if explicitly provided to preserve Ultralytics defaults
    def _set_if_not_none(key: str, value):
        if value is not None:
            train_kwargs[key] = value

    for k, v in (
        ("mosaic", mosaic), ("degrees", degrees), ("translate", translate), ("scale", scale),
        ("shear", shear), ("perspective", perspective), ("flipud", flipud), ("fliplr", fliplr),
        ("hsv_h", hsv_h), ("hsv_s", hsv_s), ("hsv_v", hsv_v), ("mixup", mixup), ("cutmix", cutmix),
        ("copy_paste", copy_paste), ("optimizer", optimizer), ("lr0", lr0), ("close_mosaic", close_mosaic),
        ("freeze", freeze),
    ):
        _set_if_not_none(k, v)

    model.train(**train_kwargs)
    return model  # 150 epochs, augment on-off?
    


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
    p.add_argument("--eval_conf", type=float, default=0.05, help="Baseline confidence threshold for reporting metrics (default: 0.05).")
    p.add_argument(
        "--conf_sweep",
        dest="conf_sweep",
        action="store_true",
        default=True,
        help="Evaluate metrics across confidence sweep (default: enabled; range 0.01-0.95 step 0.05).",
    )
    p.add_argument(
        "--no-conf_sweep",
        dest="conf_sweep",
        action="store_false",
        help="Disable confidence sweep and evaluate only at --eval_conf.",
    )
    p.add_argument("--conf_sweep_min", type=float, default=0.01, help="Minimum confidence for sweep (inclusive).")
    p.add_argument("--conf_sweep_max", type=float, default=0.95, help="Maximum confidence for sweep (inclusive).")
    p.add_argument("--conf_sweep_step", type=float, default=0.05, help="Step size for confidence sweep.")
    p.add_argument("--name", type=str, default=None, help=f"Name for the detector run under {YOLO_BASE_DIRNAME}/{PROJECT_DIRNAME}/. Default '{DEFAULT_RUN_NAME}' (overwrite).")
    p.add_argument("--overwrite", action="store_true", help=f"Overwrite existing {YOLO_BASE_DIRNAME}/{PROJECT_DIRNAME}/<name> if it exists.")

    # Augmentation parameters (optional; omit to keep Ultralytics defaults)
    p.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability (default: Ultralytics)")
    p.add_argument("--degrees", type=float, default=None, help="Rotation degrees (+/-, default: Ultralytics)")
    p.add_argument("--translate", type=float, default=None, help="Translation fraction (+/-, default: Ultralytics)")
    p.add_argument("--scale", type=float, default=None, help="Scale gain (+/-, default: Ultralytics)")
    p.add_argument("--shear", type=float, default=None, help="Shear degrees (+/-, default: Ultralytics)")
    p.add_argument("--perspective", type=float, default=None, help="Perspective fraction (default: Ultralytics)")
    p.add_argument("--flipud", type=float, default=None, help="Vertical flip probability (default: Ultralytics)")
    p.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability (default: Ultralytics)")
    p.add_argument("--hsv_h", type=float, default=None, help="HSV hue augmentation fraction (default: Ultralytics)")
    p.add_argument("--hsv_s", type=float, default=None, help="HSV saturation augmentation fraction (default: Ultralytics)")
    p.add_argument("--hsv_v", type=float, default=None, help="HSV value (brightness) augmentation fraction (default: Ultralytics)")
    p.add_argument("--mixup", type=float, default=None, help="MixUp augmentation probability (default: Ultralytics)")
    p.add_argument("--cutmix", type=float, default=None, help="CutMix augmentation probability (default: Ultralytics)")
    p.add_argument("--copy_paste", type=float, default=None, help="Copy-paste augmentation probability (default: Ultralytics)")

    # Additional training parameters (optional; omit to keep defaults)
    p.add_argument("--optimizer", type=str, default=None, help="Optimizer (default: Ultralytics)")
    p.add_argument("--lr0", type=float, default=None, help="Initial learning rate (default: Ultralytics)")
    p.add_argument("--close_mosaic", type=int, default=None, help="Disable mosaic augmentation for final N epochs (default: Ultralytics)")
    p.add_argument("--freeze", type=int, default=None, help="Freeze first N layers (default: not set)")
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

    def make_unique_name(base_name: str) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{base_name}_{ts}"
        # ensure uniqueness within out_project
        suffix = 2
        candidate = out_project / name
        while candidate.exists():
            name = f"{base_name}_{ts}_{suffix}"
            candidate = out_project / name
            suffix += 1
        return name

    provided_name = (args.name.strip() if isinstance(args.name, str) else None)
    if provided_name:
        run_name = provided_name
    else:
        run_name = make_unique_name(DEFAULT_RUN_NAME)

    out_dir = out_project / run_name
    if out_dir.exists():
        if args.overwrite and not args.skip_train:
            print(f"[YOLO] overwrite: removing existing {out_dir}")
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"[YOLO] Output directory already exists: {out_dir}. Use --overwrite to reuse this name or omit --name for a fresh timestamped run."
            )
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
            fold_name=train_fold.name,
            # Augmentation parameters (only pass if provided)
            mosaic=args.mosaic,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            mixup=args.mixup,
            cutmix=args.cutmix,
            copy_paste=args.copy_paste,
            # Additional training parameters (only pass if provided)
            optimizer=args.optimizer,
            lr0=args.lr0,
            close_mosaic=args.close_mosaic,
            freeze=args.freeze,
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

    # Persist effective configuration for this run
    save_training_config(out_dir, args, run_root, train_fold.name)

    # Run custom metrics (TCR, NMR, CMR, F)
    noise = float(train_fold.name.split('_noise_')[1]) if '_noise_' in train_fold.name else 0.0
    model_name = f"YOLO_{run_name}"
    test_images_dir = ds_root / "images" / "test"
    labels_dir = ds_root / "labels" / "test"
    print(f"[YOLO] Computing custom metrics for {train_fold.name}")
    conf_thresholds = None
    if args.conf_sweep:
        sweep_vals = np.arange(args.conf_sweep_min, args.conf_sweep_max + 1e-9, args.conf_sweep_step)
        conf_thresholds = [float(round(v, 6)) for v in sweep_vals if v >= 0.0]

    custom_metrics, cm_path = run_yolo_predictions_and_metrics(
        model,
        test_images_dir,
        labels_dir,
        lbl,
        out_dir,
        model_name,
        noise,
        conf_thresh=args.eval_conf,
        conf_thresholds=conf_thresholds,
    )


if __name__ == "__main__":
    main()
