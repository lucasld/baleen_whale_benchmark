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
import time

import numpy as np
import pandas as pd

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ---- Environment: set BEFORE importing ultralytics ----
os.environ["RICH_PROGRESS_BAR"] = "0"   # clean logs on SLURM / non-TTY
os.environ["ULTRALYTICS_QUIET"] = "1"   # suppress batch tqdm spam; we'll print per-epoch

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils import DEFAULT_CFG as UL_DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg

# Match logging style from yolo_test.py: concise, readable logs
LOGGER.setLevel(logging.WARNING)

# Add base directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.yolo.aug import NoiseMix
from src.yolo.yolo_eval_metrics import run_yolo_predictions_and_metrics

# =========================
# Constants
# =========================

YOLO_BASE_DIRNAME = "yolo"
PROJECT_DIRNAME = "runs_det"
DEFAULT_RUN_NAME = "det"
MODEL_SIZES = ("n", "m", "x")
NOISE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class CustomDetectionTrainer(DetectionTrainer):
    """
    Custom Trainer that allows injecting custom Albumentations transforms (like NoiseMix)
    into the dataset pipeline.
    """
    def __init__(self, overrides=None, _callbacks=None, custom_transforms=None):
        super().__init__(overrides, _callbacks)
        self.custom_transforms = custom_transforms

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Override build_dataset to inject custom Albumentations transforms.
        """
        # Use the parent's logic to build the dataset correctly
        dataset = super().build_dataset(img_path, mode, batch)

        # If this is training and we have custom transforms, we wrap the dataset's transform pipeline
        if mode == "train" and self.custom_transforms:
            # In Ultralytics 8.3.x, the dataset has a .transforms attribute that is a Compose object
            old_transforms = dataset.transforms

            def wrapped_transform(labels):
                # 1. Apply our custom NoiseMix first
                # NoiseMix is an ImageOnlyTransform, so we pass 'image' and get 'image' back
                for t in self.custom_transforms:
                    labels["img"] = t(image=labels["img"])["image"]

                # 2. Apply standard YOLO transforms (mosaic, mixup, etc.)
                if old_transforms:
                    return old_transforms(labels)
                return labels

            dataset.transforms = wrapped_transform

        return dataset


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
        "rect": args.rect,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "model_size": args.model_size,
        "pretrained": args.pretrained,
        "augs_preset": args.augs,
        "noise_mix_p": args.noise_mix_p,
        "noise_mix_alpha": args.noise_mix_alpha,
        "optimizer": effective_value("optimizer", getattr(args, "optimizer", None)) if hasattr(args, "optimizer") else UL_DEFAULT_CFG.get("optimizer"),
        "lr0": effective_value("lr0", getattr(args, "lr0", None)) if hasattr(args, "lr0") else UL_DEFAULT_CFG.get("lr0"),
        "close_mosaic": effective_value("close_mosaic", getattr(args, "close_mosaic", None)) if hasattr(args, "close_mosaic") else UL_DEFAULT_CFG.get("close_mosaic"),
        "freeze": getattr(args, "freeze", None),
        "patience": getattr(args, "patience", None),
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
    rect: bool,
    seed: int,
    deterministic: bool,
    augmentations: list | None,
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
    patience: int | None = None,
) -> YOLO:
    model = YOLO(weights)
    add_epoch_logger(model)

    # Build comprehensive parameter summary for logging
    param_summary = [
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={device}",
        f"rect={rect}",
        f"seed={seed}",
        f"deterministic={deterministic}",
    ]

    # Add optional parameters if they differ from None (meaning they're being explicitly set)
    optional_params = {
        "mosaic": mosaic,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "flipud": flipud,
        "fliplr": fliplr,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "mixup": mixup,
        "cutmix": cutmix,
        "copy_paste": copy_paste,
        "optimizer": optimizer,
        "lr0": lr0,
        "close_mosaic": close_mosaic,
        "freeze": freeze,
    }

    for param_name, param_value in optional_params.items():
        if param_value is not None:
            param_summary.append(f"{param_name}={param_value}")

    # Add model info (extract from weights path if possible)
    if "yolo12" in weights:
        model_info = f"model={weights.split('/')[-1] if '/' in weights else weights}"
    else:
        model_info = f"weights={weights}"

    param_summary.insert(0, model_info)

    print(f"[YOLO] Starting DET training: {' | '.join(param_summary)}")
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
        rect=rect,
        seed=seed,
        deterministic=deterministic,
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
        ("freeze", freeze), ("patience", patience),
    ):
        _set_if_not_none(k, v)

    # Branching Logic for Custom Augmentations:
    # If we have custom Albumentations transforms (like NoiseMix), we manually instantiate our 
    # CustomDetectionTrainer. This bypasses model.train()'s strict argument validation which 
    # rejects the 'trainer' argument in some versions, and gives us full control.
    if augmentations:
        print("[YOLO] Using CustomDetectionTrainer for custom augmentations (NoiseMix).")
        
        # 1. Ensure model weights are in the overrides so the trainer can load them
        train_kwargs["model"] = weights
        
        # 2. Merge overrides with default config to create a valid cfg object
        # This prevents AttributeError/TypeError during trainer initialization
        cfg = get_cfg(cfg=UL_DEFAULT_CFG, overrides=train_kwargs)
        
        # 3. Instantiate custom trainer
        trainer = CustomDetectionTrainer(overrides=cfg, custom_transforms=augmentations)
        
        # 4. Re-attach logger (trainer creation might have reset it)
        # Note: trainer.model is a string path at this point, so we can't attach logger yet.
        # The logger will be attached inside the trainer during training or we can skip it 
        # as Ultralytics has its own logging.
        
        # 5. Train
        trainer.train()
        
        # 7. Update the local model object with the trained weights for evaluation
        # The trainer saves best.pt to the run directory
        best_pt = out_project / run_name / "weights" / "best.pt"
        if best_pt.exists():
             model = YOLO(str(best_pt))
    else:
        # Standard path for "defaults" and "spec_opt"
        model.train(**train_kwargs)
    return model  # 150 epochs, augment on-off?
    


def best_weights_or(weights: str, out_project: Path, run_name: str) -> Path:
    best = out_project / run_name / "weights" / "best.pt"
    return best if best.exists() else Path(weights)


def resolve_weights_source(args: argparse.Namespace, repo_root: Path) -> str:
    """Return the effective weights/model definition path for this run."""
    if args.weights:
        return str(Path(args.weights).expanduser().resolve())
    model_size = args.model_size.lower()
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unsupported model_size '{args.model_size}'. Expected one of {MODEL_SIZES}.")
    if args.pretrained:
        candidate = repo_root / "models" / f"yolo12{model_size}.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"Pretrained weights not found at {candidate}.")
        return str(candidate.resolve())
    return f"yolo12{model_size}.yaml"


def _resolve_image_path(images_dir: Path, stem: str) -> Path:
    for ext in NOISE_IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for stem '{stem}' not found in {images_dir} with extensions {NOISE_IMAGE_EXTENSIONS}.")


def collect_noise_image_paths(ds_root: Path) -> list[Path]:
    labels_dir = ds_root / "labels" / "train"
    images_dir = ds_root / "images" / "train"
    if not labels_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(f"Expected train labels/images under {ds_root}.")
    noise_paths: list[Path] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.stat().st_size == 0:
            stem = label_path.stem
            noise_paths.append(_resolve_image_path(images_dir, stem))
    return noise_paths


def build_augmentation_overrides(
    args: argparse.Namespace,
    noise_paths: list[Path] | None = None,
) -> tuple[dict[str, float | None], list | None]:
    aug_params = {
        "mosaic": args.mosaic,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "copy_paste": args.copy_paste,
    }
    albumentations_transforms = None
    if args.augs in {"spec_opt", "noise_mix"}:
        aug_params.update({
            "mosaic": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
        })
    if args.augs == "noise_mix":
        if not noise_paths:
            raise RuntimeError("NoiseMix preset selected but no noise images were found in the training split.")
        albumentations_transforms = [
            NoiseMix(noise_paths=noise_paths, alpha=args.noise_mix_alpha, p=args.noise_mix_p)
        ]
    return aug_params, albumentations_transforms


def select_best_threshold(metrics_df: pd.DataFrame, strategy: str = "top1") -> tuple[float, dict]:
    df = metrics_df[metrics_df["strategy"] == strategy]
    if df.empty:
        raise RuntimeError(f"No metrics available for strategy '{strategy}'.")
    df_sorted = df.sort_values(["F", "threshold"], ascending=[False, True])
    row = df_sorted.iloc[0]
    best_threshold = float(row["threshold"])
    best_f = float(row["F"])
    print(f"[YOLO] Selected best {strategy} threshold: {best_threshold:.3f} (F={best_f:.3f}) from {len(df)} candidates")
    return best_threshold, row.to_dict()


def find_metrics_for_threshold(metrics_df: pd.DataFrame, strategy: str, threshold: float) -> dict:
    df = metrics_df[(metrics_df["strategy"] == strategy) & (np.isclose(metrics_df["threshold"], threshold))]
    if df.empty:
        raise RuntimeError(f"Metrics for strategy '{strategy}' at threshold {threshold:.4f} not found.")
    return df.iloc[0].to_dict()


def append_selection_to_config(config_path: Path, selection: dict) -> None:
    data = json.loads(config_path.read_text())
    data["selection"] = selection
    config_path.write_text(json.dumps(data, indent=2))


# =========================
# Main workflow
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO detector using pre-cleaned dataset from CNN training.")
    p.add_argument("--run_dir", type=str, default=None, help="Path to CNN run folder (outputs/cnn_results/YYMMDD_...).")
    p.add_argument("--train_fold", type=str, default=None, help="Fold name to train on (e.g., fold_XXX). Auto-pick if omitted.")
    p.add_argument("--weights", type=str, default=None, help="Optional explicit detector weights (overrides --model_size/--pretrained).")
    p.add_argument("--model_size", choices=["n", "m", "x"], default="m", help="YOLO12 model size to use when --weights is omitted.")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True, help="Use pretrained weights (default).")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Train from scratch using YOLO12 YAML.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", type=str, default="0", help="GPU index or 'cpu'.")
    p.add_argument("--rect", dest="rect", action="store_true", default=True, help="Enable rectangular training/eval batches (default: True).")
    p.add_argument("--no-rect", dest="rect", action="store_false", help="Disable rectangular training/eval batches.")
    p.add_argument("--seed", type=int, default=42, help="Random seed passed to Ultralytics trainer.")
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=False, help="Enable deterministic mode: ensures reproducible results across identical runs (slower training).")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="Disable deterministic mode (default): faster training, results may vary slightly between identical runs.")
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
    p.add_argument("--conf_sweep_min", type=float, default=0.0, help="Minimum confidence for sweep (inclusive).")
    p.add_argument("--conf_sweep_max", type=float, default=0.9, help="Maximum confidence for sweep (inclusive).")
    p.add_argument("--conf_sweep_step", type=float, default=0.1, help="Step size for confidence sweep.")
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
    p.add_argument("--augs", choices=["defaults", "spec_opt", "noise_mix"], default="defaults", help="High-level augmentation preset.")
    p.add_argument("--noise_mix_p", type=float, default=0.5, help="NoiseMix application probability when --augs noise_mix.")
    p.add_argument("--noise_mix_alpha", type=float, default=0.25, help="NoiseMix blend factor (alpha) when --augs noise_mix.")

    # Additional training parameters (optional; omit to keep defaults)
    p.add_argument("--optimizer", type=str, default=None, help="Optimizer (default: Ultralytics)")
    p.add_argument("--lr0", type=float, default=None, help="Initial learning rate (default: Ultralytics)")
    p.add_argument("--close_mosaic", type=int, default=None, help="Disable mosaic augmentation for final N epochs (default: Ultralytics)")
    p.add_argument("--freeze", type=int, default=None, help="Freeze first N layers (default: not set)")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs). None keeps Ultralytics default (100). Set to 0 to disable.")
    return p.parse_args()


def main():
    start_time = time.time()
    print(f"[YOLO] Starting run at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()
    if not args.conf_sweep:
        raise ValueError("Confidence sweep must remain enabled for validation-selected confidence policy.")

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

    noise_paths: list[Path] | None = None
    if args.augs == "noise_mix":
        noise_paths = collect_noise_image_paths(ds_root)

    aug_params, albumentations_transforms = build_augmentation_overrides(args, noise_paths)
    resolved_weights = resolve_weights_source(args, base_dir)
    args.weights = resolved_weights

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
        "workers": 8,
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
            rect=args.rect,
            seed=args.seed,
            deterministic=args.deterministic,
            augmentations=albumentations_transforms,
            # Augmentation parameters (only pass if provided)
            mosaic=aug_params.get("mosaic"),
            degrees=aug_params.get("degrees"),
            translate=aug_params.get("translate"),
            scale=aug_params.get("scale"),
            shear=aug_params.get("shear"),
            perspective=aug_params.get("perspective"),
            flipud=aug_params.get("flipud"),
            fliplr=aug_params.get("fliplr"),
            hsv_h=aug_params.get("hsv_h"),
            hsv_s=aug_params.get("hsv_s"),
            hsv_v=aug_params.get("hsv_v"),
            mixup=aug_params.get("mixup"),
            cutmix=aug_params.get("cutmix"),
            copy_paste=aug_params.get("copy_paste"),
            # Additional training parameters (only pass if provided)
            optimizer=args.optimizer,
            lr0=args.lr0,
            close_mosaic=args.close_mosaic,
            freeze=args.freeze,
            patience=args.patience,
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
        rect=args.rect,
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
    config_path = save_training_config(out_dir, args, run_root, train_fold.name)

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
        print(f"[YOLO] Using confidence sweep: {len(conf_thresholds)} thresholds from {min(conf_thresholds):.3f} to {max(conf_thresholds):.3f}")
    val_images_dir = ds_root / "images" / "valid"
    val_labels_dir = ds_root / "labels" / "valid"
    val_eval_dir = out_dir / "validation_eval"
    val_eval_dir.mkdir(exist_ok=True)

    print(f"[YOLO] Starting validation evaluation...")
    _, _, val_metrics_df = run_yolo_predictions_and_metrics(
        model,
        val_images_dir,
        val_labels_dir,
        lbl,
        val_eval_dir,
        f"{model_name}_val",
        noise,
        conf_thresh=args.eval_conf,
        conf_thresholds=conf_thresholds,
        imgsz=args.imgsz,
        device=args.device,
    )
    selected_thr, val_row = select_best_threshold(val_metrics_df, strategy="top1")
    print(f"[YOLO] Validation-selected confidence: {selected_thr:.3f}")

    # Re-plot validation sweep curves with selected threshold marked
    from src.yolo.evaluation.reporting import plot_f_vs_confidence_curves, plot_tcr_vs_nmr_curves
    try:
        plot_f_vs_confidence_curves(val_metrics_df, val_eval_dir, selected_threshold=selected_thr)
    except Exception as e:
        print(f"[YOLO] Warning: Failed to re-plot F vs confidence for validation: {e}")
    try:
        plot_tcr_vs_nmr_curves(val_metrics_df, val_eval_dir, selected_threshold=selected_thr)
    except Exception as e:
        print(f"[YOLO] Warning: Failed to re-plot TCR vs NMR for validation: {e}")

    print(f"[YOLO] Starting test evaluation with validation-selected threshold {selected_thr:.3f}...")
    custom_metrics, cm_path, test_metrics_df = run_yolo_predictions_and_metrics(
        model,
        test_images_dir,
        labels_dir,
        lbl,
        out_dir,
        model_name,
        noise,
        conf_thresh=args.eval_conf,
        conf_thresholds=conf_thresholds,
        imgsz=args.imgsz,
        device=args.device,
    )
    # Re-plot test sweep curves with selected threshold marked
    try:
        plot_f_vs_confidence_curves(test_metrics_df, out_dir, selected_threshold=selected_thr)
    except Exception as e:
        print(f"[YOLO] Warning: Failed to re-plot F vs confidence for test: {e}")
    try:
        plot_tcr_vs_nmr_curves(test_metrics_df, out_dir, selected_threshold=selected_thr)
    except Exception as e:
        print(f"[YOLO] Warning: Failed to re-plot TCR vs NMR for test: {e}")

    selected_test_row = find_metrics_for_threshold(test_metrics_df, "top1", selected_thr)
    test_best_thr, test_best_row = select_best_threshold(test_metrics_df, strategy="top1")

    def _extract_metric_payload(row: dict) -> dict:
        return {
            "TCR": float(row["TCR"]),
            "NMR": float(row["NMR"]),
            "CMR": float(row["CMR"]),
            "F": float(row["F"]),
            "ACC": float(row["ACC"]),
        }

    # Extract metrics for logging
    val_metrics = _extract_metric_payload(val_row)
    test_selected_metrics = _extract_metric_payload(selected_test_row)
    test_best_metrics = _extract_metric_payload(test_best_row)

    print(f"[YOLO] Validation metrics at selected threshold {selected_thr:.3f}:")
    print(f"       TCR={val_metrics['TCR']:.3f}, NMR={val_metrics['NMR']:.3f}, CMR={val_metrics['CMR']:.3f}, F={val_metrics['F']:.3f}")
    print(f"[YOLO] Test metrics at validation-selected threshold {selected_thr:.3f}:")
    print(f"       TCR={test_selected_metrics['TCR']:.3f}, NMR={test_selected_metrics['NMR']:.3f}, CMR={test_selected_metrics['CMR']:.3f}, F={test_selected_metrics['F']:.3f}")
    print(f"[YOLO] Test best threshold: {test_best_thr:.3f} (F={test_best_metrics['F']:.3f})")

    selection_summary = {
        "strategy": "top1",
        "selected_threshold": selected_thr,
        "validation_metrics": val_metrics,
        "test_metrics_at_selected_threshold": test_selected_metrics,
        "test_best_threshold": test_best_thr,
        "test_best_metrics": test_best_metrics,
    }
    selection_path = out_dir / "selected_threshold_summary.json"
    selection_path.write_text(json.dumps(selection_summary, indent=2))
    append_selection_to_config(config_path, selection_summary)

    # Report total timing
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"[YOLO] Run completed in {hours:02d}:{minutes:02d}:{seconds:02d} (HH:MM:SS)")
    print(f"[YOLO] Total wall-clock time: {total_time:.1f} seconds")

    print(f"[YOLO] Evaluation complete. Results saved to:")
    print(f"       {selection_path}")
    print(f"       {out_dir / f'{model_name}_metrics_confidence_sweep.csv'}")
    print(f"       {cm_path}")


if __name__ == "__main__":
    main()
