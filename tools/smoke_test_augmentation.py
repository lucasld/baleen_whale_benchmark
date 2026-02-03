#!/usr/bin/env python3
"""
Lightweight smoke test for augmentation presets using Ultralytics `fraction`.

This runs a short YOLO training on a fraction of a fold's dataset to verify
that augmentations (especially NoiseMix) behave correctly without paying the
full training cost. Outputs a normal YOLO run folder under the fold's
`yolo/runs_det/` so you can inspect logs/curves if needed.

Usage example (2–5 minute sanity check):
  python tools/smoke_test_augmentation.py \
    --run-dir outputs/cnn_results/251008_160341 \
    --fold fold_BallenyIslands2015_noise_0.25 \
    --preset noise_mix \
    --fraction 0.02 --epochs 3 --batch 16 --model-size n --imgsz 320 \
    --name smoke_nm_frac2
"""

from __future__ import annotations

# Add base directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from src.yolo.aug.noise_mix import NoiseMix

# Supported model sizes
MODEL_SIZES = ("n", "m", "x")
NOISE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test YOLO augmentations on a small data fraction.")
    p.add_argument("--run-dir", type=Path, required=True, help="CNN run directory (outputs/cnn_results/<run_id>).")
    p.add_argument("--fold", type=str, required=True, help="Fold name, e.g., fold_BallenyIslands2015_noise_0.25.")
    p.add_argument("--preset", choices=["defaults", "spec_opt", "noise_mix"], default="noise_mix", help="Augmentation preset.")
    p.add_argument("--fraction", type=float, default=0.02, help="Fraction of dataset to use for training (0 < f ≤ 1).")
    p.add_argument("--epochs", type=int, default=3, help="Epochs for the smoke test.")
    p.add_argument("--batch", type=int, default=16, help="Batch size for the smoke test.")
    p.add_argument("--imgsz", type=int, default=320, help="Training image size.")
    p.add_argument("--model-size", choices=MODEL_SIZES, default="n", help="YOLO12 model size (n/m/x).")
    p.add_argument("--device", type=str, default="0", help="GPU index or 'cpu'.")
    p.add_argument("--seed", type=int, default=42, help="Random seed passed to Ultralytics.")
    p.add_argument("--deterministic", action="store_true", default=False, help="Enable deterministic mode (slower).")
    p.add_argument("--name", type=str, default=None, help="Run name under fold/yolo/runs_det/.")
    p.add_argument("--alpha", type=float, default=0.25, help="NoiseMix alpha (blend factor) when preset=noise_mix.")
    p.add_argument("--p", type=float, default=0.5, help="NoiseMix probability when preset=noise_mix.")
    p.add_argument("--weights", type=str, default=None, help="Optional explicit weights path (overrides model-size).")
    return p.parse_args()


def resolve_weights(args: argparse.Namespace, repo_root: Path) -> str:
    if args.weights:
        return str(Path(args.weights).expanduser().resolve())
    model_size = args.model_size.lower()
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unsupported model_size '{model_size}'. Expected one of {MODEL_SIZES}.")
    candidate = repo_root / "models" / f"yolo12{model_size}.pt"
    if candidate.exists():
        return str(candidate.resolve())
    # Fallback: rely on Ultralytics model name (must be available in env)
    return f"yolo12{model_size}.pt"


def collect_noise_paths(ds_root: Path) -> List[Path]:
    labels_dir = ds_root / "labels" / "train"
    images_dir = ds_root / "images" / "train"
    if not labels_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(f"Expected train labels/images under {ds_root}")
    noise_paths: List[Path] = []
    for label_path in labels_dir.glob("*.txt"):
        if label_path.stat().st_size == 0:
            stem = label_path.stem
            for ext in NOISE_EXTS:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    noise_paths.append(candidate)
                    break
    if not noise_paths:
        raise RuntimeError("NoiseMix preset selected but no noise images were found in the training split.")
    return noise_paths


def build_aug_params(preset: str, alpha: float, prob: float, noise_paths: List[Path]):
    """
    Return (train_kwargs_overrides, albumentations_transforms)
    compatible with Ultralytics .train(... augmentations=...).
    """
    aug_params = {
        "mosaic": None,
        "degrees": None,
        "translate": None,
        "scale": None,
        "shear": None,
        "perspective": None,
        "flipud": None,
        "fliplr": None,
        "hsv_h": None,
        "hsv_s": None,
        "hsv_v": None,
        "mixup": None,
        "cutmix": None,
        "copy_paste": None,
    }
    albumentations_transforms = None

    if preset in {"spec_opt", "noise_mix"}:
        # Disable spatial/color augs for spectrograms
        aug_params.update(
            {
                "mosaic": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "mixup": 0.0,
                "cutmix": 0.0,
                "copy_paste": 0.0,
            }
        )
    if preset == "noise_mix":
        albumentations_transforms = [NoiseMix(noise_paths=noise_paths, alpha=alpha, p=prob)]
    return aug_params, albumentations_transforms


def write_yaml(path: Path, data: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    if not (0 < args.fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")

    repo_root = Path(__file__).resolve().parents[1]
    fold_dir = args.run_dir / args.fold
    ds_root = fold_dir / "yolo_dataset"
    if not ds_root.exists():
        raise FileNotFoundError(f"YOLO dataset not found at {ds_root}")

    # Prepare output dir under the fold
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"smoke_{args.preset}_frac{args.fraction:.3f}_{timestamp}"
    out_dir = fold_dir / "yolo" / "runs_det" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data YAML for this smoke run
    data_yaml = out_dir / "data.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(ds_root.resolve()),
            "train": str((ds_root / "images" / "train").resolve()),
            "val": str((ds_root / "images" / "valid").resolve()),
            "test": str((ds_root / "images" / "test").resolve()),
            "names": ["20Hz20Plus", "ABZ", "DDswp"],  # matches labels.json (Noise excluded)
            "nc": 3,
            "workers": 8,
        },
    )

    noise_paths: List[Path] = []
    if args.preset == "noise_mix":
        noise_paths = collect_noise_paths(ds_root)

    aug_params, albumentations_transforms = build_aug_params(
        preset=args.preset,
        alpha=args.alpha,
        prob=args.p,
        noise_paths=noise_paths,
    )

    weights = resolve_weights(args, repo_root)
    print(f"[smoke] Using weights: {weights}")
    print(f"[smoke] Output dir: {out_dir}")

    model = YOLO(weights)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(out_dir.parent),  # fold/yolo/runs_det
        name=out_dir.name,
        exist_ok=True,
        verbose=False,
        plots=True,
        rect=True,  # keep consistent with main pipeline default
        seed=args.seed,
        deterministic=args.deterministic,
        fraction=args.fraction,
    )

    # Apply optional overrides (Ultralytics respects None as "use default")
    for k, v in aug_params.items():
        if v is not None:
            train_kwargs[k] = v
    if albumentations_transforms:
        train_kwargs["augmentations"] = albumentations_transforms

    model.train(**train_kwargs)

    # Persist a small config snapshot for reproducibility
    config = {
        "run_dir": str(args.run_dir),
        "fold": args.fold,
        "preset": args.preset,
        "fraction": args.fraction,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "model_size": args.model_size,
        "device": args.device,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "alpha": args.alpha,
        "p": args.p,
        "weights": weights,
        "out_dir": str(out_dir),
        "data_yaml": str(data_yaml),
    }
    (out_dir / "smoke_config.json").write_text(json.dumps(config, indent=2))
    print(f"[smoke] Completed. Results in: {out_dir}")


if __name__ == "__main__":
    main()

