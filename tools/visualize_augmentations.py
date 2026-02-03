#!/usr/bin/env python3
"""
Visualize how the custom augmentations affect spectrograms.

Outputs paired original/augmented images (and optional diffs) that can be
included in the thesis. This is a read-only visualization tool; it does NOT
train anything.

Usage example:
  python tools/visualize_augmentations.py \
    --run-dir outputs/cnn_results/251008_160341 \
    --fold fold_BallenyIslands2015_noise_0.25 \
    --preset noise_mix --n 8 --seed 42 \
    --out-dir ./aug_viz_examples
"""

from __future__ import annotations

import sys
import os
# Add workspace root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np

# Local imports
from src.yolo.aug.noise_mix import NoiseMix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize spectrogram augmentations.")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="CNN run directory (outputs/cnn_results/<run_id>).",
    )
    p.add_argument(
        "--fold",
        type=str,
        required=True,
        help="Fold name, e.g., fold_BallenyIslands2015_noise_0.25",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Which split to draw labeled samples from (default: train).",
    )
    p.add_argument(
        "--preset",
        choices=["defaults", "spec_opt", "noise_mix"],
        default="noise_mix",
        help="Augmentation preset to visualize (default: noise_mix).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="NoiseMix alpha (blend factor) when preset=noise_mix.",
    )
    p.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="NoiseMix application probability when preset=noise_mix.",
    )
    p.add_argument(
        "--max-noise",
        type=int,
        default=200,
        help="Maximum number of noise images to collect (speeds up large folds).",
    )
    p.add_argument("--n", type=int, default=8, help="Number of samples to visualize.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for images/metadata. "
        "If omitted, a timestamped folder under the fold will be created.",
    )
    return p.parse_args()


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def load_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (cls, x_center, y_center, width, height) in YOLO format."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append((cls, x, y, w, h))
    return boxes


def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, float, float, float, float]]) -> np.ndarray:
    h, w, _ = image.shape
    out = image.copy()
    for cls, x, y, bw, bh in boxes:
        cx, cy, ww, hh = x * w, y * h, bw * w, bh * h
        x1 = int(cx - ww / 2)
        y1 = int(cy - hh / 2)
        x2 = int(cx + ww / 2)
        y2 = int(cy + hh / 2)
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(out, str(cls), (x1, max(y1 - 2, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def collect_noise_paths(ds_root: Path) -> List[Path]:
    labels_dir = ds_root / "labels" / "train"
    images_dir = ds_root / "images" / "train"
    if not labels_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(f"Missing YOLO dataset under {ds_root}")
    noise: List[Path] = []
    for label_path in labels_dir.glob("*.txt"):
        if label_path.stat().st_size == 0:
            stem = label_path.stem
            candidate = images_dir / f"{stem}.png"
            if candidate.exists():
                noise.append(candidate)
                if len(noise) >= getattr(collect_noise_paths, "_max_noise", 200):
                    break
    if not noise:
        raise RuntimeError("No noise images (empty label files) found in the training split.")
    return noise


def select_samples(images_dir: Path, labels_dir: Path, n: int, seed: int) -> List[Tuple[Path, Path]]:
    pairs = []
    for img in images_dir.glob("*.png"):
        label = labels_dir / f"{img.stem}.txt"
        if label.exists() and label.stat().st_size > 0:
            pairs.append((img, label))
    if not pairs:
        raise RuntimeError(f"No labeled samples found in {labels_dir}")
    random.seed(seed)
    random.shuffle(pairs)
    return pairs[:n]


def build_transform(preset: str, noise_paths: List[Path], alpha: float, p: float):
    if preset == "noise_mix":
        return A.Compose([NoiseMix(noise_paths=noise_paths, alpha=alpha, p=p)])
    if preset == "spec_opt":
        # No-op transform for visualization; spec_opt mainly disables defaults.
        return A.Compose([])
    return A.Compose([])  # defaults -> no custom augmentation here


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    fold_dir = args.run_dir / args.fold
    ds_root = fold_dir / "yolo_dataset"
    images_dir = ds_root / "images" / args.split
    labels_dir = ds_root / "labels" / args.split

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected YOLO dataset at {images_dir} / {labels_dir}")

    out_dir = args.out_dir
    if out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = fold_dir / "yolo" / "aug_viz" / f"{args.preset}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Speed up noise scan: limit to max_noise and png only
    collect_noise_paths._max_noise = args.max_noise  # type: ignore[attr-defined]
    noise_paths = collect_noise_paths(ds_root) if args.preset == "noise_mix" else []
    transform = build_transform(args.preset, noise_paths, args.alpha, args.p)

    samples = select_samples(images_dir, labels_dir, args.n, args.seed)

    metadata = {
        "run_dir": str(args.run_dir),
        "fold": args.fold,
        "split": args.split,
        "preset": args.preset,
        "alpha": args.alpha,
        "p": args.p,
        "n": args.n,
        "seed": args.seed,
        "noise_count": len(noise_paths),
        "samples": [],
    }

    for idx, (img_path, lbl_path) in enumerate(samples):
        img = load_image(img_path)
        boxes = load_boxes(lbl_path)
        viz_original = draw_boxes(img, boxes)

        aug_img = transform(image=img)["image"] if transform is not None else img
        viz_aug = draw_boxes(aug_img, boxes)

        base_name = f"sample_{idx:02d}"
        orig_path = out_dir / f"{base_name}_original.png"
        aug_path = out_dir / f"{base_name}_augmented.png"
        diff_path = out_dir / f"{base_name}_diff.png"

        cv2.imwrite(str(orig_path), viz_original)
        cv2.imwrite(str(aug_path), viz_aug)

        # Optional difference image (absolute difference)
        diff = cv2.absdiff(viz_original, viz_aug)
        cv2.imwrite(str(diff_path), diff)

        metadata["samples"].append(
            {
                "image": str(img_path),
                "label": str(lbl_path),
                "augmented": str(aug_path),
                "original": str(orig_path),
                "diff": str(diff_path),
            }
        )

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[viz] Wrote {len(samples)} pairs to {out_dir}")
    print(f"[viz] Metadata: {meta_path}")


if __name__ == "__main__":
    main()


