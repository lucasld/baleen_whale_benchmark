#!/usr/bin/env python3
"""
Validation suite for YOLO spectrogram augmentations.
Performs mathematical unit tests and generates visual verification grids.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import json
import time
import random
from typing import List, Tuple

# Add workspace root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.yolo.aug.noise_mix import NoiseMix

def test_blending_logic():
    """Unit test: Verify alpha blending math is correct."""
    print("[test] Verifying blending math...")
    # Create dummy images
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    noise_img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    
    # Save dummy noise image to temp
    temp_noise = Path("temp_noise_unit_test.png")
    cv2.imwrite(str(temp_noise), noise_img)
    
    try:
        # Test 1: Alpha 0.5 (equal mix)
        alpha = 0.5
        aug = NoiseMix(noise_paths=[temp_noise], alpha=alpha, p=1.0)
        result = aug.apply(img)
        # Expected: 0*0.5 + 200*0.5 = 100
        assert np.allclose(result[0,0], [100, 100, 100], atol=1), f"Expected 100, got {result[0,0]}"
        
        # Test 2: Alpha 1.0 (pure noise)
        alpha = 1.0
        aug = NoiseMix(noise_paths=[temp_noise], alpha=alpha, p=1.0)
        result = aug.apply(img)
        assert np.all(result == 200), f"Expected 200, got {result[0,0]}"
        
        # Test 3: Alpha 0.0 (no change)
        alpha = 0.0
        aug = NoiseMix(noise_paths=[temp_noise], alpha=alpha, p=1.0)
        result = aug.apply(img)
        assert np.all(result == 0), f"Expected 0, got {result[0,0]}"
        
        print("[test] \u2705 Blending math verified.")
    finally:
        if temp_noise.exists():
            temp_noise.unlink()

def test_label_invariance():
    """Unit test: Verify labels (bounding boxes) are NOT modified by NoiseMix."""
    print("[test] Verifying label invariance...")
    # Mock some data
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_noise = Path("dummy_noise.png")
    cv2.imwrite(str(dummy_noise), img)
    
    try:
        aug = NoiseMix(noise_paths=[dummy_noise], alpha=0.5, p=1.0)
        
        # Albumentations standard usage: targets are passed as kwargs
        # ImageOnlyTransform should ignore labels
        dummy_boxes = [[0.5, 0.5, 0.2, 0.2]]
        result = aug(image=img, bboxes=dummy_boxes)
        
        assert np.array_equal(result["bboxes"], dummy_boxes), "ERROR: Bounding boxes were modified!"
        print("[test] \u2705 Label invariance verified.")
    finally:
        if dummy_noise.exists():
            dummy_noise.unlink()

def get_samples(ds_root: Path, n: int = 3) -> List[Tuple[Path, Path]]:
    images_dir = ds_root / "images" / "train"
    labels_dir = ds_root / "labels" / "train"
    
    pairs = []
    if not images_dir.exists(): return []
    for img in sorted(images_dir.glob("*.png")):
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:
            pairs.append((img, lbl))
            if len(pairs) >= n:
                break
    return pairs

def get_noises(ds_root: Path, n: int = 5) -> List[Path]:
    labels_dir = ds_root / "labels" / "train"
    images_dir = ds_root / "images" / "train"
    noises = []
    if not labels_dir.exists(): return []
    for lbl in sorted(labels_dir.glob("*.txt")):
        if lbl.stat().st_size == 0:
            img = images_dir / f"{lbl.stem}.png"
            if img.exists():
                noises.append(img)
                if len(noises) >= n:
                    break
    return noises

def draw_yolo_boxes(img, label_path):
    h, w = img.shape[:2]
    out = img.copy()
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts: continue
        cls, x, y, bw, bh = map(float, parts)
        cx, cy, ww, hh = x * w, y * h, bw * w, bh * h
        x1, y1 = int(cx - ww/2), int(cy - hh/2)
        x2, y2 = int(cx + ww/2), int(cy + hh/2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return out

def generate_visual_grid(ds_root: Path, out_path: Path):
    print(f"[viz] Generating visual grid to {out_path}...")
    samples = get_samples(ds_root, n=3)
    noises = get_noises(ds_root, n=5)
    
    if not samples or not noises:
        print("[viz] \u274c Could not find enough samples/noises for visualization.")
        return

    # Presets
    aug_normal = NoiseMix(noise_paths=noises, alpha=0.25, p=1.0)
    aug_extreme = NoiseMix(noise_paths=noises, alpha=0.8, p=1.0)
    
    rows = []
    # Header size calculation
    sample_w = 320
    sample_h = 107
    grid_w = sample_w * 3
    
    header = np.zeros((50, grid_w, 3), dtype=np.uint8)
    cv2.putText(header, "Original | Augmented (alpha=0.25) | Extreme (alpha=0.8)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(header)

    for img_p, lbl_p in samples:
        img = cv2.imread(str(img_p))
        if img is None: continue
        img = cv2.resize(img, (sample_w, sample_h)) 
        
        # 1. Original
        col1 = draw_yolo_boxes(img, lbl_p)
        
        # 2. Normal Aug
        res_n = aug_normal(image=img)["image"]
        col2 = draw_yolo_boxes(res_n, lbl_p)
        
        # 3. Extreme Aug
        res_e = aug_extreme(image=img)["image"]
        col3 = draw_yolo_boxes(res_e, lbl_p)
        
        row = np.hstack([col1, col2, col3])
        rows.append(row)
        # Spacer
        rows.append(np.zeros((10, grid_w, 3), dtype=np.uint8))

    full_grid = np.vstack(rows)
    cv2.imwrite(str(out_path), full_grid)
    print(f"[viz] \u2705 Grid saved.")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--fold", type=str, required=True)
    args = p.parse_args()
    
    ds_root = args.run_dir / args.fold / "yolo_dataset"
    if not ds_root.exists():
        print(f"Error: Dataset not found at {ds_root}")
        sys.exit(1)

    # 1. Run Unit Tests
    print("-" * 30)
    test_blending_logic()
    test_label_invariance()
    print("-" * 30)
    
    # 2. Run Visual Validation
    out_dir = Path("augmentation_validation")
    out_dir.mkdir(exist_ok=True)
    grid_path = out_dir / f"validation_grid_{args.fold}.png"
    generate_visual_grid(ds_root, grid_path)
    
    print(f"\n[SUMMARY] Validation complete.")
    print(f"1. Mathematical tests: PASSED")
    print(f"2. Visual approval needed: check '{grid_path}'")
    print("If the whale calls are still visible but faint in the 'Extreme' column, and labels align, you are ready.")

if __name__ == "__main__":
    main()
