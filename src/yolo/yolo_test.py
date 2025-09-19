"""
YOLO classification training + evaluation helper.

Behavior:
- Auto-detect the latest CNN run under outputs/cnn_results unless --run_dir is provided
- Pick a base fold (or user-provided) to train on using its yolo_dataset/images/{train,val}
- Create a dataset YAML for training (train/val only)
- Train once starting from provided weights
- Evaluate the trained classifier on every fold's per-noise test set created by new_test.py:
    <run_dir>/fold_*/yolo_dataset_test_noise*/images/test
- For each test set, create a small YAML that reuses the base train/val and sets test accordingly, then run model.val(split='test').

Note: This script expects that CNN training already created yolo_dataset for each fold (train/val),
and that CNN evaluation (new_test.py) created per-noise yolo_dataset_test_noise* trees for test sets.
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging

# --- Environment setup (before importing Ultralytics) ---
# Disable fancy tqdm bars in SLURM logs
os.environ["RICH_PROGRESS_BAR"] = "0"
# Suppress most chatter, but still allow INFO logs and our callback
os.environ["ULTRALYTICS_QUIET"] = "1"

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Keep INFO so we see one-liners like "Starting training..."
LOGGER.setLevel(logging.WARNING)

def find_latest_run(cnn_results_root: Path) -> Path:
    runs = [p for p in cnn_results_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found under {cnn_results_root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def select_train_fold(fold_dirs: List[Path], requested_fold: Optional[str]) -> Path:
    """Pick a fold that has train/val images. If requested_fold is provided, validate it."""
    if requested_fold:
        candidate = next((p for p in fold_dirs if p.name == requested_fold), None)
        if candidate is None:
            raise FileNotFoundError(f"Requested fold not found: {requested_fold}")
        base_yolo = candidate / "yolo_dataset"
        has_train = (base_yolo / "images" / "train").exists()
        has_valid = (base_yolo / "images" / "valid").exists() or (base_yolo / "images" / "val").exists()
        if not (has_train and has_valid):
            raise FileNotFoundError(f"Requested fold missing train/val under {base_yolo}")
        return candidate

    for p in fold_dirs:
        base_yolo = p / "yolo_dataset"
        has_train = (base_yolo / "images" / "train").exists()
        has_valid = (base_yolo / "images" / "valid").exists() or (base_yolo / "images" / "val").exists()
        if has_train and has_valid:
            return p
    raise FileNotFoundError("No fold with yolo_dataset/images/{train,val} found.")


def find_latest_best_weights(yolo_runs_root: Path) -> Optional[Path]:
    """Find most recent best.pt under yolo_runs/*/weights/best.pt inside a run dir."""
    if not yolo_runs_root.exists():
        return None
    candidates = []
    for run in yolo_runs_root.iterdir():
        if not run.is_dir():
            continue
        best = run / "weights" / "best.pt"
        if best.exists():
            candidates.append(best)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensure_symlink(target: Path, link_path: Path):
    """Create a symlink at link_path pointing to target. Replace if it exists and is different."""
    try:
        if link_path.is_symlink() or link_path.exists():
            # If already correct, keep; else remove and recreate
            try:
                current = Path(os.readlink(link_path)) if link_path.is_symlink() else None
            except OSError:
                current = None
            if current is None or current.resolve() != target.resolve():
                if link_path.is_dir() and not link_path.is_symlink():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink(missing_ok=True)
                link_path.symlink_to(target, target_is_directory=True)
        else:
            # Parent must exist
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(target, target_is_directory=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create symlink {link_path} -> {target}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO classifier and evaluate on per-noise test sets.")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to CNN run folder (outputs/cnn_results/YYMMDD_...).")
    parser.add_argument("--train_fold", type=str, default=None, help="Fold name to train on (e.g., fold_BallenyIslands2015_noise_0.25). Auto-pick if omitted.")
    parser.add_argument("--weights", type=str, default="yolo11n-cls.pt", help="Weights file. If training, used as init weights; if --skip_train, used as eval weights.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=str, default="0", help="GPU index or 'cpu'.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run evaluation with provided or latest best weights.")
    parser.add_argument("--name", type=str, default=None, help="Name for the YOLO run under yolo_runs/. If omitted, uses 'cls' and overwrites.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing yolo_runs/<name> directory if it exists.")
    args = parser.parse_args()

    base_dir = Path(os.getcwd())
    results_root = base_dir / "outputs" / "cnn_results"

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(results_root)
    print(f"[YOLO] Using run_dir: {run_dir}")

    # Discover folds present in the run
    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {run_dir}")
    # Pick a fold that has train/val images
    train_fold_dir = select_train_fold(fold_dirs, args.train_fold)
    print(f"[YOLO] Using fold for train/val: {train_fold_dir.name}")

    base_yolo_dir = train_fold_dir / "yolo_dataset"
    train_dir = (base_yolo_dir / "images" / "train").resolve()
    # Prefer 'valid', but fallback to 'val' if needed
    val_candidate_valid = (base_yolo_dir / "images" / "valid").resolve()
    val_candidate_val = (base_yolo_dir / "images" / "val").resolve()
    val_dir = val_candidate_valid if val_candidate_valid.exists() else val_candidate_val
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Train/val directories not found under {base_yolo_dir}")

    # Train
    model = YOLO(args.weights)
    # Place YOLO outputs under the training fold to keep everything self-contained per fold
    out_project = train_fold_dir / "yolo_runs"
    # Default to a single fixed directory 'cls' and overwrite it to avoid clutter
    if not args.name:
        args.name = "cls"
        args.overwrite = True
    out_dir = out_project / args.name
    if out_dir.exists() and args.overwrite:
        print(f"[YOLO] overwrite: removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # For training, pass the images root directly to Ultralytics to avoid YAML handling issues
    train_images_root = (base_yolo_dir / "images").resolve()
    print(f"[YOLO] Using training dataset root: {train_images_root}")

    # Train if requested
    if not args.skip_train:
        # Add concise per-epoch logging
        def _epoch_log(trainer):
            try:
                epoch = trainer.epoch + 1
                total = trainer.epochs
                # LR
                try:
                    lr = trainer.optimizer.param_groups[0].get('lr', None)
                except Exception:
                    lr = None
                # Train loss
                train_loss = None
                for attr in ('loss', 'tloss'):
                    v = getattr(trainer, attr, None)
                    if v is not None:
                        try:
                            train_loss = float(v) if isinstance(v, (int, float)) else float(v.item() if hasattr(v, 'item') else v)
                            break
                        except Exception:
                            pass
                # Val metrics
                vmetrics = getattr(trainer.validator, 'metrics', None)
                top1 = getattr(vmetrics, 'top1', None)
                top5 = getattr(vmetrics, 'top5', None)
                def fmt(x):
                    try:
                        return f"{float(x):.4f}"
                    except Exception:
                        return "-"
                print(f"[YOLO][epoch {epoch}/{total}] loss={fmt(train_loss)} val_top1={fmt(top1)} val_top5={fmt(top5)} lr={fmt(lr)}")
            except Exception:
                pass

        model = YOLO(args.weights)
        try:
            model.add_callback('on_fit_epoch_end', _epoch_log)
        except Exception:
            pass
        print(f"[YOLO] Starting training: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}")
        model.train(
            data=str(train_images_root),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(out_project),
            name=out_dir.name,
            exist_ok=True,
            verbose=False,
            plots=False,
            workers=4,
        )
    else:
        print("[YOLO] Skipping training (--skip_train)")

    # Reload best weights for evaluation
    # Choose weights for evaluation
    best_weights = out_project / out_dir.name / "weights" / "best.pt"
    eval_weights: Optional[Path]
    if args.skip_train:
        # If user provided weights, use them; else try latest best.pt under yolo_runs
        eval_weights = Path(args.weights) if args.weights and Path(args.weights).exists() else find_latest_best_weights(out_project)
    else:
        eval_weights = best_weights if best_weights.exists() else None
    if eval_weights and Path(eval_weights).exists():
        model = YOLO(str(eval_weights))
        print(f"[YOLO] Loaded eval weights: {eval_weights}")
    else:
        # Fall back to current model if training just ran; otherwise load init weights
        try:
            model
        except NameError:
            model = YOLO(args.weights)
            print(f"[YOLO] Warning: eval weights not found; using provided weights {args.weights}")

    # Evaluate for every fold and every noise test set. Store all eval outputs under the single out_dir.
    eval_root = out_dir / "eval"
    datasets_root = out_dir / "datasets"
    for fold_dir in fold_dirs:
        # Find any per-noise test roots
        noise_roots = sorted([p for p in fold_dir.iterdir() if p.is_dir() and p.name.startswith("yolo_dataset_test_noise")])
        if not noise_roots:
            print(f"[YOLO] No per-noise test datasets found in {fold_dir.name}; skipping.")
            continue
        for noise_root in noise_roots:
            test_dir = (noise_root / "images" / "test").resolve()
            if not test_dir.exists():
                print(f"[YOLO] Test images not found at {test_dir}; skipping noise set {noise_root.name}.")
                continue
            # Create per-fold/per-noise eval folder under the single run folder
            fold_eval_dir = eval_root / fold_dir.name / noise_root.name
            fold_eval_dir.mkdir(parents=True, exist_ok=True)
            # Prepare a directory-based dataset that has train/val/test by symlinking to sources
            ds_noise_images_root = datasets_root / fold_dir.name / noise_root.name / "images"
            # Symlink train and val/valid from the base training fold
            ensure_symlink(train_dir, ds_noise_images_root / "train")
            ensure_symlink(val_dir, ds_noise_images_root / ("valid" if val_candidate_valid.exists() else "val"))
            # Symlink test from the noise root
            ensure_symlink(test_dir, ds_noise_images_root / "test")
            print(f"[YOLO] Evaluating on {fold_dir.name} | {noise_root.name}")
            metrics = model.val(
                data=str(ds_noise_images_root),
                split="test",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                verbose=False,
                plots=False,
                workers=4,
            )
            # Print single-line evaluation summary if available
            try:
                d = getattr(metrics, 'results_dict', None)
                if isinstance(d, dict) and d:
                    keys = [k for k in d if any(x in k.lower() for x in ['acc', 'top', 'loss'])]
                    msg = " ".join([f"{k}={d[k]:.4f}" for k in keys])
                    if msg:
                        print(f"[YOLO][eval] {fold_dir.name} {noise_root.name} {msg}")
            except Exception:
                pass
            # Persist a brief summary next to the dataset for traceability
            try:
                summary_path = fold_eval_dir / "eval_metrics.txt"
                with summary_path.open("w") as f:
                    f.write(str(metrics))
                print(f"[YOLO] Wrote metrics to {summary_path}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
