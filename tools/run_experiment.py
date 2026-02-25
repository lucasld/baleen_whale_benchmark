#!/usr/bin/env python3
"""Dispatch YOLO detection experiments defined in experiments/registry.yaml."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml

BOOL_FLAGS_WITH_NEGATION = {
    # For a registry key `k` with value False, we emit `--{neg}` below.
    # Keys here must match argparse dest names; values must match the
    # corresponding long option *without* leading dashes.
    "rect": "no-rect",
    "pretrained": "no-pretrained",
    "deterministic": "no-deterministic",
    "conf_sweep": "no-conf_sweep",
}


def load_registry(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def merge_args(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def build_cli_args(k: str, v: Any) -> list[str]:
    """Convert a single registry key/value into CLI arguments.

    Registry keys are expected to match argparse dest names (e.g. `eval_conf`,
    `model_size`, `train_fold`, `conf_sweep`). We emit `--{key}` directly for
    non-boolean values, and `--{key}` / `--no-{...}` for booleans where a
    negated flag exists.
    """
    if isinstance(v, bool):
        neg = BOOL_FLAGS_WITH_NEGATION.get(k)
        if v:
            # Positive case: use the key name directly, e.g. `--pretrained`.
            return [f"--{k}"]
        if not neg:
            raise ValueError(f"Boolean flag '{k}' does not support automatic --no- form.")
        # Negative case: emit the explicitly-configured negated flag name,
        # e.g. `--no-pretrained`, `--no-rect`, `--no-conf_sweep`.
        return [f"--{neg}"]
    # For non-booleans, use the key name as-is so that `eval_conf` → `--eval_conf`.
    return [f"--{k}", str(v)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO experiments from registry.")
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry.yaml"))
    parser.add_argument("--run-dir", type=str, required=True, help="CNN run directory (outputs/cnn_results/YYMMDD_...).")
    parser.add_argument("--ids", type=str, required=True, help="Comma-separated experiment IDs to run.")
    parser.add_argument(
        "--fold-override",
        type=str,
        default="",
        help="Force a single fold for all requested IDs (useful for SLURM array dispatch).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--extra", type=str, default="", help="Additional CLI args appended to every command.")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    base_args = registry.get("base_args", {})
    exp_index = {exp["id"]: exp for exp in registry.get("experiments", [])}
    requested = [eid.strip() for eid in args.ids.split(",") if eid.strip()]
    missing = [eid for eid in requested if eid not in exp_index]
    if missing:
        raise SystemExit(f"Experiment IDs not found in registry: {missing}")

    base_cmd = ["python", "-u", "src/yolo/yolo_detect.py", "--run_dir", args.run_dir]
    extra_tokens = shlex.split(args.extra)
    forced_fold = args.fold_override.strip()
    if forced_fold:
        forced_fold_path = Path(args.run_dir) / forced_fold
        if not forced_fold_path.is_dir():
            raise SystemExit(f"--fold-override '{forced_fold}' not found under {args.run_dir}")

    for eid in requested:
        exp = exp_index[eid]
        exp_args = merge_args(base_args, exp.get("args", {}))
        
        # Identify target folds: explicit in registry, or all found in run_dir
        if forced_fold:
            target_folds = [forced_fold]
        elif exp.get("fold"):
            target_folds = [exp["fold"]]
        else:
            # Auto-discover all folds if not specified (e.g. for confirmatory runs)
            run_path = Path(args.run_dir)
            target_folds = sorted([p.name for p in run_path.glob("fold_*") if p.is_dir()])
            if not target_folds:
                print(f"[run_experiment] Warning: No 'fold_*' directories found in {args.run_dir} for ID {eid}")
                continue
            print(f"[run_experiment] ID {eid} has no fixed fold. Running on {len(target_folds)} detected folds: {target_folds}")

        for fold_name in target_folds:
            # Create a localized args copy for this fold
            current_args = dict(exp_args)
            current_args["train_fold"] = fold_name
            
            cmd = list(base_cmd)
            for key, value in current_args.items():
                cmd.extend(build_cli_args(key, value))
            
            cmd.extend(extra_tokens)
            print(f"[run_experiment] [{eid}::{fold_name}]", " ".join(shlex.quote(c) for c in cmd))
            
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)
if __name__ == "__main__":
    main()
