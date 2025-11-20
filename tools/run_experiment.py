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
    flag = k.replace("_", "-")
    if isinstance(v, bool):
        neg = BOOL_FLAGS_WITH_NEGATION.get(k)
        if v:
            return [f"--{flag}"]
        if not neg:
            raise ValueError(f"Boolean flag '{k}' does not support automatic --no- form.")
        return [f"--{neg}"]
    if isinstance(v, (list, tuple)):
        args = []
        for item in v:
            args.extend(build_cli_args(k, item))
        return args
    return [f"--{flag}", str(v)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO experiments from registry.")
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry.yaml"))
    parser.add_argument("--run-dir", type=str, required=True, help="CNN run directory (outputs/cnn_results/YYMMDD_...).")
    parser.add_argument("--ids", type=str, required=True, help="Comma-separated experiment IDs to run.")
    parser.add_argument("--fold", type=str, default=None, help="Override fold for all experiments (default: use registry entry).")
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

    for eid in requested:
        exp = exp_index[eid]
        exp_args = merge_args(base_args, exp.get("args", {}))
        if args.fold:
            exp_args["train_fold"] = args.fold
        elif exp.get("fold"):
            exp_args["train_fold"] = exp["fold"]
        cmd = list(base_cmd)
        for key, value in exp_args.items():
            cmd.extend(build_cli_args(key, value))
        cmd.extend(extra_tokens)
        print("[run_experiment]", " ".join(shlex.quote(c) for c in cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)
if __name__ == "__main__":
    main()
