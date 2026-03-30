#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_plots.config import (
    DEFAULT_EXPLORATORY_FOLD,
    DEFAULT_RUN_DIR,
    DEFAULT_RUN_NAME,
    build_output_paths,
)
from thesis_plots.loaders import (
    load_exploratory_points,
    load_exploratory_sweeps,
    load_final_by_fold_comparison,
    load_final_selected_summary,
    load_final_selected_thresholds,
    load_final_sweep_rows,
    load_multiclass_per_fold,
    load_multiclass_summary,
    load_run_colors,
    summarize_final_sweep_rows,
)
from thesis_plots.manifest import get_manifest
from thesis_plots.theme import apply_theme, save_figure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis-ready figure set from saved benchmark artifacts.")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scope", choices=["all", "main", "appendix"], default="all")
    p.add_argument("--figures", type=str, default="")
    p.add_argument("--formats", type=str, default="png")
    return p.parse_args()


def ensure_dirs(root: Path):
    paths = build_output_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.main.mkdir(parents=True, exist_ok=True)
    paths.appendix.mkdir(parents=True, exist_ok=True)
    return paths


def build_shared_data(run_dir: Path, run_name: str, requested_specs: list) -> dict:
    figure_ids = {spec.figure_id for spec in requested_specs}
    data = {
        "run_name": run_name,
        "run_colors": load_run_colors(),
    }
    if any(fid.startswith("exploratory_") for fid in figure_ids):
        exploratory_points = load_exploratory_points(run_dir, DEFAULT_EXPLORATORY_FOLD)
        exploratory_run_names = sorted(set(exploratory_points["run_name"]) - {"CNN"})
        data["exploratory_points"] = exploratory_points
        data["exploratory_sweeps"] = load_exploratory_sweeps(run_dir, DEFAULT_EXPLORATORY_FOLD, exploratory_run_names)
    if "exploratory_decision_summary" in figure_ids and "exploratory_points" not in data:
        data["exploratory_points"] = load_exploratory_points(run_dir, DEFAULT_EXPLORATORY_FOLD)

    if any(fid.startswith("final_") for fid in figure_ids):
        data["final_selected_summary"] = load_final_selected_summary(run_dir, run_name)
        data["final_selected_thresholds"] = load_final_selected_thresholds(run_dir, run_name)
        data["final_by_fold"] = load_final_by_fold_comparison(run_dir, run_name)
    if any(
        fid in figure_ids
        for fid in [
            "final_operating_tradeoff_tcr_vs_nmr",
            "final_per_fold_tcr_vs_nmr",
            "final_f_vs_confidence",
            "final_tcr_vs_confidence",
            "final_nmr_vs_confidence",
        ]
    ):
        final_sweep_rows = load_final_sweep_rows(run_dir, run_name)
        data["final_sweep_rows"] = final_sweep_rows
        data["final_sweep_summary"] = summarize_final_sweep_rows(final_sweep_rows, run_name)

    if any(fid.startswith("multiclass_") for fid in figure_ids):
        data["multiclass_summary"] = load_multiclass_summary(run_dir, run_name)
        data["multiclass_per_fold"] = load_multiclass_per_fold(run_dir, run_name)

    return data


def select_specs(manifest: dict, scope: str, figure_ids: list[str]) -> list:
    specs = list(manifest.values())
    if scope != "all":
        specs = [s for s in specs if s.scope == scope]
    if figure_ids:
        wanted = set(figure_ids)
        missing = wanted - set(manifest.keys())
        if missing:
            raise SystemExit(f"Unknown figure ids: {', '.join(sorted(missing))}")
        specs = [manifest[fid] for fid in figure_ids if manifest[fid] in specs or scope == "all"]
    return specs


def main() -> None:
    args = parse_args()
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    out = ensure_dirs(args.output_dir)
    apply_theme()
    manifest = get_manifest(args.run_name)
    figure_ids = [x.strip() for x in args.figures.split(",") if x.strip()]
    specs = select_specs(manifest, args.scope, figure_ids)
    data = build_shared_data(args.run_dir, args.run_name, specs)

    written = []
    for spec in specs:
        fig = spec.builder(data)
        base = (out.main if spec.scope == "main" else out.appendix) / spec.output_stem
        files = save_figure(fig, base, formats)
        written.append(
            {
                "figure_id": spec.figure_id,
                "scope": spec.scope,
                "output_stem": spec.output_stem,
                "files": files,
                "source_artifacts": spec.source_artifacts,
            }
        )

    manifest_path = out.root / "figure_manifest.json"
    manifest_path.write_text(json.dumps({"run_dir": str(args.run_dir), "run_name": args.run_name, "figures": written}, indent=2))
    print(f"Wrote {len(written)} figures under {out.root}")


if __name__ == "__main__":
    main()
