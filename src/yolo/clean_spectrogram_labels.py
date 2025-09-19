#!/usr/bin/env python3
"""
Create cleaned spectrogram labels by merging raw class IDs to the merged taxonomy
and dropping Noise boxes. Outputs mirror the input site folders but under a
new root: datasets/preprocessed_dataset_new/spectrograms_labels_cleaned.

Inputs:
- A CNN run root (outputs/cnn_results/<RUN_ID>/) that provides:
  - config.json: raw CATEGORIES and CATEGORIES_TO_JOIN
  - labels.json: final merged label space including Noise
- An input labels root with per-site flat label files:
  datasets/preprocessed_dataset_new/spectrograms_labels/<Site>/<stem>.txt

Outputs:
- Cleaned labels under:
  datasets/preprocessed_dataset_new/spectrograms_labels_cleaned/<Site>/<stem>.txt
  with merged class IDs (e.g., 20Hz20Plus/ABZ/DDswp) and no Noise boxes.

The script is idempotent: each site gets a sentinel .cleaned.json with stats;
if present and a quick sample validates IDs, the site is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


SENTINEL_NAME = ".cleaned.json"


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


@dataclass
class CleaningStats:
    files_total: int = 0
    files_written: int = 0
    boxes_in_raw: int = 0
    boxes_noise_dropped: int = 0
    boxes_kept_after_merge: int = 0
    empty_files_written: int = 0


@dataclass
class MergeMapping:
    old_idx_to_name: Dict[int, str]
    old_idx_to_merged_idx: Dict[int, int]
    merged_names_in_order: List[str]
    noise_old_idx: Optional[int]


def build_merge_mapping(run_root: Path) -> MergeMapping:
    cfg = read_json(run_root / "config.json")
    lbl = read_json(run_root / "labels.json")

    categories: List[str] = cfg["CATEGORIES"]
    groups: Dict[str, List[str]] = cfg["CATEGORIES_TO_JOIN"]

    # final merged taxonomy for detection (exclude Noise)
    merged_name_to_idx = {k: int(v) for k, v in lbl.items() if k.lower() != "noise"}

    raw_to_merged: Dict[str, str] = {}
    for merged_name, raw_list in groups.items():
        if merged_name not in merged_name_to_idx:
            continue
        for raw_name in raw_list:
            raw_to_merged[raw_name] = merged_name

    old_idx_to_name = {i: n for i, n in enumerate(categories)}
    noise_old_idx = None
    for i, n in old_idx_to_name.items():
        if n.lower() == "noise":
            noise_old_idx = i
            break

    old_idx_to_merged_idx: Dict[int, int] = {}
    for old_idx, raw_name in old_idx_to_name.items():
        if raw_name in raw_to_merged:
            old_idx_to_merged_idx[old_idx] = merged_name_to_idx[raw_to_merged[raw_name]]

    merged_names_in_order = [k for k, _ in sorted(merged_name_to_idx.items(), key=lambda kv: kv[1])]
    return MergeMapping(
        old_idx_to_name=old_idx_to_name,
        old_idx_to_merged_idx=old_idx_to_merged_idx,
        merged_names_in_order=merged_names_in_order,
        noise_old_idx=noise_old_idx,
    )


def is_site_cleaned(site_out: Path, valid_ids: set) -> bool:
    sentinel = site_out / SENTINEL_NAME
    if not sentinel.exists():
        return False
    # quick sample check
    scanned = 0
    for p in sorted(site_out.glob("*.txt")):
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                cid = int(float(s.split()[0]))
            except Exception:
                return False
            if cid not in valid_ids:
                return False
        scanned += 1
        if scanned >= 50:
            break
    return True


def clean_site(site_in: Path, site_out: Path, mapping: MergeMapping) -> CleaningStats:
    site_out.mkdir(parents=True, exist_ok=True)
    stats = CleaningStats()
    txts = sorted([p for p in site_in.glob("*.txt") if p.is_file()])
    stats.files_total = len(txts)
    for src in txts:
        dst = site_out / src.name
        try:
            raw_lines = src.read_text().splitlines()
        except Exception:
            raw_lines = []
        out_lines = []
        img_boxes_in_raw = 0
        img_boxes_kept = 0
        img_noise_dropped = 0

        for s in raw_lines:
            s = s.strip()
            if not s:
                continue
            parts = s.split()
            try:
                old_cid = int(float(parts[0]))
            except Exception:
                continue
            img_boxes_in_raw += 1

            if mapping.noise_old_idx is not None and old_cid == mapping.noise_old_idx:
                img_noise_dropped += 1
                continue

            if old_cid in mapping.old_idx_to_merged_idx:
                new_cid = mapping.old_idx_to_merged_idx[old_cid]
                parts[0] = str(new_cid)
                out_lines.append(" ".join(parts))
                img_boxes_kept += 1
            else:
                # unknown class for detection -> drop
                continue

        stats.boxes_in_raw += img_boxes_in_raw
        stats.boxes_kept_after_merge += img_boxes_kept
        stats.boxes_noise_dropped += img_noise_dropped

        if img_boxes_kept == 0:
            stats.empty_files_written += 1

        dst.write_text("\n".join(out_lines))
        stats.files_written += 1

    return stats


def default_in_out_roots(run_root: Path) -> tuple[Path, Path]:
    cfg = read_json(run_root / "config.json")
    data_dir = Path(cfg["DATA_DIR"]).resolve()
    # Expect: .../spectrograms
    parent = data_dir.parent
    in_root = parent / "spectrograms_labels"
    out_root = parent / "spectrograms_labels_cleaned"
    return in_root, out_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge raw spectrogram labels into cleaned detection labels (drop Noise).")
    p.add_argument("--run_dir", type=str, default=None, help="CNN run dir providing config.json + labels.json")
    p.add_argument("--in_root", type=str, default=None, help="Input labels root (spectrograms_labels)")
    p.add_argument("--out_root", type=str, default=None, help="Output labels root (spectrograms_labels_cleaned)")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(os.getcwd())
    results_root = base / "outputs" / "cnn_results"
    if args.run_dir:
        run_root = Path(args.run_dir)
    else:
        # pick latest
        runs = [p for p in results_root.iterdir() if p.is_dir()]
        if not runs:
            raise FileNotFoundError(f"No runs in {results_root}")
        run_root = max(runs, key=lambda p: p.stat().st_mtime)
    print(f"[CLEAN] Using run_dir: {run_root}")

    mapping = build_merge_mapping(run_root)
    in_root, out_root = default_in_out_roots(run_root)
    if args.in_root:
        in_root = Path(args.in_root)
    if args.out_root:
        out_root = Path(args.out_root)

    print(f"[CLEAN] Input : {in_root}")
    print(f"[CLEAN] Output: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    valid_ids = set(mapping.old_idx_to_merged_idx.values())

    sites = sorted([p for p in in_root.iterdir() if p.is_dir()])
    if not sites:
        raise FileNotFoundError(f"No site folders under {in_root}")

    for site_in in sites:
        site_out = out_root / site_in.name
        if is_site_cleaned(site_out, valid_ids):
            print(f"[CLEAN][skip] {site_in.name} already cleaned")
            continue
        print(f"[CLEAN][site] {site_in.name}")
        stats = clean_site(site_in, site_out, mapping)
        payload = {
            "cleaned_by": "clean_spectrogram_labels.py",
            "run_id": run_root.name,
            "source_categories": mapping.old_idx_to_name,
            "merged_names": mapping.merged_names_in_order,
            "stats": asdict(stats),
        }
        write_json(site_out / SENTINEL_NAME, payload)
        print(f"[CLEAN][done] {site_in.name}: files={stats.files_total} kept boxes={stats.boxes_kept_after_merge} dropped_noise boxes={stats.boxes_noise_dropped} empty boxes={stats.empty_files_written}")


if __name__ == "__main__":
    main()

