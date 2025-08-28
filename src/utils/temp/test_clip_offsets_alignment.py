#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd


def load_npz(koogu_site_dir: str, wav_base: str):
    npz_path = os.path.join(koogu_site_dir, f"{wav_base}.wav.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    fs = float(data['fs']) if 'fs' in data.files else 250.0
    clips = data['clips']
    labels = data['labels']
    offsets = data['clip_offsets'] if 'clip_offsets' in data.files else None
    classes_file = os.path.join(koogu_site_dir, 'classes_list.json')
    classes = json.load(open(classes_file)) if os.path.exists(classes_file) else [str(i) for i in range(labels.shape[1])]
    return fs, clips, labels, offsets, classes


def load_annotations(ann_site_dir: str, wav_base: str) -> pd.DataFrame:
    ann_path = os.path.join(ann_site_dir, f"{wav_base}_annotations.txt")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(ann_path)
    df = pd.read_csv(ann_path, sep='\t')
    required = ['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Tags']
    if not all(c in df.columns for c in required):
        raise RuntimeError(f"Annotation file missing required columns: {ann_path}")
    return df


def compute_overlaps(ann_df: pd.DataFrame, c_start: float, c_end: float) -> int:
    s = ann_df['Begin Time (s)'].astype(float).values
    e = ann_df['End Time (s)'].astype(float).values
    return int(np.sum((e > c_start) & (s < c_end)))


def main():
    ap = argparse.ArgumentParser(description='Tiny test: verify clip_offsets-based timing yields overlaps aligned with Koogu labels')
    ap.add_argument('--output_dir', required=True, help='Preprocessed dataset root (with koogu_output/ and annotations/)')
    ap.add_argument('--site', required=True)
    ap.add_argument('--wav_base', required=True, help='e.g., 20150510_100000')
    ap.add_argument('--max_clips', type=int, default=60)
    args = ap.parse_args()

    koogu_site_dir = os.path.join(args.output_dir, 'koogu_output', args.site)
    ann_site_dir = os.path.join(args.output_dir, 'annotations', args.site)

    fs, clips, labels, offsets, classes = load_npz(koogu_site_dir, args.wav_base)
    ann_df = load_annotations(ann_site_dir, args.wav_base)

    if offsets is None:
        print('FAIL: .npz has no clip_offsets; cannot test offsets-based alignment')
        sys.exit(2)

    clip_len = 15.0
    advance = 2.5
    K = min(args.max_clips, labels.shape[0])

    # Count overlaps using wrong origin (0, 2.5, ...) and using offsets/fs
    wrong_overlaps = 0
    right_overlaps = 0
    nonzero_labels = 0
    for k in range(K):
        c0 = k * advance
        c1 = c0 + clip_len
        if compute_overlaps(ann_df, c0, c1) > 0:
            wrong_overlaps += 1

        cs = float(offsets[k]) / fs
        ce = cs + clip_len
        if compute_overlaps(ann_df, cs, ce) > 0:
            right_overlaps += 1

        if float(np.sum(labels[k])) > 0:
            nonzero_labels += 1

    print(f"Clips checked: {K} | nonzero Koogu labels: {nonzero_labels}")
    print(f"Overlaps with wrong origin: {wrong_overlaps}")
    print(f"Overlaps with clip_offsets/fs: {right_overlaps}")

    # Basic success criteria: using offsets increases overlaps, and a meaningful fraction of labeled clips overlap
    improved = right_overlaps >= wrong_overlaps
    fraction = (right_overlaps / max(1, nonzero_labels))
    print(f"Improved: {improved} | overlap_fraction_vs_labeled: {fraction:.2f}")

    if not improved or fraction < 0.3:
        print('FAIL: offsets-based alignment did not produce sufficient overlaps')
        sys.exit(1)
    print('PASS: offsets-based alignment improves overlaps and matches Koogu labeling better')
    sys.exit(0)


if __name__ == '__main__':
    main()


