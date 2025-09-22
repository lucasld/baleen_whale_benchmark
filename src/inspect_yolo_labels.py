#!/usr/bin/env python3
"""
Script to inspect the recreated YOLO labels: check counts, sizes, and sample contents.
Run from the baleen_whale_benchmark directory.
"""

import os
from pathlib import Path

def inspect_yolo_labels(yolo_base_path):
    labels_dir = Path(yolo_base_path) / 'labels'
    partitions = ['train', 'valid', 'test']

    for partition in partitions:
        part_dir = labels_dir / partition
        if not part_dir.exists():
            print(f"Partition {partition} directory not found.")
            continue

        txt_files = list(part_dir.glob('*.txt'))
        total_files = len(txt_files)
        non_empty = sum(1 for f in txt_files if f.stat().st_size > 0)
        empty = total_files - non_empty

        print(f"\n=== {partition.upper()} Partition ===")
        print(f"Total label files: {total_files}")
        print(f"Non-empty files: {non_empty}")
        print(f"Empty files: {empty}")

        # Show sample contents from non-empty files
        sample_files = [f for f in txt_files if f.stat().st_size > 0][:3]  # First 3 non-empty
        for sample in sample_files:
            print(f"\nSample from {sample.name}:")
            with open(sample, 'r') as file:
                lines = file.readlines()[:5]  # First 5 lines
                for line in lines:
                    print(f"  {line.strip()}")
                if len(lines) == 5 and len(file.readlines()) > 0:
                    print("  ... (truncated)")

if __name__ == "__main__":
    # Hardcoded path for this run
    yolo_path = "/share/klab/danthes/lliessduques/test/baleen_whale_benchmark/outputs/cnn_results/250920_021045/fold_BallenyIslands2015_noise_0.25/yolo_dataset"
    inspect_yolo_labels(yolo_path)
