#!/usr/bin/env python3
"""
Script to analyze YOLO dataset for class counts and imbalance.
Counts samples per class by parsing label files.
Noise is considered as images with no bounding boxes (empty .txt files).
"""

import os
import glob
from collections import defaultdict

def analyze_yolo_dataset(dataset_path):
    splits = ['train', 'val', 'test']
    classes = ['20Hz20Plus', 'ABZ', 'DDswp', 'Noise']  # 0,1,2,3 but Noise is implicit

    results = {}

    for split in splits:
        images_dir = os.path.join(dataset_path, 'images', split)
        labels_dir = os.path.join(dataset_path, 'labels', split)

        if not os.path.exists(labels_dir):
            print(f"Labels dir {labels_dir} does not exist. Skipping {split}.")
            continue

        class_counts = defaultdict(int)
        total_samples = 0
        noise_samples = 0

        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        total_samples = len(label_files)

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if not lines:  # Empty file = noise
                    noise_samples += 1
                    class_counts['Noise'] += 1
                else:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id == 0:
                                class_counts['20Hz20Plus'] += 1
                            elif class_id == 1:
                                class_counts['ABZ'] += 1
                            elif class_id == 2:
                                class_counts['DDswp'] += 1
                            # Note: If multiple classes per image, counts per bounding box

        # For per-image counts, we need to adjust
        # Actually, to count unique images per class, but since one image can have multiple classes, it's tricky.
        # For imbalance, count bounding boxes or images with at least one class.

        # Let's count images per class (presence)
        image_class_counts = defaultdict(int)
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                classes_in_image = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)
                if not classes_in_image:
                    image_class_counts['Noise'] += 1
                else:
                    for cid in classes_in_image:
                        if cid == 0:
                            image_class_counts['20Hz20Plus'] += 1
                        elif cid == 1:
                            image_class_counts['ABZ'] += 1
                        elif cid == 2:
                            image_class_counts['DDswp'] += 1

        results[split] = {
            'total_images': total_samples,
            'noise_images': image_class_counts.get('Noise', 0),
            'class_images': {k: v for k, v in image_class_counts.items() if k != 'Noise'},
            'bounding_boxes': class_counts
        }

    return results

if __name__ == "__main__":
    dataset_path = "/share/klab/danthes/lliessduques/test/baleen_whale_benchmark/outputs/cnn_results/250922_003148/fold_BallenyIslands2015_noise_0.25/yolo_dataset"
    results = analyze_yolo_dataset(dataset_path)

    for split, data in results.items():
        print(f"\n=== {split.upper()} SPLIT ===")
        print(f"Total images: {data['total_images']}")
        print(f"Noise images (no calls): {data['noise_images']}")
        print("Images with calls per class:")
        for cls, count in data['class_images'].items():
            print(f"  {cls}: {count}")
        print("Bounding boxes per class:")
        for cls, count in data['bounding_boxes'].items():
            if cls != 'Noise':
                print(f"  {cls}: {count}")

    # Check imbalance
    print("\n=== IMBALANCE ANALYSIS ===")
    for split, data in results.items():
        total_calls = sum(data['class_images'].values())
        noise = data['noise_images']
        if total_calls + noise > 0:
            noise_ratio = noise / (total_calls + noise)
            print(f"{split}: Noise ratio: {noise_ratio:.2%} ({noise}/{total_calls + noise})")
            max_class = max(data['class_images'].values()) if data['class_images'] else 0
            min_class = min(data['class_images'].values()) if data['class_images'] else 0
            if max_class > 0:
                imbalance_ratio = min_class / max_class
                print(f"{split}: Class imbalance ratio (min/max): {imbalance_ratio:.2f}")
