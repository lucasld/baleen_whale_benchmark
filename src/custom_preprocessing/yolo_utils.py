"""
YOLO utilities for baleen whale detection and annotation processing.

This module contains all YOLO-related functionality for:
- Loading and processing annotations
- Converting annotations to YOLO format
- Generating YOLO labels and preview images
- Managing class mappings
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def load_annotations_for_wav(filename_base: str, annotations_output_dir: str, site_name: str):
    """Load per-WAV annotations from file."""
    if not annotations_output_dir:
        return None
    ann_dir = os.path.join(annotations_output_dir, site_name)
    ann_path = os.path.join(ann_dir, f"{filename_base}_annotations.txt")
    if not os.path.exists(ann_path):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(ann_path, sep='\t')
        # Ensure required columns exist
        required = ['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Tags']
        if not all(col in df.columns for col in required):
            return None
        return df
    except Exception:
        return None


def create_class_to_idx_mapping(expected_classes):
    """Create mapping from class names to indices."""
    return {c: i for i, c in enumerate(expected_classes or [])}


def tags_to_class_ids(tag_str: str, class_to_idx: dict):
    """Convert tag string to list of class IDs."""
    if tag_str is None or (isinstance(tag_str, float) and np.isnan(tag_str)):
        return []
    tags = [t.strip() for t in str(tag_str).split(',') if t.strip()]
    ids = []
    for t in tags:
        if t in class_to_idx:
            ids.append(class_to_idx[t])
    return ids


def write_yolo_labels(label_path: str, boxes: list):
    """Write YOLO format labels to file."""
    if not boxes:
        # Write empty file to indicate no boxes (optional). Safer for tooling.
        open(label_path, 'w').close()
        return
    with open(label_path, 'w') as f:
        for cls_id, x_center, y_center, w_norm, h_norm in boxes:
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def clip_and_normalize_boxes(ann_df, clip_start_s: float, clip_end_s: float, 
                           img_w: int, img_h: int, audio_settings: dict, class_to_idx: dict):
    """Convert annotations to YOLO format boxes for a specific clip."""
    # img_w=30 (time), img_h=90 (freq) after resize; YOLO needs normalized [0,1]
    duration = clip_end_s - clip_start_s
    if duration <= 0:
        return []
    yolo_boxes = []
    for _, row in ann_df.iterrows():
        sel_start = float(row['Begin Time (s)'])
        sel_end = float(row['End Time (s)'])
        low_f = float(row['Low Freq (Hz)'])
        high_f = float(row['High Freq (Hz)'])
        # Overlap with clip in time
        inter_start = max(sel_start, clip_start_s)
        inter_end = min(sel_end, clip_end_s)
        if inter_end <= inter_start:
            continue
        # Time normalization
        x_center = ((inter_start + inter_end) / 2 - clip_start_s) / duration
        w_norm = (inter_end - inter_start) / duration
        # Frequency normalization (0..fs/2 ~ 0..125 Hz), image is flipped vertically
        f_max = audio_settings['desired_fs'] / 2.0  # 125 Hz
        low_n = max(0.0, min(1.0, low_f / f_max))
        high_n = max(0.0, min(1.0, high_f / f_max))
        if high_n <= low_n:
            continue
        y_center = 1.0 - ((low_n + high_n) / 2.0)
        h_norm = (high_n - low_n)
        # Clamp to [0,1]
        x_center = float(max(0.0, min(1.0, x_center)))
        y_center = float(max(0.0, min(1.0, y_center)))
        w_norm = float(max(0.0, min(1.0, w_norm)))
        h_norm = float(max(0.0, min(1.0, h_norm)))
        # One entry per class tag present on this selection
        for cls_id in tags_to_class_ids(row.get('Tags'), class_to_idx):
            yolo_boxes.append((cls_id, x_center, y_center, w_norm, h_norm))
    return yolo_boxes


def generate_yolo_labels_for_clip(spec_label: str, ann_df, clip_start_s: float, clip_end_s: float,
                                audio_settings: dict, class_to_idx: dict, yolo_site_dir: str):
    """Generate and save YOLO labels for a single clip."""
    boxes = clip_and_normalize_boxes(ann_df, clip_start_s, clip_end_s, 30, 90, audio_settings, class_to_idx)
    label_basename = os.path.splitext(spec_label)[0] + '.txt'
    label_path = os.path.join(yolo_site_dir, label_basename)
    write_yolo_labels(label_path, boxes)
    return boxes


def generate_preview_with_annotations(im: Image.Image, spec_label: str, boxes: list, preview_site_dir: str):
    """Generate preview image with annotation overlays."""
    try:
        draw_img = im.copy().convert('RGB')
        if boxes:
            draw = ImageDraw.Draw(draw_img)
            for (cls_id, x_c, y_c, w_n, h_n) in boxes:
                # Convert normalized YOLO to pixel rectangle
                x0 = int((x_c - w_n / 2) * 30)
                x1 = int((x_c + w_n / 2) * 30)
                y0 = int((y_c - h_n / 2) * 90)
                y1 = int((y_c + h_n / 2) * 90)
                # Clamp
                x0 = max(0, min(29, x0)); x1 = max(0, min(29, x1))
                y0 = max(0, min(89, y0)); y1 = max(0, min(89, y1))
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=1)
        preview_name = os.path.splitext(spec_label)[0] + '_annot.png'
        preview_path = os.path.join(preview_site_dir, preview_name)
        draw_img.save(preview_path)
    except Exception:
        pass
