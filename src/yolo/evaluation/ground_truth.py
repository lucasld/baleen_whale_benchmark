import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd


def _derive_join_map() -> Dict[str, str]:
    """
    Map raw names to merged names as used throughout the project.
    Raw→Merged mapping aligns with src/dataset.py and existing evaluation.
    """
    return {
        "20Plus": "20Hz20Plus",
        "20Hz": "20Hz20Plus",
        "A": "ABZ",
        "B": "ABZ",
        "Z": "ABZ",
        "D": "DDswp",
        "Dswp": "DDswp",
        "Noise": "Noise",
    }


def label_from_filename(img_path: str, class_to_int: Dict[str, int]) -> int:
    """
    Derive the primary (single) class from the spectrogram filename to match the CNN's single-label setting.
    Example: 1427_BallenyIslands2015_20Hz.png → 20Hz → 20Hz20Plus → int id
    """
    join_map = _derive_join_map()
    label_str = os.path.basename(img_path).split('_')[2].split('.')[0]
    merged = join_map.get(label_str, label_str)
    if merged not in class_to_int:
        raise ValueError(f"Label '{merged}' from filename not present in class_to_int")
    return class_to_int[merged]


def parse_yolo_label_file(txt_path: Path) -> List[int]:
    """
    Parse YOLO label file and return list of class ids for all boxes (can be empty).
    """
    if not txt_path.exists():
        return []
    classes: List[int] = []
    try:
        with txt_path.open('r') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                try:
                    cid = int(float(parts[0]))
                    classes.append(cid)
                except Exception:
                    continue
    except Exception:
        pass
    return classes


def build_ground_truth_dataframe(
    test_images_dir: Path,
    labels_dir: Path,
    class_to_int: Dict[str, int],
) -> pd.DataFrame:
    """
    Build a DataFrame with ground-truth information for each test image:
      - path: absolute image path
      - gt_primary: int label derived from filename (CNN-comparable)
      - gt_set: set of class ids present in YOLO labels (can be empty)
      - gt_multiset: list of class ids from YOLO labels (duplicates kept)
    """
    rows = []
    test_images_dir = Path(test_images_dir)
    labels_dir = Path(labels_dir)
    for img_path in sorted(test_images_dir.glob('*.png')):
        stem = img_path.stem
        yolo_txt = labels_dir / f"{stem}.txt"
        classes = parse_yolo_label_file(yolo_txt)
        gt_primary = label_from_filename(str(img_path), class_to_int)
        rows.append({
            'path': str(img_path),
            'gt_primary': gt_primary,
            'gt_set': set(classes),
            'gt_multiset': classes,
        })
    return pd.DataFrame(rows)


