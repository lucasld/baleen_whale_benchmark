#!/usr/bin/env python3
"""IoU-aware multiclass analysis for the final YOLO detector.

This tool is intentionally inference-only. It reads existing fold weights and
selected thresholds, reruns YOLO prediction on each saved test split, and writes
new analysis artifacts without modifying the original run directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - cluster dependency
    YOLO = None


csv.field_size_limit(sys.maxsize)

CALL_CLASSES = ("20Hz20Plus", "ABZ", "DDswp")
NOISE_CLASS = "Noise"
DEFAULT_RUN_NAME = "F1"
DEFAULT_IOU_THRESHOLDS = (0.25, 0.5, 0.7)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruthBox:
    class_id: int
    xyxy: tuple[float, float, float, float]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_label_map(run_dir: Path) -> dict[str, int]:
    labels_path = run_dir / "labels.json"
    if not labels_path.exists():
        return {"20Hz20Plus": 0, "ABZ": 1, "DDswp": 2, "Noise": 3}
    return {str(k): int(v) for k, v in read_json(labels_path).items()}


def label_from_filename(path: Path, class_to_int: dict[str, int]) -> int:
    join_map = {
        "20Plus": "20Hz20Plus",
        "20Hz": "20Hz20Plus",
        "A": "ABZ",
        "B": "ABZ",
        "Z": "ABZ",
        "D": "DDswp",
        "Dswp": "DDswp",
        "Noise": "Noise",
    }
    raw = path.name.split("_")[2].split(".")[0]
    merged = join_map.get(raw, raw)
    if merged not in class_to_int:
        raise ValueError(f"Label '{merged}' from filename is not in label map: {path}")
    return class_to_int[merged]


def yolo_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)


def parse_gt_boxes(label_path: Path) -> list[GroundTruthBox]:
    if not label_path.exists():
        return []
    boxes: list[GroundTruthBox] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        x, y, w, h = (float(v) for v in parts[1:5])
        boxes.append(GroundTruthBox(class_id=class_id, xyxy=yolo_xywh_to_xyxy(x, y, w, h)))
    return boxes


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    detections: list[Detection],
    gt_boxes: list[GroundTruthBox],
    threshold: float,
) -> tuple[list[Detection], list[float]]:
    """Greedily match detections to same-class GT boxes at a fixed IoU threshold."""
    used_gt: set[int] = set()
    matched: list[Detection] = []
    matched_ious: list[float] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt or gt.class_id != det.class_id:
                continue
            iou = box_iou(det.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= threshold:
            used_gt.add(best_idx)
            matched.append(det)
            matched_ious.append(best_iou)
    return matched, matched_ious


def relation(gt_set: set[int], pred_set: set[int]) -> str:
    if gt_set == pred_set:
        return "exact"
    if not pred_set:
        return "no_prediction"
    overlap = gt_set & pred_set
    if not overlap:
        return "no_overlap"
    if pred_set < gt_set:
        return "missing_only"
    if pred_set > gt_set:
        return "extra_only"
    return "partial_overlap"


def discover_folds(run_dir: Path, run_name: str, fold_subset: set[str] | None) -> list[Path]:
    folds = sorted(p for p in run_dir.glob("fold_*_noise_*") if p.is_dir())
    if fold_subset:
        folds = [p for p in folds if p.name in fold_subset]
    return [p for p in folds if (p / "yolo" / "runs_det" / run_name).exists()]


def resolve_split_dir(fold_dir: Path, split: str, kind: str, image_root: Path | None) -> Path:
    relative = Path(fold_dir.name) / "yolo_dataset" / kind / split
    if image_root is not None:
        candidate = image_root / relative
        if candidate.exists():
            return candidate
    candidate = fold_dir / "yolo_dataset" / kind / split
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing {kind}/{split} directory for {fold_dir.name}: {candidate}")


def list_images(images_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.glob(f"*{ext}"))
    return sorted(images)


def collect_predictions(
    model,
    images_dir: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    batch: int,
    expected_count: int | None = None,
    progress_every: int = 250,
) -> dict[str, list[Detection]]:
    total_label = str(expected_count) if expected_count is not None else "unknown"
    print(
        f"[multiclass-iou] predicting {total_label} images from {images_dir} "
        f"(conf={conf:.3f}, imgsz={imgsz}, batch={batch}, device={device})",
        flush=True,
    )
    results = model.predict(
        source=str(images_dir),
        conf=float(conf),
        imgsz=int(imgsz),
        device=device,
        batch=int(batch),
        save=False,
        verbose=False,
        workers=4,
        stream=True,
    )
    out: dict[str, list[Detection]] = {}
    for idx, result in enumerate(results, start=1):
        path = str(Path(result.path).resolve())
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None and boxes.cls is not None and len(boxes.cls) > 0:
            classes = boxes.cls.cpu().numpy().tolist()
            confidences = boxes.conf.cpu().numpy().tolist()
            coords = boxes.xyxyn.cpu().numpy().tolist()
            for class_id, confidence, xyxy in zip(classes, confidences, coords):
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        confidence=float(confidence),
                        xyxy=tuple(float(v) for v in xyxy),
                    )
                )
        out[path] = detections
        if progress_every > 0 and (idx % progress_every == 0 or (expected_count is not None and idx == expected_count)):
            print(f"[multiclass-iou] predicted {idx}/{total_label} images", flush=True)
    print(f"[multiclass-iou] prediction pass complete: {len(out)} images", flush=True)
    return out


def encode_boxes(boxes: Iterable[GroundTruthBox | Detection]) -> str:
    rows = []
    for box in boxes:
        rows.append(
            {
                "class_id": int(box.class_id),
                "xyxy": [round(float(v), 8) for v in box.xyxy],
                **({"confidence": round(float(box.confidence), 8)} if isinstance(box, Detection) else {}),
            }
        )
    return json.dumps(rows)


def one_hot_from_set(class_ids: set[int], call_ids: list[int]) -> dict[int, int]:
    return {class_id: int(class_id in class_ids) for class_id in call_ids}


def set_aware_metrics(
    frame: pd.DataFrame,
    pred_set_col: str,
    call_ids: list[int],
    noise_id: int,
) -> dict[str, float]:
    tcrs = []
    cmrs = []
    for class_id in call_ids:
        contains = frame["gt_set"].apply(lambda xs: class_id in xs)
        total = int(contains.sum())
        if total == 0:
            tcrs.append(0.0)
            cmrs.append(0.0)
            continue
        correct = frame.loc[contains, pred_set_col].apply(lambda xs: class_id in xs).sum()
        tcrs.append(float(correct) / total)
        extras = frame.loc[contains].apply(
            lambda row: len((row[pred_set_col] - row["gt_set"]) & set(call_ids)) > 0,
            axis=1,
        ).sum()
        cmrs.append(float(extras) / total)

    noise_only = frame["gt_set"].apply(len).eq(0)
    noise_total = int(noise_only.sum())
    if noise_total:
        nmr = frame.loc[noise_only, pred_set_col].apply(lambda xs: len(xs & set(call_ids)) > 0).sum() / noise_total
    else:
        nmr = 0.0

    tcr = float(np.mean(tcrs)) if tcrs else 0.0
    cmr = float(np.mean(cmrs)) if cmrs else 0.0
    f_score = float(np.mean([tcr, 1.0 - nmr, 1.0 - nmr, 1.0 - cmr]))

    exact = frame.apply(lambda row: row[pred_set_col] == row["gt_set"], axis=1).mean()
    subset_recall = frame.apply(
        lambda row: (
            len(row[pred_set_col] & row["gt_set"]) / len(row["gt_set"])
            if len(row["gt_set"]) > 0
            else float(len(row[pred_set_col]) == 0)
        ),
        axis=1,
    ).mean()
    return {
        "set_TCR": tcr,
        "set_NMR": float(nmr),
        "set_CMR_extra": cmr,
        "set_F": f_score,
        "set_exact_match_rate": float(exact),
        "mean_per_snippet_set_recall": float(subset_recall),
    }


def top1_paper_metrics(frame: pd.DataFrame, class_to_int: dict[str, int]) -> dict[str, float]:
    noise_id = class_to_int[NOISE_CLASS]
    call_ids = [class_to_int[name] for name in CALL_CLASSES]
    tcrs = []
    cmrs = []
    for class_id in call_ids:
        class_rows = frame[frame["gt_primary"].eq(class_id)]
        total = len(class_rows)
        if total == 0:
            tcrs.append(0.0)
            cmrs.append(0.0)
            continue
        tcrs.append(float(class_rows["yolo_top1_pred"].eq(class_id).sum()) / total)
        cmrs.append(
            float(class_rows["yolo_top1_pred"].apply(lambda pred: pred in call_ids and pred != class_id).sum())
            / total
        )
    noise_rows = frame[frame["gt_primary"].eq(noise_id)]
    if len(noise_rows):
        nmr = float(noise_rows["yolo_top1_pred"].apply(lambda pred: pred in call_ids).sum()) / len(noise_rows)
    else:
        nmr = 0.0
    tcr = float(np.mean(tcrs)) if tcrs else 0.0
    cmr = float(np.mean(cmrs)) if cmrs else 0.0
    acc = float(frame["yolo_top1_pred"].eq(frame["gt_primary"]).mean()) if len(frame) else 0.0
    return {
        "TCR": tcr,
        "NMR": nmr,
        "CMR": cmr,
        "F": float(np.mean([tcr, 1.0 - nmr, 1.0 - nmr, 1.0 - cmr])),
        "ACC": acc,
    }


def load_cnn_prediction_sets(run_dir: Path, fold_dir: Path, class_to_int: dict[str, int]) -> dict[str, set[int]]:
    eval_dir = run_dir / "evaluation"
    fold_key = fold_dir.name
    matches = sorted(eval_dir.glob(f"predictions_{fold_key}_noise*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CNN prediction CSV found for {fold_key} under {eval_dir}")
    df = pd.read_csv(matches[0])
    label_cols = [str(i) for i in range(len(class_to_int))]
    noise_id = class_to_int[NOISE_CLASS]
    pred_sets: dict[str, set[int]] = {}
    for _, row in df.iterrows():
        path = Path(str(row["path"]))
        vals = [float(row[col]) for col in label_cols]
        pred = int(np.argmax(vals))
        pred_sets[path.name] = set() if pred == noise_id else {pred}
    return pred_sets


def build_fold_frame(
    fold_dir: Path,
    run_name: str,
    class_to_int: dict[str, int],
    image_root: Path | None,
    imgsz: int,
    device: str | None,
    batch: int,
    progress_every: int,
    iou_thresholds: list[float],
    run_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed; run this tool in the YOLO compute environment.")

    yolo_dir = fold_dir / "yolo" / "runs_det" / run_name
    weights = yolo_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    selection = read_json(yolo_dir / "selected_threshold_summary.json")
    threshold = float(selection["selected_threshold"])
    images_dir = resolve_split_dir(fold_dir, "test", "images", image_root)
    labels_dir = resolve_split_dir(fold_dir, "test", "labels", image_root)
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No test images found under {images_dir}")

    model = YOLO(str(weights))
    predictions = collect_predictions(
        model,
        images_dir,
        threshold,
        imgsz,
        device,
        batch,
        expected_count=len(images),
        progress_every=progress_every,
    )
    cnn_sets = load_cnn_prediction_sets(run_dir, fold_dir, class_to_int)

    noise_id = class_to_int[NOISE_CLASS]
    rows = []
    print(f"[multiclass-iou] matching boxes for {fold_dir.name}", flush=True)
    for idx, image_path in enumerate(images, start=1):
        abs_path = str(image_path.resolve())
        gt_boxes = parse_gt_boxes(labels_dir / f"{image_path.stem}.txt")
        detections = predictions.get(abs_path, [])
        gt_set = {box.class_id for box in gt_boxes}
        pred_set = {det.class_id for det in detections}
        gt_primary = label_from_filename(image_path, class_to_int)
        top1_pred = detections[0].class_id if detections else noise_id
        row = {
            "fold": fold_dir.name,
            "path": abs_path,
            "filename": image_path.name,
            "selected_threshold": threshold,
            "gt_primary": gt_primary,
            "gt_set": gt_set,
            "gt_box_count": len(gt_boxes),
            "gt_class_count": len(gt_set),
            "yolo_pred_set_class_only": pred_set,
            "yolo_pred_box_count": len(detections),
            "yolo_pred_class_count": len(pred_set),
            "yolo_top1_pred": top1_pred,
            "cnn_pred_set": cnn_sets.get(image_path.name, set()),
            "gt_boxes_json": encode_boxes(gt_boxes),
            "yolo_pred_boxes_json": encode_boxes(detections),
        }
        for iou_thr in iou_thresholds:
            matched, matched_ious = match_detections(detections, gt_boxes, iou_thr)
            tag = f"iou_{iou_thr:g}".replace(".", "p")
            row[f"yolo_pred_set_{tag}"] = {det.class_id for det in matched}
            row[f"yolo_matched_box_count_{tag}"] = len(matched)
            row[f"yolo_matched_ious_{tag}"] = matched_ious
            row[f"yolo_set_relation_{tag}"] = relation(gt_set, row[f"yolo_pred_set_{tag}"])
        rows.append(row)
        if progress_every > 0 and (idx % progress_every == 0 or idx == len(images)):
            print(f"[multiclass-iou] matched {idx}/{len(images)} images", flush=True)

    frame = pd.DataFrame(rows)
    check = {
        "fold": fold_dir.name,
        "selected_threshold": threshold,
        "n_images": len(images),
        "old_test_metrics_at_selected_threshold": selection.get("test_metrics_at_selected_threshold", {}),
    }
    return frame, check


def summarize_fold(
    frame: pd.DataFrame,
    call_ids: list[int],
    noise_id: int,
    iou_thresholds: list[float],
    class_to_int: dict[str, int],
    old_metrics: dict,
) -> dict:
    recomputed_top1 = top1_paper_metrics(frame, class_to_int)
    summary = {
        "fold": str(frame["fold"].iloc[0]),
        "selected_threshold": float(frame["selected_threshold"].iloc[0]),
        "n_samples": int(len(frame)),
        "gt_multiclass": int(frame["gt_class_count"].gt(1).sum()),
        "gt_multiclass_rate": float(frame["gt_class_count"].gt(1).mean()),
        "class_only_exact_multiclass": int(
            frame[frame["gt_class_count"].gt(1)].apply(
                lambda row: row["yolo_pred_set_class_only"] == row["gt_set"], axis=1
            ).sum()
        ),
    }
    for key, value in recomputed_top1.items():
        summary[f"recomputed_top1_{key}"] = float(value)
        if key in old_metrics:
            summary[f"recomputed_top1_delta_{key}"] = float(value) - float(old_metrics[key])
    summary.update({f"cnn_{k}": v for k, v in set_aware_metrics(frame, "cnn_pred_set", call_ids, noise_id).items()})
    summary.update(
        {
            f"yolo_class_only_{k}": v
            for k, v in set_aware_metrics(frame, "yolo_pred_set_class_only", call_ids, noise_id).items()
        }
    )
    multiclass = frame[frame["gt_class_count"].gt(1)]
    for iou_thr in iou_thresholds:
        tag = f"iou_{iou_thr:g}".replace(".", "p")
        pred_col = f"yolo_pred_set_{tag}"
        summary.update({f"yolo_{tag}_{k}": v for k, v in set_aware_metrics(frame, pred_col, call_ids, noise_id).items()})
        if len(multiclass):
            exact = multiclass.apply(lambda row: row[pred_col] == row["gt_set"], axis=1).sum()
            any_overlap = multiclass.apply(lambda row: bool(row[pred_col] & row["gt_set"]), axis=1).sum()
        else:
            exact = 0
            any_overlap = 0
        summary[f"yolo_{tag}_exact_multiclass"] = int(exact)
        summary[f"yolo_{tag}_exact_multiclass_rate"] = float(exact / len(multiclass)) if len(multiclass) else 0.0
        summary[f"yolo_{tag}_any_overlap_multiclass"] = int(any_overlap)
        summary[f"yolo_{tag}_any_overlap_multiclass_rate"] = float(any_overlap / len(multiclass)) if len(multiclass) else 0.0
    return summary


def serialize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if col.endswith("_set") or "_pred_set" in col:
            out[col] = out[col].apply(lambda xs: json.dumps(sorted(int(x) for x in xs)))
        elif col.startswith("yolo_matched_ious_"):
            out[col] = out[col].apply(lambda xs: json.dumps([round(float(x), 6) for x in xs]))
    return out


def plot_set_metrics(summary_df: pd.DataFrame, out_path: Path, iou_thresholds: list[float]) -> None:
    labels = ["CNN", "YOLO\nclass only"]
    cols = ["cnn_set_F", "yolo_class_only_set_F"]
    colors = ["#4D4D4D", "#6A51A3"]
    for iou_thr in iou_thresholds:
        tag = f"iou_{iou_thr:g}".replace(".", "p")
        labels.append(f"YOLO\nIoU >= {iou_thr:g}")
        cols.append(f"yolo_{tag}_set_F")
        colors.append("#238B45" if iou_thr >= 0.5 else "#41AB5D")
    means = [summary_df[col].mean() for col in cols]
    stds = [summary_df[col].std(ddof=1) for col in cols]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Set-aware F")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_set_metric_components(summary_df: pd.DataFrame, out_path: Path, iou_thresholds: list[float]) -> None:
    series = [
        ("CNN", "cnn", "#4D4D4D"),
        ("YOLO class only", "yolo_class_only", "#6A51A3"),
    ]
    for iou_thr in iou_thresholds:
        tag = f"iou_{iou_thr:g}".replace(".", "p")
        series.append((f"YOLO IoU >= {iou_thr:g}", f"yolo_{tag}", "#238B45" if iou_thr >= 0.5 else "#41AB5D"))

    metrics = [
        ("set_TCR", "TCR", True),
        ("set_NMR", "NMR", False),
        ("set_CMR_extra", "CMR", False),
        ("set_F", "F", True),
        ("set_exact_match_rate", "Exact set", True),
    ]
    x = np.arange(len(metrics))
    width = min(0.16, 0.75 / max(1, len(series)))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for offset, (label, prefix, color) in zip(offsets, series):
        vals = []
        for col_suffix, _, _ in metrics:
            col = f"{prefix}_{col_suffix}"
            vals.append(float(summary_df[col].mean()))
        ax.bar(x + offset, vals, width=width, label=label, color=color, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in metrics])
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def aggregate_summary(summary_df: pd.DataFrame, iou_thresholds: list[float]) -> dict:
    metric_cols = [col for col in summary_df.columns if col.endswith(("set_TCR", "set_NMR", "set_CMR_extra", "set_F"))]
    out: dict[str, object] = {
        "n_folds": int(len(summary_df)),
        "n_samples": int(summary_df["n_samples"].sum()),
        "gt_multiclass": int(summary_df["gt_multiclass"].sum()),
        "gt_multiclass_rate": float(summary_df["gt_multiclass"].sum() / summary_df["n_samples"].sum()),
        "metrics_mean": {col: float(summary_df[col].mean()) for col in metric_cols},
        "metrics_std": {col: float(summary_df[col].std(ddof=1)) for col in metric_cols},
    }
    for iou_thr in iou_thresholds:
        tag = f"iou_{iou_thr:g}".replace(".", "p")
        exact = int(summary_df[f"yolo_{tag}_exact_multiclass"].sum())
        out[f"yolo_{tag}_exact_multiclass"] = exact
        out[f"yolo_{tag}_exact_multiclass_rate"] = float(exact / out["gt_multiclass"]) if out["gt_multiclass"] else 0.0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IoU-aware YOLO multiclass analysis without overwriting final runs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/cnn_results/<run_id>.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Detector run name under fold_*/yolo/runs_det/.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for new analysis artifacts.")
    parser.add_argument("--folds", default="", help="Optional comma-separated fold directory names.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root containing fold_*/yolo_dataset/images/test.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size used by the final YOLO runs.")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--batch", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--iou-thresholds", default="0.25,0.5,0.7", help="Comma-separated IoU thresholds.")
    parser.add_argument("--progress-every", type=int, default=250, help="Print progress every N images; 0 disables.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute folds even if fold_tables output already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or (run_dir / "yolo" / "multiclass_iou_analysis" / args.run_name / "latest")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_subset = {x.strip() for x in args.folds.split(",") if x.strip()} or None
    iou_thresholds = [float(x.strip()) for x in args.iou_thresholds.split(",") if x.strip()]
    class_to_int = load_label_map(run_dir)
    noise_id = class_to_int[NOISE_CLASS]
    call_ids = [class_to_int[name] for name in CALL_CLASSES]
    folds = discover_folds(run_dir, args.run_name, fold_subset)
    if not folds:
        raise FileNotFoundError(f"No folds found for run '{args.run_name}' under {run_dir}")

    all_frames = []
    summaries = []
    checks = []
    fold_tables = output_dir / "fold_tables"
    fold_tables.mkdir(exist_ok=True)
    for fold_dir in folds:
        print(f"[multiclass-iou] {fold_dir.name}")
        fold_csv = fold_tables / f"{fold_dir.name}_iou_predictions.csv"
        if fold_csv.exists() and not args.no_resume:
            print(f"[multiclass-iou] reusing existing fold table: {fold_csv}", flush=True)
            cached = pd.read_csv(fold_csv)

            def parse_set(value: str) -> set[int]:
                return {int(x) for x in json.loads(value)}

            for col in cached.columns:
                if col.endswith("_set") or "_pred_set" in col:
                    cached[col] = cached[col].apply(parse_set)
                elif col.startswith("yolo_matched_ious_"):
                    cached[col] = cached[col].apply(json.loads)
            frame = cached
            selection = read_json(fold_dir / "yolo" / "runs_det" / args.run_name / "selected_threshold_summary.json")
            check = {
                "fold": fold_dir.name,
                "selected_threshold": float(frame["selected_threshold"].iloc[0]),
                "n_images": int(len(frame)),
                "old_test_metrics_at_selected_threshold": selection.get("test_metrics_at_selected_threshold", {}),
                "resumed_from": str(fold_csv),
            }
        else:
            frame, check = build_fold_frame(
                fold_dir=fold_dir,
                run_name=args.run_name,
                class_to_int=class_to_int,
                image_root=args.image_root.resolve() if args.image_root else None,
                imgsz=args.imgsz,
                device=args.device,
                batch=args.batch,
                progress_every=args.progress_every,
                iou_thresholds=iou_thresholds,
                run_dir=run_dir,
            )
            serialize_frame(frame).to_csv(fold_csv, index=False)
        all_frames.append(frame)
        summaries.append(
            summarize_fold(
                frame,
                call_ids,
                noise_id,
                iou_thresholds,
                class_to_int,
                check["old_test_metrics_at_selected_threshold"],
            )
        )
        checks.append(check)

    combined = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("fold").reset_index(drop=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    serialize_frame(combined).to_csv(tables_dir / "iou_predictions_all_folds.csv", index=False)
    summary_df.to_csv(tables_dir / "set_aware_metrics_by_fold.csv", index=False)
    write_json(output_dir / "run_checks.json", {"folds": checks})
    write_json(output_dir / "aggregate_summary.json", aggregate_summary(summary_df, iou_thresholds))
    plot_set_metrics(summary_df, plots_dir / "set_aware_f_comparison.png", iou_thresholds)
    plot_set_metric_components(summary_df, plots_dir / "set_aware_metric_components.png", iou_thresholds)
    print(f"[multiclass-iou] wrote {output_dir}")


if __name__ == "__main__":
    main()
