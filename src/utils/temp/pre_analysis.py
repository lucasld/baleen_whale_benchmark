#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import platform
import subprocess
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib


def _ensure_matplotlib_backend():
    # Force non-interactive backend for headless environments
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return
    except Exception:
        pass
    matplotlib.use("Agg")


_ensure_matplotlib_backend()
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def discover_files(data_dir: str, categories: List[str], max_per_class: Optional[int] = None) -> pd.DataFrame:
    records = []
    for category in categories:
        class_dir = os.path.join(data_dir, category)
        if not os.path.isdir(class_dir):
            print(f"Warning: class directory missing: {class_dir}")
            continue
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
        files.sort()
        if max_per_class is not None and len(files) > max_per_class:
            rng = np.random.default_rng(42)
            files = list(rng.choice(files, size=max_per_class, replace=False))
        for fname in files:
            # Expected pattern: n_locationyear_class.png
            parts = os.path.splitext(fname)[0].split("_")
            location = parts[1] if len(parts) > 1 else "unknown"
            records.append({
                "path": os.path.join(class_dir, fname),
                "file": fname,
                "category": category,
                "site": location,
            })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(f"No PNG files discovered under {data_dir}")
    return df


def load_images(df: pd.DataFrame, target_h: int = 90, target_w: int = 30) -> np.ndarray:
    imgs = []
    bad = 0
    for p in df["path"].values:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            bad += 1
            imgs.append(np.zeros((target_h, target_w), dtype=np.uint8))
            continue
        # Images are saved as 90x30 in generate_spectrograms.py; ensure consistency
        if img.shape != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    if bad:
        print(f"Warning: {bad} images failed to load; replaced with zeros.")
    X = np.stack(imgs, axis=0).astype(np.float32) / 255.0
    # Model uses (width,height)= (30,90) but for analysis we do not need to transpose; keep (90,30)
    return X


def compute_embeddings(X: np.ndarray, pca_components: int = 50, tsne: bool = True, tsne_perplexity: int = 30):
    flat = X.reshape(len(X), -1)
    pca = PCA(n_components=min(pca_components, flat.shape[1], len(flat) - 1), random_state=42)
    Zp = pca.fit_transform(flat)
    result = {"pca": Zp, "pca_model": pca}
    if tsne:
        tsne_model = TSNE(n_components=2, perplexity=min(tsne_perplexity, max(5, len(flat) // 10)), init="pca", random_state=42)
        Z2 = tsne_model.fit_transform(Zp)
        result["tsne"] = Z2
    return result


def plot_scatter(Z: np.ndarray, labels: List[str], out_path: str, title: str):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, s=8, linewidth=0, palette="tab10", alpha=0.8)
    plt.title(title)
    plt.legend(loc="best", markerscale=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_baselines(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    flat = X.reshape(len(X), -1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_knn, metrics_lr = [], []
    for train_idx, test_idx in skf.split(flat, y):
        Xtr, Xte = flat[train_idx], flat[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        # k-NN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr, ytr)
        ypred = knn.predict(Xte)
        metrics_knn.append({
            "bal_acc": balanced_accuracy_score(yte, ypred),
            "f1_macro": f1_score(yte, ypred, average="macro"),
        })
        # Logistic Regression (linear baseline)
        lr = LogisticRegression(max_iter=2000, n_jobs=None, multi_class="auto")
        lr.fit(Xtr, ytr)
        ypred2 = lr.predict(Xte)
        metrics_lr.append({
            "bal_acc": balanced_accuracy_score(yte, ypred2),
            "f1_macro": f1_score(yte, ypred2, average="macro"),
        })
    def agg(ms):
        return {
            "bal_acc_mean": float(np.mean([m["bal_acc"] for m in ms])),
            "bal_acc_std": float(np.std([m["bal_acc"] for m in ms])),
            "f1_macro_mean": float(np.mean([m["f1_macro"] for m in ms])),
            "f1_macro_std": float(np.std([m["f1_macro"] for m in ms])),
        }
    return {"knn": agg(metrics_knn), "logreg": agg(metrics_lr)}


def save_mean_std_images(X: np.ndarray, y: np.ndarray, class_names: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for c, name in enumerate(class_names):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        mean_img = (X[idx].mean(axis=0) * 255.0).clip(0, 255).astype(np.uint8)
        std_img = (X[idx].std(axis=0) * 255.0 / max(1e-6, X[idx].std().max())).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"mean_{name}.png"), mean_img)
        cv2.imwrite(os.path.join(out_dir, f"std_{name}.png"), std_img)


def site_class_analysis(df: pd.DataFrame, out_dir: str):
    # Contingency table
    pivot = pd.pivot_table(df, index="site", columns="category", values="file", aggfunc="count", fill_value=0)
    pivot.to_csv(os.path.join(out_dir, "site_class_counts.csv"))
    plt.figure(figsize=(max(6, 0.6 * len(pivot.columns)), max(4, 0.4 * len(pivot.index))))
    sns.heatmap(pivot, annot=False, cmap="viridis")
    plt.title("Counts per site and class")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "site_class_heatmap.png"), dpi=200)
    plt.close()


def encode_labels(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    classes = sorted(series.unique())
    mapping = {c: i for i, c in enumerate(classes)}
    y = series.map(mapping).values
    return y, classes


@dataclass
class RunMetadata:
    repo_root: str
    data_dir: str
    config_path: str
    timestamp: str
    python_version: str
    platform: str
    conda_env: Optional[str]
    conda_prefix: Optional[str]
    numpy_version: Optional[str]
    pandas_version: Optional[str]
    sklearn_version: Optional[str]
    cv2_version: Optional[str]
    matplotlib_version: Optional[str]
    seaborn_version: Optional[str]
    nvidia_smi: Optional[str]
    git_commit: Optional[str]


def gather_run_metadata(repo_root: str, data_dir: str, config_path: str, timestamp: str) -> RunMetadata:
    def safe_ver(mod, attr="__version__"):
        try:
            return getattr(mod, attr)
        except Exception:
            return None
    try:
        nvidia = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    except Exception:
        nvidia = None
    try:
        git_commit = subprocess.getoutput("git rev-parse --short HEAD")
        if "fatal" in git_commit.lower():
            git_commit = None
    except Exception:
        git_commit = None
    return RunMetadata(
        repo_root=repo_root,
        data_dir=data_dir,
        config_path=config_path,
        timestamp=timestamp,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} | {platform.machine()}",
        conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
        conda_prefix=os.environ.get("CONDA_PREFIX"),
        numpy_version=safe_ver(np),
        pandas_version=safe_ver(pd),
        sklearn_version=safe_ver(__import__('sklearn')),
        cv2_version=safe_ver(cv2),
        matplotlib_version=safe_ver(matplotlib),
        seaborn_version=safe_ver(sns),
        nvidia_smi=nvidia,
        git_commit=git_commit,
    )


def setup_logger(log_path: str):
    logger = logging.getLogger("pre_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Pre-analysis: check class separability of spectrogram dataset")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config.json"), help="Path to config.json")
    parser.add_argument("--data_dir", default=None, help="Override data directory (class subfolders)")
    parser.add_argument("--max_per_class", type=int, default=1000, help="Max samples per class to load")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "analysis"), help="Output directory for figures and reports")
    parser.add_argument("--disable_tsne", action="store_true", help="Disable t-SNE computation (faster)")
    args = parser.parse_args()

    cfg = read_config(args.config)
    data_dir = args.data_dir or cfg.get("DATA_DIR", "datasets/preprocessed_dataset/spectrograms")
    # Resolve to absolute path relative to repo root if needed
    if not os.path.isabs(data_dir):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        data_dir = os.path.join(repo_root, data_dir)

    timestamp = time.strftime("%y%m%d_%H%M%S")
    out_base = os.path.join(os.path.abspath(args.out_dir), timestamp)
    os.makedirs(out_base, exist_ok=True)
    log_path = os.path.join(out_base, "run.log")
    logger = setup_logger(log_path)
    logger.info(f"Writing analysis outputs to: {out_base}")

    categories = cfg.get("CATEGORIES", [])
    if not categories:
        raise RuntimeError("CATEGORIES missing in config")

    # Record metadata
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    meta = gather_run_metadata(repo_root, data_dir, os.path.abspath(args.config), timestamp)
    with open(os.path.join(out_base, "metadata.json"), "w") as f:
        json.dump(asdict(meta), f, indent=2)
    with open(os.path.join(out_base, "cli_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(out_base, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Environment: Python {meta.python_version} | numpy {meta.numpy_version} | pandas {meta.pandas_version} | sklearn {meta.sklearn_version} | cv2 {meta.cv2_version}")
    if meta.nvidia_smi:
        logger.info(f"GPU(s): {meta.nvidia_smi}")
    if meta.git_commit:
        logger.info(f"Git commit: {meta.git_commit}")

    # Discover and load
    t0 = time.time()
    df = discover_files(data_dir, categories, max_per_class=args.max_per_class)
    logger.info(f"Discovered {len(df)} files across {len(categories)} categories under {data_dir}")
    # Ensure uniform sampling across classes for fair evaluation
    counts = df.groupby("category").size().to_dict()
    logger.info(f"Class counts (capped at {args.max_per_class}): {counts}")
    X = load_images(df)
    logger.info(f"Loaded images: shape={X.shape}, elapsed={(time.time()-t0):.2f}s")
    y, class_names = encode_labels(df["category"])  # preserve alphabetical order

    # Baselines
    tb = time.time()
    metrics = evaluate_baselines(X, y, n_splits=5)
    with open(os.path.join(out_base, "baselines.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Baselines: " + json.dumps(metrics, indent=2))

    # Silhouette on PCA space
    te = time.time()
    logger.info(f"Baseline CV elapsed: {(te-tb):.2f}s")
    embeds = compute_embeddings(X, pca_components=50, tsne=(not args.disable_tsne))
    Zp = embeds["pca"]
    sil = float(silhouette_score(Zp, y)) if len(np.unique(y)) > 1 else float("nan")
    with open(os.path.join(out_base, "silhouette.json"), "w") as f:
        json.dump({"silhouette_pca": sil}, f, indent=2)
    logger.info(f"Silhouette (PCA space): {sil:.4f}")

    # 2D plots
    if "tsne" in embeds:
        try:
            plot_scatter(embeds["tsne"], df["category"].tolist(), os.path.join(out_base, "tsne_classes.png"), "t-SNE by class")
            logger.info("Saved t-SNE visualization")
        except Exception as e:
            logger.warning(f"t-SNE plot failed: {e}")

    # Per-class means/stds
    save_mean_std_images(X, y, class_names, os.path.join(out_base, "class_means"))
    logger.info("Saved per-class mean and std images")

    # Confusion of a simple baseline for signal of separability
    flat = X.reshape(len(X), -1)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(flat, y)
    yhat = lr.predict(flat)
    cm = confusion_matrix(y, yhat, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])
    cm_df.to_csv(os.path.join(out_base, "confusion_logreg_train.csv"))
    logger.info("Saved in-sample logistic regression confusion matrix")

    # Site vs class distribution
    site_class_analysis(df, out_base)
    logger.info("Saved site-class heatmap and counts")

    # Save a small grid of random examples per class for sanity-check
    grid_dir = os.path.join(out_base, "grids")
    os.makedirs(grid_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for c, name in enumerate(class_names):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        take = idx if len(idx) <= 64 else rng.choice(idx, 64, replace=False)
        tiles = X[take]
        # Create 8x8 grid
        rows = []
        for r in range(8):
            row_imgs = []
            for k in range(8):
                i = r * 8 + k
                if i < len(tiles):
                    row_imgs.append((tiles[i] * 255).astype(np.uint8))
                else:
                    row_imgs.append(np.zeros_like(tiles[0], dtype=np.uint8))
            rows.append(np.concatenate(row_imgs, axis=1))
        grid = np.concatenate(rows, axis=0)
        cv2.imwrite(os.path.join(grid_dir, f"grid_{name}.png"), grid)
    logger.info("Saved example grids per class")

    # Save a summary json
    summary = {
        "data_dir": data_dir,
        "num_samples": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "classes": class_names,
        "outputs": out_base,
    }
    with open(os.path.join(out_base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Done. Inspect outputs for diagnostics.")


if __name__ == "__main__":
    main()


