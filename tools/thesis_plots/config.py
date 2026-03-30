from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = WORKSPACE_ROOT / "baleen_whale_benchmark" / "outputs" / "cnn_results" / "251008_160341"
DEFAULT_RUN_NAME = "F1"
DEFAULT_EXPLORATORY_FOLD = "fold_BallenyIslands2015_noise_0.25"
DEFAULT_STYLE_FILE = WORKSPACE_ROOT / "baleen_whale_benchmark" / "experiments" / "plot_style.json"

EXPLORATORY_GROUPS: Dict[str, List[str]] = {
    "BS": ["BS1", "BS2"],
    "R0/A": ["R0a", "R0b", "A1", "A2", "A3"],
    "B": ["B1", "B2", "B3"],
    "C": ["B1", "C1", "C2"],
    "D": ["C2", "D2"],
}

CHOSEN_RUNS: Dict[str, str] = {
    "BS": "BS1",
    "R0/A": "A2",
    "B": "B1",
    "C": "C2",
    "D": "C2",
}

DECISION_GROUPS: Dict[str, List[str]] = {
    "BS": ["BS1", "BS2"],
    "R0": ["R0a", "R0b"],
    "A": ["A1", "A2", "A3"],
    "B": ["B1", "B2", "B3"],
    "C": ["B1", "C1", "C2"],
    "D": ["C2", "D2"],
}

DECISION_CHOSEN_RUNS: Dict[str, str] = {
    "BS": "BS1",
    "R0": "R0b",
    "A": "A2",
    "B": "B1",
    "C": "C2",
    "D": "C2",
}

RUN_ORDER = [
    "CNN",
    "BS1",
    "BS2",
    "R0a",
    "R0b",
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
    "C1",
    "C2",
    "D2",
    "F1",
]

EXPLORATORY_HARMONIZED_THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

APPENDIX_TCR_NMR_ANNOTATION_THRESHOLDS = [0.01, 0.05, 0.20, 0.40, 0.80]

MAIN_TCR_NMR_ANNOTATION_THRESHOLDS = [0.01, 0.05, 0.20, 0.40]


@dataclass(frozen=True)
class FigurePaths:
    root: Path
    main: Path
    appendix: Path


def build_output_paths(output_dir: Path) -> FigurePaths:
    return FigurePaths(
        root=output_dir,
        main=output_dir / "main",
        appendix=output_dir / "appendix",
    )


def format_fold_label(fold_name: str) -> str:
    base = fold_name
    if base.startswith("fold_"):
        base = base[5:]
    base = re.sub(r"_noise_[0-9.]+$", "", base)
    base = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base)
    m = re.match(r"^(.*?)(\d{4})$", base)
    if m:
        base = f"{m.group(1).strip()} {m.group(2)}"
    return base.strip()
