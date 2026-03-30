from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


MAIN_STANDARD = (7.0, 4.8)
MAIN_WIDE = (8.4, 4.8)
APPENDIX_STANDARD = (7.2, 4.8)
APPENDIX_PANEL = (7.6, 5.2)


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.6,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def style_axis(ax, *, xlabel: str, ylabel: str, xlim=None, ylim=None, grid_axis: str = "both") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.4)


def save_figure(fig, base_path: Path, formats: Iterable[str]) -> list[str]:
    written = []
    for fmt in formats:
        fmt = fmt.lower()
        out = base_path.with_suffix(f".{fmt}")
        fig.savefig(out, bbox_inches="tight")
        written.append(str(out))
    plt.close(fig)
    return written

