from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from thesis_plots.theme import APPENDIX_PANEL, style_axis


def build_multiclass_rate_by_fold(data: dict):
    df = data["multiclass_per_fold"]
    fig, ax = plt.subplots(figsize=APPENDIX_PANEL)
    x = np.arange(len(df))
    ax.plot(x, df["gt_multiclass_rate"], color=data["run_colors"].get(data["run_name"], "#666666"), marker="o", lw=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["fold_label"], rotation=45, ha="right")
    style_axis(ax, xlabel="Fold", ylabel="Multiclass snippet rate", ylim=(0, max(0.22, df["gt_multiclass_rate"].max() * 1.15)), grid_axis="y")
    return fig


def build_multiclass_rescued_breakdown_by_fold(data: dict):
    df = data["multiclass_per_fold"].copy()
    fig, ax = plt.subplots(figsize=APPENDIX_PANEL)
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    columns = [
        ("wrong_no_relevant_prediction_rate_wrong", "No relevant prediction", "#B8B8B8"),
        ("rescued_primary_rate_wrong", "Primary predicted but not Top-1", "#80B1D3"),
        ("rescued_other_gt_only_rate_wrong", "Other GT class predicted but not primary", data["run_colors"].get(data["run_name"], "#542788")),
    ]
    for col, label, color in columns:
        vals = df[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(df["fold_label"], rotation=45, ha="right")
    style_axis(ax, xlabel="Fold", ylabel="Share within Top-1 wrong samples", ylim=(0, 1), grid_axis="y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    return fig
