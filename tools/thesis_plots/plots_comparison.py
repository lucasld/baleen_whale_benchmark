from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from thesis_plots.config import format_run_label
from thesis_plots.theme import APPENDIX_PANEL, MAIN_STANDARD, style_axis


METRICS = ["TCR", "NMR", "CMR", "F"]


def build_final_cnn_vs_yolo_selected_metrics(data: dict):
    df = data["final_selected_summary"].copy()
    colors = data["run_colors"]
    fig, ax = plt.subplots(figsize=MAIN_STANDARD)
    x = np.arange(len(METRICS), dtype=float)
    width = 0.18
    runs = ["CNN", data["run_name"]]
    for idx, run_name in enumerate(runs):
        row = df[df["run_name"] == run_name].iloc[0]
        xs = x + (idx - 0.5) * width
        vals = [row[f"test_selected_{m}_mean"] for m in METRICS]
        errs = [row[f"test_selected_{m}_std"] for m in METRICS]
        ax.errorbar(
            xs,
            vals,
            yerr=errs,
            fmt="o",
            capsize=4,
            lw=1.5,
            ms=6,
            color=colors.get(run_name, "#666666"),
            label=format_run_label(run_name),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    style_axis(ax, xlabel="Metric", ylabel="Score", ylim=(0, 1), grid_axis="y")
    ax.legend(frameon=False)
    return fig


def build_final_selected_metrics_by_fold(data: dict):
    df = data["final_by_fold"].copy()
    colors = data["run_colors"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
    axes = axes.flatten()
    labels = sorted(df["fold_label"].unique())
    x = np.arange(len(labels))
    for ax, metric in zip(axes, METRICS):
        for run_name in ["CNN", data["run_name"]]:
            sdf = df[df["run_name"] == run_name].set_index("fold_label").reindex(labels).reset_index()
            ax.plot(x, sdf[f"test_selected_{metric}"], marker="o", lw=1.6, ms=4, color=colors.get(run_name), label=format_run_label(run_name))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        style_axis(ax, xlabel="", ylabel=metric, ylim=(0, 1), grid_axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def build_final_selected_scatter_tcr_vs_nmr_by_fold(data: dict):
    df = data["final_by_fold"].copy()
    colors = data["run_colors"]
    fig, ax = plt.subplots(figsize=APPENDIX_PANEL)
    for run_name in ["CNN", data["run_name"]]:
        sdf = df[df["run_name"] == run_name]
        ax.scatter(
            sdf["test_selected_NMR"],
            sdf["test_selected_TCR"],
            s=45,
            color=colors.get(run_name),
            label=format_run_label(run_name),
            alpha=0.9,
        )
    style_axis(ax, xlabel="NMR", ylabel="TCR", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    return fig
