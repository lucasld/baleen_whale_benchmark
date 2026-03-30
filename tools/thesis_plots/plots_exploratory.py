from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from thesis_plots.config import (
    APPENDIX_TCR_NMR_ANNOTATION_THRESHOLDS,
    DECISION_CHOSEN_RUNS,
    DECISION_GROUPS,
)
from thesis_plots.theme import APPENDIX_PANEL, MAIN_WIDE, style_axis


def build_exploratory_decision_summary(data: dict):
    df = data["exploratory_points"]
    colors = data["run_colors"]
    fig, ax = plt.subplots(figsize=MAIN_WIDE)
    group_names = list(DECISION_GROUPS.keys())
    x_positions = np.arange(len(group_names), dtype=float)
    for idx, group in enumerate(group_names):
        if idx % 2 == 0:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#F3F3F3", zorder=0)
        runs = [r for r in DECISION_GROUPS[group] if r in set(df["run_name"])]
        offsets = np.linspace(-0.14, 0.14, max(len(runs), 1))
        for off, run_name in zip(offsets, runs):
            row = df[df["run_name"] == run_name].iloc[0]
            chosen = DECISION_CHOSEN_RUNS[group] == run_name
            ax.scatter(
                idx + off,
                row["test_selected_F"],
                s=110 if chosen else 45,
                color=colors.get(run_name, "#666666"),
                edgecolor="#111111" if chosen else "white",
                linewidth=1.4 if chosen else 0.8,
                zorder=4 if chosen else 3,
            )
            ax.text(idx + off, row["test_selected_F"] + (0.010 if chosen else 0.006), run_name, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names)
    style_axis(ax, xlabel="Exploratory ablation group", ylabel="Selected-threshold F", ylim=(0.78, 0.91), grid_axis="y")
    return fig


def _build_curve_figure(data: dict, run_names: list[str], curve_kind: str):
    sweeps = data["exploratory_sweeps"]
    points = data["exploratory_points"]
    colors = data["run_colors"]
    fig, ax = plt.subplots(figsize=APPENDIX_PANEL)
    for run_name in run_names:
        sdf = sweeps[sweeps["run_name"] == run_name].sort_values("threshold")
        if sdf.empty:
            continue
        color = colors.get(run_name, "#666666")
        if curve_kind == "tcr_vs_nmr":
            ax.plot(sdf["NMR"], sdf["TCR"], color=color, lw=2, label=run_name)
            annotation_rows = sdf[sdf["threshold"].round(2).isin(APPENDIX_TCR_NMR_ANNOTATION_THRESHOLDS)]
            for _, row in annotation_rows.iterrows():
                ax.annotate(
                    f"{float(row['threshold']):.2f}",
                    (float(row["NMR"]), float(row["TCR"])),
                    textcoords="offset points",
                    xytext=(3, 2),
                    fontsize=6,
                    color=color,
                    alpha=0.85,
                )
            prow = points[points["run_name"] == run_name]
            if not prow.empty:
                ax.scatter(
                    prow["test_selected_NMR"],
                    prow["test_selected_TCR"],
                    color=color,
                    s=35,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
            style_axis(ax, xlabel="NMR", ylabel="TCR", xlim=(0, 1), ylim=(0, 1))
        else:
            ax.plot(sdf["threshold"], sdf["F"], color=color, lw=2, label=run_name)
            prow = points[points["run_name"] == run_name]
            if not prow.empty:
                ax.scatter(
                    prow["selected_threshold"],
                    prow["test_selected_F"],
                    color=color,
                    s=35,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
            style_axis(ax, xlabel="Confidence threshold", ylabel="F", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, ncol=2 if len(run_names) > 3 else 1, loc="best")
    return fig


def build_exploratory_bs_tcr_vs_nmr(data: dict):
    return _build_curve_figure(data, ["BS1", "BS2"], "tcr_vs_nmr")


def build_exploratory_r0_a_tcr_vs_nmr(data: dict):
    return _build_curve_figure(data, ["R0a", "R0b", "A1", "A2", "A3"], "tcr_vs_nmr")


def build_exploratory_b_tcr_vs_nmr(data: dict):
    return _build_curve_figure(data, ["B1", "B2", "B3"], "tcr_vs_nmr")


def build_exploratory_c_tcr_vs_nmr(data: dict):
    return _build_curve_figure(data, ["B1", "C1", "C2"], "tcr_vs_nmr")


def build_exploratory_d_tcr_vs_nmr(data: dict):
    return _build_curve_figure(data, ["C2", "D2"], "tcr_vs_nmr")


def build_exploratory_bs_f_vs_confidence(data: dict):
    return _build_curve_figure(data, ["BS1", "BS2"], "f_vs_confidence")


def build_exploratory_r0_a_f_vs_confidence(data: dict):
    return _build_curve_figure(data, ["R0a", "R0b", "A1", "A2", "A3"], "f_vs_confidence")


def build_exploratory_b_f_vs_confidence(data: dict):
    return _build_curve_figure(data, ["B1", "B2", "B3"], "f_vs_confidence")


def build_exploratory_c_f_vs_confidence(data: dict):
    return _build_curve_figure(data, ["B1", "C1", "C2"], "f_vs_confidence")


def build_exploratory_d_f_vs_confidence(data: dict):
    return _build_curve_figure(data, ["C2", "D2"], "f_vs_confidence")
