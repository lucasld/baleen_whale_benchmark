from __future__ import annotations

import matplotlib.pyplot as plt

from thesis_plots.config import MAIN_TCR_NMR_ANNOTATION_THRESHOLDS, format_run_label
from thesis_plots.theme import APPENDIX_PANEL, MAIN_STANDARD, style_axis


def build_final_operating_tradeoff_tcr_vs_nmr(data: dict):
    curve = data["final_sweep_summary"].sort_values("threshold")
    summary = data["final_selected_summary"]
    colors = data["run_colors"]
    fig, ax = plt.subplots(figsize=MAIN_STANDARD)
    ax.plot(curve["NMR_mean"], curve["TCR_mean"], lw=2.2, color=colors.get(data["run_name"]), label=format_run_label(data["run_name"]))
    for _, row in curve[curve["threshold"].round(2).isin(MAIN_TCR_NMR_ANNOTATION_THRESHOLDS)].iterrows():
        threshold = float(row["threshold"])
        offset = {
            0.01: (5, -13),
            0.05: (5, -13),
            0.10: (5, -13),
            0.20: (5, -13),
            0.30: (5, -13),
            0.40: (5, -13),
        }.get(round(threshold, 2), (4, -11))
        ax.annotate(
            f"{threshold:.2f}",
            (float(row["NMR_mean"]), float(row["TCR_mean"])),
            textcoords="offset points",
            xytext=offset,
            fontsize=7,
            color=colors.get(data["run_name"]),
            alpha=0.9,
        )
    f1_row = summary[summary["run_name"] == data["run_name"]].iloc[0]
    cnn_row = summary[summary["run_name"] == "CNN"].iloc[0]
    ax.scatter(f1_row["test_selected_NMR_mean"], f1_row["test_selected_TCR_mean"], color=colors.get(data["run_name"]), s=75, edgecolor="white", linewidth=1.0, zorder=4)
    ax.scatter(cnn_row["test_selected_NMR_mean"], cnn_row["test_selected_TCR_mean"], color=colors.get("CNN"), s=65, edgecolor="white", linewidth=1.0, zorder=4, label=format_run_label("CNN"))
    ax.annotate(
        "CNN baseline",
        xy=(cnn_row["test_selected_NMR_mean"], cnn_row["test_selected_TCR_mean"]),
        xytext=(0.05, 0.70),
        textcoords="axes fraction",
        fontsize=8,
        color=colors.get("CNN"),
        ha="left",
        va="top",
    )
    style_axis(ax, xlabel="NMR", ylabel="TCR", xlim=(0, 1), ylim=(0, 1), grid_axis="both")
    ax.legend(frameon=False)
    return fig


def build_final_per_fold_tcr_vs_nmr(data: dict):
    df = data["final_sweep_rows"]
    fig, ax = plt.subplots(figsize=APPENDIX_PANEL)
    for fold_label, sdf in df.groupby("fold_label"):
        sdf = sdf.sort_values("threshold")
        ax.plot(sdf["NMR"], sdf["TCR"], lw=1.2, alpha=0.7)
    style_axis(ax, xlabel="NMR", ylabel="TCR", xlim=(0, 1), ylim=(0, 1))
    return fig


def _build_metric_vs_confidence(data: dict, metric: str):
    curve = data["final_sweep_summary"].sort_values("threshold")
    selected = data["final_selected_thresholds"]
    colors = data["run_colors"]
    fig, ax = plt.subplots(figsize=APPENDIX_PANEL)
    ax.plot(curve["threshold"], curve[f"{metric}_mean"], lw=2.2, color=colors.get(data["run_name"]))
    ax.axvline(selected["selected_threshold"].mean(), color="#777777", lw=1.1, linestyle="--")
    style_axis(ax, xlabel="Confidence threshold", ylabel=metric, xlim=(0, 1), ylim=(0, 1))
    return fig


def build_final_f_vs_confidence(data: dict):
    return _build_metric_vs_confidence(data, "F")


def build_final_tcr_vs_confidence(data: dict):
    return _build_metric_vs_confidence(data, "TCR")


def build_final_nmr_vs_confidence(data: dict):
    return _build_metric_vs_confidence(data, "NMR")


def build_final_selected_threshold_by_fold(data: dict):
    df = data["final_selected_thresholds"]
    fig, ax = plt.subplots(figsize=APPENDIX_PANEL)
    x = range(len(df))
    ax.scatter(list(x), df["selected_threshold"], s=40, color=data["run_colors"].get(data["run_name"]))
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["fold_label"], rotation=45, ha="right")
    style_axis(ax, xlabel="Fold", ylabel="Selected threshold", xlim=(-0.5, len(df) - 0.5), ylim=(0, 1), grid_axis="y")
    return fig
