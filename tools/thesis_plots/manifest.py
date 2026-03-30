from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from thesis_plots import plots_comparison, plots_exploratory, plots_multiclass, plots_thresholds


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    scope: str
    output_stem: str
    builder: Callable[[dict], object]
    source_artifacts: List[str]


def get_manifest(run_name: str) -> Dict[str, FigureSpec]:
    return {
        "exploratory_decision_summary": FigureSpec(
            "exploratory_decision_summary",
            "main",
            "exploratory_decision_summary",
            plots_exploratory.build_exploratory_decision_summary,
            [f"fold_BallenyIslands2015_noise_0.25/yolo/comparisons/latest/comparison_runs.csv"],
        ),
        "final_cnn_vs_yolo_selected_metrics": FigureSpec(
            "final_cnn_vs_yolo_selected_metrics",
            "main",
            "final_cnn_vs_yolo_selected_metrics",
            plots_comparison.build_final_cnn_vs_yolo_selected_metrics,
            [f"yolo/analysis/manual_compare/02_compare_{run_name}_aggregated/tables/summary_mean_std.csv"],
        ),
        "final_operating_tradeoff_tcr_vs_nmr": FigureSpec(
            "final_operating_tradeoff_tcr_vs_nmr",
            "main",
            "final_operating_tradeoff_tcr_vs_nmr",
            plots_thresholds.build_final_operating_tradeoff_tcr_vs_nmr,
            [
                f"yolo/analysis/manual_compare/07_compare_{run_name}_agg_with_cnn_curves/tables/sweep_summary_mean_std.csv",
                f"yolo/analysis/manual_compare/02_compare_{run_name}_aggregated/tables/summary_mean_std.csv",
            ],
        ),
        "exploratory_bs_tcr_vs_nmr": FigureSpec("exploratory_bs_tcr_vs_nmr", "appendix", "exploratory_bs_tcr_vs_nmr", plots_exploratory.build_exploratory_bs_tcr_vs_nmr, []),
        "exploratory_r0_a_tcr_vs_nmr": FigureSpec("exploratory_r0_a_tcr_vs_nmr", "appendix", "exploratory_r0_a_tcr_vs_nmr", plots_exploratory.build_exploratory_r0_a_tcr_vs_nmr, []),
        "exploratory_b_tcr_vs_nmr": FigureSpec("exploratory_b_tcr_vs_nmr", "appendix", "exploratory_b_tcr_vs_nmr", plots_exploratory.build_exploratory_b_tcr_vs_nmr, []),
        "exploratory_c_tcr_vs_nmr": FigureSpec("exploratory_c_tcr_vs_nmr", "appendix", "exploratory_c_tcr_vs_nmr", plots_exploratory.build_exploratory_c_tcr_vs_nmr, []),
        "exploratory_d_tcr_vs_nmr": FigureSpec("exploratory_d_tcr_vs_nmr", "appendix", "exploratory_d_tcr_vs_nmr", plots_exploratory.build_exploratory_d_tcr_vs_nmr, []),
        "exploratory_bs_f_vs_confidence": FigureSpec("exploratory_bs_f_vs_confidence", "appendix", "exploratory_bs_f_vs_confidence", plots_exploratory.build_exploratory_bs_f_vs_confidence, []),
        "exploratory_r0_a_f_vs_confidence": FigureSpec("exploratory_r0_a_f_vs_confidence", "appendix", "exploratory_r0_a_f_vs_confidence", plots_exploratory.build_exploratory_r0_a_f_vs_confidence, []),
        "exploratory_b_f_vs_confidence": FigureSpec("exploratory_b_f_vs_confidence", "appendix", "exploratory_b_f_vs_confidence", plots_exploratory.build_exploratory_b_f_vs_confidence, []),
        "exploratory_c_f_vs_confidence": FigureSpec("exploratory_c_f_vs_confidence", "appendix", "exploratory_c_f_vs_confidence", plots_exploratory.build_exploratory_c_f_vs_confidence, []),
        "exploratory_d_f_vs_confidence": FigureSpec("exploratory_d_f_vs_confidence", "appendix", "exploratory_d_f_vs_confidence", plots_exploratory.build_exploratory_d_f_vs_confidence, []),
        "final_selected_metrics_by_fold": FigureSpec("final_selected_metrics_by_fold", "appendix", "final_selected_metrics_by_fold", plots_comparison.build_final_selected_metrics_by_fold, []),
        "final_per_fold_tcr_vs_nmr": FigureSpec("final_per_fold_tcr_vs_nmr", "appendix", "final_per_fold_tcr_vs_nmr", plots_thresholds.build_final_per_fold_tcr_vs_nmr, []),
        "final_f_vs_confidence": FigureSpec("final_f_vs_confidence", "appendix", "final_f_vs_confidence", plots_thresholds.build_final_f_vs_confidence, []),
        "final_tcr_vs_confidence": FigureSpec("final_tcr_vs_confidence", "appendix", "final_tcr_vs_confidence", plots_thresholds.build_final_tcr_vs_confidence, []),
        "final_nmr_vs_confidence": FigureSpec("final_nmr_vs_confidence", "appendix", "final_nmr_vs_confidence", plots_thresholds.build_final_nmr_vs_confidence, []),
        "final_selected_threshold_by_fold": FigureSpec("final_selected_threshold_by_fold", "appendix", "final_selected_threshold_by_fold", plots_thresholds.build_final_selected_threshold_by_fold, []),
        "multiclass_rescued_breakdown_by_fold": FigureSpec("multiclass_rescued_breakdown_by_fold", "appendix", "multiclass_rescued_breakdown_by_fold", plots_multiclass.build_multiclass_rescued_breakdown_by_fold, []),
    }
