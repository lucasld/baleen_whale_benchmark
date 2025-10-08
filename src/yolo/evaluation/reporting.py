from typing import Dict, List
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def write_preds_debug(
    out_dir: Path,
    merged_df: pd.DataFrame,
    strategy_to_preds: Dict[str, List[int]],
    int_to_class: Dict[int, str],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = merged_df.copy()
    # human-readable helpers
    df['gt_primary_name'] = df['gt_primary'].map(int_to_class)
    for strat, vals in strategy_to_preds.items():
        df[f'pred_{strat}'] = vals
        df[f'pred_{strat}_name'] = df[f'pred_{strat}'].map(int_to_class)
    # Flatten lists for CSV readability
    df['all_pred_classes'] = df['all_pred_classes'].apply(lambda xs: json.dumps(xs))
    df['gt_multiset'] = df['gt_multiset'].apply(lambda xs: json.dumps(xs))
    df['gt_set'] = df['gt_set'].apply(lambda s: json.dumps(sorted(list(s))))
    df.to_csv(out_dir / 'preds_debug.csv', index=False)


def write_summary(out_dir: Path, summary: Dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)


def plot_tcr_vs_nmr_curves(metrics_df: pd.DataFrame, out_dir: Path):
    """
    Create TCR vs NMR plots from a confidence sweep metrics dataframe.

    - Combined plot overlaying all strategies
    - Individual plots per strategy
    """
    out_dir = Path(out_dir)
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    if metrics_df.empty:
        return

    # Ensure expected columns exist
    required = {'threshold', 'strategy', 'TCR', 'NMR'}
    if not required.issubset(set(metrics_df.columns)):
        return

    # Combined plot: overlay strategies
    fig, ax = plt.subplots(figsize=(6, 6))
    for strategy, df_s in metrics_df.groupby('strategy'):
        df_s = df_s.sort_values('threshold')
        ax.plot(df_s['NMR'], df_s['TCR'], marker='o', markersize=3, linewidth=1, label=strategy)
    ax.set_xlabel('NMR (Noise Misclassification Rate)')
    ax.set_ylabel('TCR (Mean Recall on Calls)')
    ax.set_title('TCR vs NMR across confidence thresholds')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / 'tcr_vs_nmr_all_strategies.png', dpi=200)
    plt.close(fig)

    # Individual plots per strategy
    for strategy, df_s in metrics_df.groupby('strategy'):
        df_s = df_s.sort_values('threshold')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(df_s['NMR'], df_s['TCR'], marker='o', markersize=3, linewidth=1, label=strategy)
        ax.set_xlabel('NMR (Noise Misclassification Rate)')
        ax.set_ylabel('TCR (Mean Recall on Calls)')
        ax.set_title(f'TCR vs NMR ({strategy})')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f'tcr_vs_nmr_{strategy}.png', dpi=200)
        plt.close(fig)


