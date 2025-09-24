from typing import Dict, List
import json
import os
from pathlib import Path
import pandas as pd


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


