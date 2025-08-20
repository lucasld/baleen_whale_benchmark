#!/usr/bin/env python3
"""
Simplified script to create scatter plots of model evaluation metrics.
Only generates PNG scatter plots showing individual fold results with means and std.

Usage:
python analyze_model_metrics.py --run_ids 250817_224430 250817_110328 --output_dir ./analysis_output
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import glob

def parse_metrics_from_filename(filename: str) -> Dict[str, float]:
    """
    Parse metrics from evaluation filename.
    Expected format: predictions_fold[location]_confusion_NMR=X.XX_CMR=X.XX_TCR=X.XX_F=X.XX.csv
    """
    patterns = {
        'NMR': r'NMR=(\d+\.?\d*)',
        'CMR': r'CMR=(\d+\.?\d*)', 
        'TCR': r'TCR=(\d+\.?\d*)',
        'F': r'F=(\d+\.?\d*)'
    }
    
    metrics = {}
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            value_str = match.group(1).rstrip('.')
            try:
                metrics[metric_name] = float(value_str)
            except ValueError as e:
                print(f"Warning: Could not convert {metric_name} value '{value_str}' to float in filename {filename}: {e}")
        else:
            print(f"Warning: Could not find {metric_name} in filename {filename}")
            
    return metrics

def extract_fold_name(filename: str) -> str:
    """Extract the fold/location name from the filename."""
    match = re.search(r'fold([^_]+)', filename)
    if match:
        return match.group(1)
    return "unknown"

def collect_metrics_from_run(run_id: str, base_dir: str = "outputs/cnn_results") -> pd.DataFrame:
    """Collect all metrics from a single model run."""
    evaluation_dir = os.path.join(base_dir, run_id, "evaluation")
    
    if not os.path.exists(evaluation_dir):
        print(f"Warning: Evaluation directory not found: {evaluation_dir}")
        return pd.DataFrame()
    
    pattern = os.path.join(evaluation_dir, "*confusion*.csv")
    files = glob.glob(pattern)
    
    data = []
    for file_path in files:
        filename = os.path.basename(file_path)
        metrics = parse_metrics_from_filename(filename)
        
        if len(metrics) == 4:  # All metrics found
            fold_name = extract_fold_name(filename)
            row = {
                'run_id': run_id,
                'fold': fold_name,
                **metrics
            }
            data.append(row)
        else:
            print(f"Warning: Incomplete metrics in file {filename}")
    
    return pd.DataFrame(data)

def collect_all_metrics(run_ids: List[str], base_dir: str = "outputs/cnn_results") -> pd.DataFrame:
    """Collect metrics from multiple model runs."""
    all_data = []
    
    for run_id in run_ids:
        df = collect_metrics_from_run(run_id, base_dir)
        if not df.empty:
            all_data.append(df)
        else:
            print(f"No data found for run_id: {run_id}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def get_paper_reference_data() -> Dict[str, float]:
    """Get the reference data from the paper for comparison."""
    return {
        'TCR_mean': 0.73,
        'TCR_std': 0.10,
        'NMR_mean': 0.24,
        'NMR_std': 0.17,
        'CMR_mean': 0.06,
        'CMR_std': 0.03,
        'F_mean': 0.57,
        'F_std': 0.08,
    }

def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation for each metric across runs."""
    metrics = ['TCR', 'NMR', 'CMR', 'F']
    summary_data = []
    
    for run_id in df['run_id'].unique():
        run_data = df[df['run_id'] == run_id]
        
        row = {'run_id': run_id}
        for metric in metrics:
            if metric in run_data.columns:
                values = run_data[metric].values
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_scatter_plot(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """Create scatter plots for each metric showing individual fold results with means and std."""
    metrics = ['TCR', 'NMR', 'CMR', 'F']
    paper_ref = get_paper_reference_data()
    
    plt.style.use('default')
    colors = plt.cm.Set1(np.linspace(0, 1, len(df['run_id'].unique()) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot paper reference
        paper_mean = paper_ref[f'{metric}_mean']
        paper_std = paper_ref[f'{metric}_std']
        
        ax.errorbar(0, paper_mean, yerr=paper_std, 
                   fmt='s', markersize=8, capsize=5, capthick=2,
                   color='red', alpha=0.8, label='Paper Reference')
        
        # Plot scatter points and means for each run
        for j, run_id in enumerate(df['run_id'].unique()):
            run_data = df[df['run_id'] == run_id]
            
            # Individual fold results
            x_positions = np.full(len(run_data), j + 1) + np.random.normal(0, 0.05, len(run_data))
            ax.scatter(x_positions, run_data[metric], 
                      alpha=0.6, s=40, color=colors[j + 1], 
                      edgecolors='black', linewidth=0.5)
            
            # Mean with error bars
            run_summary = summary_df[summary_df['run_id'] == run_id]
            if not run_summary.empty:
                mean_val = run_summary.iloc[0][f'{metric}_mean']
                std_val = run_summary.iloc[0][f'{metric}_std']
                ax.errorbar(j + 1, mean_val, yerr=std_val, 
                           fmt='D', markersize=8, capsize=5, capthick=2,
                           color=colors[j + 1], alpha=0.9, 
                           label=f'Run {run_id} (μ±σ)')
        
        ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric} Value')
        ax.set_title(f'{metric} Distribution: Individual Folds and Averages')
        
        # Set x-axis labels
        x_labels = ['Paper\nReference'] + [f'Run\n{run_id}' for run_id in df['run_id'].unique()]
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_scatter_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create scatter plot of model evaluation metrics')
    parser.add_argument('--run_ids', nargs='+', required=True,
                       help='One or more run IDs to analyze (e.g., 250817_224430)')
    parser.add_argument('--base_dir', default='outputs/cnn_results',
                       help='Base directory containing model results')
    parser.add_argument('--output_dir', default='./metrics_analysis',
                       help='Output directory for plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing runs: {args.run_ids}")
    print(f"Output directory: {args.output_dir}")
    
    # Collect all metrics data
    df = collect_all_metrics(args.run_ids, args.base_dir)
    
    if df.empty:
        print("No data found for the specified run IDs.")
        return
    
    print(f"\nFound data for {len(df)} fold evaluations across {len(df['run_id'].unique())} runs")
    
    # Calculate summary statistics
    summary_df = calculate_summary_statistics(df)
    
    # Create scatter plot
    print("Creating scatter plot...")
    create_scatter_plot(df, summary_df, args.output_dir)
    
    print(f"\nScatter plot saved to: {os.path.join(args.output_dir, 'metrics_scatter_plot.png')}")

if __name__ == "__main__":
    main()
