#!/usr/bin/env python3
"""
Script to analyze model evaluation metrics across folds.
Calculates averages and standard deviations for TCR, NMR, CMR, and F metrics.
Creates scatter plots and boxplots for visualization.

Usage:
python analyze_model_metrics.py --run_ids 250817_224430 250817_110328 --output_dir ./analysis_output
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import glob

def parse_metrics_from_filename(filename: str) -> Dict[str, float]:
    """
    Parse metrics from evaluation filename.
    Expected format: predictions_fold[location]_confusion_NMR=X.XX_CMR=X.XX_TCR=X.XX_F=X.XX.csv
    """
    # Extract metrics using regex - match digits and decimal points, but not trailing dots
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
            value_str = match.group(1).rstrip('.')  # Remove any trailing dots
            try:
                metrics[metric_name] = float(value_str)
            except ValueError as e:
                print(f"Warning: Could not convert {metric_name} value '{value_str}' to float in filename {filename}: {e}")
        else:
            print(f"Warning: Could not find {metric_name} in filename {filename}")
            
    return metrics

def extract_fold_name(filename: str) -> str:
    """Extract the fold/location name from the filename."""
    # Extract fold name between 'fold' and the first '_confusion'
    match = re.search(r'fold([^_]+)', filename)
    if match:
        return match.group(1)
    return "unknown"

def collect_metrics_from_run(run_id: str, base_dir: str = "outputs/cnn_results") -> pd.DataFrame:
    """
    Collect all metrics from a single model run.
    
    Args:
        run_id: The run identifier (e.g., '250817_224430')
        base_dir: Base directory containing the results
        
    Returns:
        DataFrame with columns: run_id, fold, TCR, NMR, CMR, F
    """
    evaluation_dir = os.path.join(base_dir, run_id, "evaluation")
    
    if not os.path.exists(evaluation_dir):
        print(f"Warning: Evaluation directory not found: {evaluation_dir}")
        return pd.DataFrame()
    
    # Find all confusion matrix files with metrics in filename
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
    """
    Collect metrics from multiple model runs.
    
    Args:
        run_ids: List of run identifiers
        base_dir: Base directory containing the results
        
    Returns:
        Combined DataFrame with all metrics
    """
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

def get_paper_reference_data() -> Dict[str, Dict[str, float]]:
    """
    Get the reference data from the paper for comparison.
    
    Returns:
        Dictionary with paper reference metrics
    """
    return {
        'Paper_Reference': {
            'TCR_mean': 0.73,
            'TCR_std': 0.10,
            'NMR_mean': 0.24,
            'NMR_std': 0.17,
            'CMR_mean': 0.06,
            'CMR_std': 0.03,
            'F_mean': 0.57,
            'F_std': 0.08,
            'TCR_n_folds': 10,  # Assumed based on your data
            'NMR_n_folds': 10,
            'CMR_n_folds': 10,
            'F_n_folds': 10
        }
    }

def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean and standard deviation for each metric across runs.
    
    Args:
        df: DataFrame with metrics data
        
    Returns:
        Summary statistics DataFrame including paper reference
    """
    metrics = ['TCR', 'NMR', 'CMR', 'F']
    
    # Group by run_id and calculate statistics
    summary_data = []
    
    # Add paper reference data first
    paper_ref = get_paper_reference_data()
    summary_data.append(paper_ref['Paper_Reference'])
    
    for run_id in df['run_id'].unique():
        run_data = df[df['run_id'] == run_id]
        
        row = {'run_id': run_id}
        for metric in metrics:
            if metric in run_data.columns:
                values = run_data[metric].values
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
                row[f'{metric}_n_folds'] = len(values)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_n_folds'] = 0
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_scatter_plots(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """
    Create scatter plots for each metric showing individual fold results with means and std.
    """
    metrics = ['TCR', 'NMR', 'CMR', 'F']
    
    # Set up the plotting style
    plt.style.use('default')
    colors = plt.cm.Set1(np.linspace(0, 1, len(df['run_id'].unique()) + 1))  # +1 for paper reference
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # First plot paper reference as a horizontal line with error bars
        paper_mean = summary_df.iloc[0][f'{metric}_mean']
        paper_std = summary_df.iloc[0][f'{metric}_std']
        
        ax.errorbar(0, paper_mean, yerr=paper_std, 
                   fmt='s', markersize=8, capsize=5, capthick=2,
                   color='red', alpha=0.8, label='Paper Reference')
        
        # Plot scatter points and means for each run
        for j, run_id in enumerate(df['run_id'].unique()):
            run_data = df[df['run_id'] == run_id]
            
            # Plot individual fold results as scatter points
            x_positions = np.full(len(run_data), j + 1) + np.random.normal(0, 0.05, len(run_data))
            ax.scatter(x_positions, run_data[metric], 
                      alpha=0.6, s=40, color=colors[j + 1], 
                      edgecolors='black', linewidth=0.5)
            
            # Plot mean with error bars
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
    plt.savefig(os.path.join(output_dir, 'metrics_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_scatter_plots.pdf'), bbox_inches='tight')
    plt.close()

def create_boxplots(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """
    Create boxplots for each metric with individual data points overlaid and paper reference.
    """
    metrics = ['TCR', 'NMR', 'CMR', 'F']
    
    # Create separate plots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get paper reference values
        paper_mean = summary_df.iloc[0][f'{metric}_mean']
        paper_std = summary_df.iloc[0][f'{metric}_std']
        
        # Create data for boxplot (only model runs, not paper reference)
        box_data = []
        labels = ['Paper\nReference']
        colors = plt.cm.Set1(np.linspace(0, 1, len(df['run_id'].unique()) + 1))
        
        # Add paper reference as error bar (no boxplot data)
        ax.errorbar(1, paper_mean, yerr=paper_std, 
                   fmt='s', markersize=10, capsize=8, capthick=3,
                   color='red', alpha=0.8, zorder=5)
        
        # Add model run data
        for j, run_id in enumerate(sorted(df['run_id'].unique())):
            run_values = df[df['run_id'] == run_id][metric].values
            box_data.append(run_values)
            labels.append(f'Run\n{run_id}')
        
        # Create boxplot for model runs only (starting from position 2)
        if box_data:
            positions = list(range(2, len(box_data) + 2))
            bp = ax.boxplot(box_data, positions=positions, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[1:]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Overlay individual points with jitter
            for j, run_id in enumerate(sorted(df['run_id'].unique())):
                run_values = df[df['run_id'] == run_id][metric].values
                x_positions = np.full(len(run_values), j + 2) + np.random.normal(0, 0.05, len(run_values))
                ax.scatter(x_positions, run_values, 
                          alpha=0.8, s=40, color=colors[j + 1], 
                          edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.set_ylabel(f'{metric} Value')
        ax.set_title(f'{metric} Distribution: Paper Reference vs Model Runs')
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_boxplots.pdf'), bbox_inches='tight')
    plt.close()

def create_combined_plot(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """
    Create a combined plot showing all metrics in one figure with paper reference.
    """
    metrics = ['TCR', 'NMR', 'CMR', 'F']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Set up colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(df['run_id'].unique()) + 1))
    
    # Plot paper reference values first
    x_positions_paper = np.arange(len(metrics)) - 0.2
    paper_means = [summary_df.iloc[0][f'{metric}_mean'] for metric in metrics]
    paper_stds = [summary_df.iloc[0][f'{metric}_std'] for metric in metrics]
    
    ax.errorbar(x_positions_paper, paper_means, yerr=paper_stds, 
               fmt='s', markersize=8, capsize=5, capthick=2, 
               color='red', alpha=0.8, label='Paper Reference')
    
    # Plot model runs
    for i, run_id in enumerate(df['run_id'].unique()):
        run_data = df[df['run_id'] == run_id]
        
        # Get means and stds for this run
        run_summary = summary_df[summary_df['run_id'] == run_id].iloc[0]
        run_means = [run_summary[f'{metric}_mean'] for metric in metrics]
        run_stds = [run_summary[f'{metric}_std'] for metric in metrics]
        
        # Plot individual fold points with jitter
        for j, metric in enumerate(metrics):
            fold_values = run_data[metric].values
            x_jitter = np.full(len(fold_values), j + 0.1 * (i + 1)) + np.random.normal(0, 0.02, len(fold_values))
            ax.scatter(x_jitter, fold_values, alpha=0.5, s=25, color=colors[i + 1])
        
        # Plot means with error bars
        x_positions_run = np.arange(len(metrics)) + 0.1 * (i + 1)
        ax.errorbar(x_positions_run, run_means, yerr=run_stds, 
                   fmt='D', markersize=6, capsize=4, capthick=2,
                   color=colors[i + 1], alpha=0.9, 
                   label=f'Run {run_id}')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Metrics: Paper Reference vs Our Implementation')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_combined_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_combined_plot.pdf'), bbox_inches='tight')
    plt.close()

def save_summary_tables(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """
    Save summary tables to CSV files including paper reference.
    """
    # Save raw data
    df.to_csv(os.path.join(output_dir, 'raw_metrics_data.csv'), index=False)
    
    # Save summary statistics
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    
    # Create a formatted summary table
    metrics = ['TCR', 'NMR', 'CMR', 'F']
    formatted_summary = []
    
    for idx, row in summary_df.iterrows():
        if idx == 0:  # Paper reference
            formatted_row = {'Model': 'Paper Reference'}
        else:
            formatted_row = {'Model': f"Run {row['run_id']}"}
            
        for metric in metrics:
            mean_val = row[f'{metric}_mean']
            std_val = row[f'{metric}_std']
            if not np.isnan(mean_val):
                formatted_row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                formatted_row[metric] = "N/A"
        formatted_summary.append(formatted_row)
    
    formatted_df = pd.DataFrame(formatted_summary)
    formatted_df.to_csv(os.path.join(output_dir, 'formatted_summary.csv'), index=False)
    
    print("\nSummary Statistics (including Paper Reference):")
    print("=" * 80)
    print(formatted_df.to_string(index=False))
    
    # Print comparison
    if len(summary_df) > 1:  # We have model results to compare
        print("\n" + "=" * 80)
        print("COMPARISON TO PAPER:")
        print("=" * 80)
        paper_row = summary_df.iloc[0]
        for idx in range(1, len(summary_df)):
            model_row = summary_df.iloc[idx]
            print(f"\nRun {model_row['run_id']} vs Paper Reference:")
            for metric in metrics:
                paper_val = paper_row[f'{metric}_mean']
                model_val = model_row[f'{metric}_mean']
                diff = model_val - paper_val
                diff_pct = (diff / paper_val) * 100 if paper_val != 0 else 0
                print(f"  {metric}: {model_val:.4f} vs {paper_val:.4f} (Δ={diff:+.4f}, {diff_pct:+.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze model evaluation metrics across folds')
    parser.add_argument('--run_ids', nargs='+', required=True,
                       help='One or more run IDs to analyze (e.g., 250817_224430)')
    parser.add_argument('--base_dir', default='outputs/cnn_results',
                       help='Base directory containing model results')
    parser.add_argument('--output_dir', default='./metrics_analysis',
                       help='Output directory for plots and tables')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing runs: {args.run_ids}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Collect all metrics data
    df = collect_all_metrics(args.run_ids, args.base_dir)
    
    if df.empty:
        print("No data found for the specified run IDs.")
        return
    
    print(f"\nFound data for {len(df)} fold evaluations across {len(df['run_id'].unique())} runs")
    print(f"Folds per run: {df.groupby('run_id')['fold'].count().to_dict()}")
    
    # Calculate summary statistics
    summary_df = calculate_summary_statistics(df)
    
    # Create visualizations
    print("\nCreating scatter plots...")
    create_scatter_plots(df, summary_df, args.output_dir)
    
    print("Creating boxplots...")
    create_boxplots(df, summary_df, args.output_dir)
    
    print("Creating combined plot...")
    create_combined_plot(df, summary_df, args.output_dir)
    
    # Save data and summary tables
    print("Saving summary tables...")
    save_summary_tables(df, summary_df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("Generated files:")
    print("- raw_metrics_data.csv")
    print("- summary_statistics.csv") 
    print("- formatted_summary.csv")
    print("- metrics_scatter_plots.png/pdf")
    print("- metrics_boxplots.png/pdf")
    print("- metrics_combined_plot.png/pdf")

if __name__ == "__main__":
    main()
