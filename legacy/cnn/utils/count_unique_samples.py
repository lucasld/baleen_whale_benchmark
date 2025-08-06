#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def count_unique_events(annotations_dir, config_path=None):
    """
    Counts the number of unique, labeled vocalization events for each
    location and class from the intermediate annotation files.

    Args:
        annotations_dir (str): The path to the directory containing the
                               site-based annotation subdirectories.
        config_path (str, optional): Path to a config.json file to join categories.

    Returns:
        pandas.DataFrame: A DataFrame with locations as rows, classes as columns,
                          and counts of unique events as values. Returns None on error.
    """
    counts = defaultdict(lambda: defaultdict(int))
    
    annotations_path = Path(annotations_dir)
    if not annotations_path.is_dir():
        print(f"Error: Annotations directory not found at '{annotations_dir}'")
        return None

    print(f"Scanning for unique annotations in: {annotations_dir}")
    # The subdirectories are the site names
    site_dirs = [d for d in annotations_path.iterdir() if d.is_dir()]
    
    if not site_dirs:
        print("No site subdirectories found in the annotations directory.")
        return pd.DataFrame()

    total_events_counted = 0
    for site_dir in site_dirs:
        site_name = site_dir.name
        annotation_files = list(site_dir.glob('*_annotations.txt'))
        
        if not annotation_files:
            continue

        for ann_file in annotation_files:
            try:
                df = pd.read_csv(ann_file, sep='\t')
                # A single annotation can have multiple comma-separated tags
                for tags_str in df['Tags'].dropna():
                    for tag in tags_str.split(','):
                        clean_tag = tag.strip()
                        if clean_tag:
                            counts[site_name][clean_tag] += 1
                            total_events_counted += 1
            except Exception as e:
                print(f"Warning: Could not process file {ann_file.name} in {site_name}: {e}")

    if total_events_counted == 0:
        print("No valid annotation events found.")
        return pd.DataFrame()

    print(f"Counted {total_events_counted} unique annotation events across {len(site_dirs)} sites.")

    # Convert to DataFrame
    df = pd.DataFrame(counts).transpose().fillna(0).astype(int)

    # --- Optional: Join categories based on config file ---
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'CATEGORIES_TO_JOIN' in config:
                print("\nJoining categories based on config file...")
                join_map = config['CATEGORIES_TO_JOIN']
                for new_col, old_cols in join_map.items():
                    existing_cols = [c for c in old_cols if c in df.columns]
                    if existing_cols:
                        df[new_col] = df[existing_cols].sum(axis=1)
                        df = df.drop(columns=existing_cols)
                        print(f"  - Joined {existing_cols} into {new_col}")
        except Exception as e:
            print(f"Warning: Could not read or apply config for joining: {e}")

    # --- Finalize DataFrame ---
    # Add a 'Total' column for sums per location
    if not df.empty:
        df['Total'] = df.sum(axis=1)

    # Add a 'Total' row for sums per class
    if not df.empty:
        total_row = df.sum().to_frame().T
        total_row.index = ['Total']
        df = pd.concat([df, total_row])
    
    # Sort columns and index for consistent output
    if not df.empty:
        class_columns = sorted([col for col in df.columns if col != 'Total'])
        final_columns = class_columns + ['Total']
        df = df[final_columns]
        
        locations = sorted([idx for idx in df.index if idx != 'Total'])
        df = df.reindex(locations + ['Total'])

    return df

def main():
    parser = argparse.ArgumentParser(
        description='Count unique vocalization events per location and class from '
                    'processed annotation files to replicate the table from Miller et al. (2021).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('annotations_dir', type=str,
                        help='Path to the directory containing annotation files '
                             'sorted into site subdirectories (e.g., outputs/annotations).')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the output table as a CSV file (optional).')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the project config.json file to join categories.')
    
    args = parser.parse_args()
    
    results_df = count_unique_events(args.annotations_dir, args.config)
    
    if results_df is not None and not results_df.empty:
        if args.output:
            try:
                results_df.to_csv(args.output)
                print(f"\nResults table successfully saved to {args.output}")
            except Exception as e:
                print(f"Error saving to CSV: {e}")
        else:
            print("\n--- Unique Event Composition ---")
            print(results_df.to_string())
            print("\n------------------------------\n")

if __name__ == "__main__":
    main() 