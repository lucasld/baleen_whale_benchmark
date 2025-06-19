#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def count_samples(spectrograms_dir, config_path=None):
    """
    Counts the number of spectrogram samples for each location and class.

    Args:
        spectrograms_dir (str): The path to the directory containing the
                                class-based spectrogram subdirectories.
        config_path (str, optional): Path to a config.json file to join categories.

    Returns:
        pandas.DataFrame: A DataFrame with locations as rows, classes as columns,
                          and counts as values. Returns None on error.
    """
    counts = defaultdict(lambda: defaultdict(int))
    
    spectrograms_path = Path(spectrograms_dir)
    if not spectrograms_path.is_dir():
        print(f"Error: Spectrogram directory not found at '{spectrograms_dir}'")
        return None

    print(f"Scanning directory: {spectrograms_dir}")
    # The subdirectories are the class names
    class_dirs = [d for d in spectrograms_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("No class subdirectories found in the spectrograms directory.")
        return pd.DataFrame()

    total_files_scanned = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        for spec_file in class_dir.glob('*.png'):
            total_files_scanned += 1
            # Filename format: {spec_ID}_{location_year}_{CLASSNAME}.png
            # Example: 1_Greenwich64S2015_A.png
            parts = spec_file.stem.split('_')
            
            # The location name is expected between the first and last underscore
            if len(parts) >= 3:
                location = "_".join(parts[1:-1])
                counts[location][class_name] += 1
            else:
                print(f"Warning: Could not parse location from filename: {spec_file.name}")

    if total_files_scanned == 0:
        print("No .png files found in class subdirectories.")
        return pd.DataFrame()

    print(f"Counted {total_files_scanned} samples across {len(class_dirs)} classes.")

    # Convert to DataFrame
    df = pd.DataFrame(counts)
    df = df.transpose()  # Locations as rows, classes as columns
    df = df.fillna(0).astype(int)

    # --- Optional: Join categories based on config file ---
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'CATEGORIES_TO_JOIN' in config:
                print("\nJoining categories based on config file...")
                join_map = config['CATEGORIES_TO_JOIN']
                for new_col, old_cols in join_map.items():
                    # Ensure old_cols exist in the dataframe before trying to join
                    existing_cols = [c for c in old_cols if c in df.columns]
                    if existing_cols:
                        df[new_col] = df[existing_cols].sum(axis=1)
                        df = df.drop(columns=existing_cols)
                        print(f"  - Joined {existing_cols} into {new_col}")
        except Exception as e:
            print(f"Warning: Could not read or apply config for joining: {e}")

    # --- Finalize DataFrame ---
    # Add a 'Total' column for sums per location
    df['Total'] = df.sum(axis=1)

    # Add a 'Total' row for sums per class
    total_row = df.sum().to_frame().T
    total_row.index = ['Total']
    df = pd.concat([df, total_row])
    
    # Sort columns alphabetically for consistency, keeping 'Total' at the end
    class_columns = sorted([col for col in df.columns if col != 'Total'])
    final_columns = class_columns + ['Total']
    df = df[final_columns]
    
    # Sort index (locations) alphabetically, but put 'Total' at the end.
    locations = sorted([idx for idx in df.index if idx != 'Total'])
    df = df.reindex(locations + ['Total'])

    return df

def main():
    parser = argparse.ArgumentParser(
        description='Count spectrogram samples per location and class from a processed dataset, '
                    'replicating the table from Miller et al. (2021).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('spectrograms_dir', type=str,
                        help='Path to the directory containing spectrograms '
                             'sorted into class subdirectories (e.g., outputs/spectrograms).')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the output table as a CSV file (optional). '
                             'If not provided, the table is printed to the console.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the project config.json file. If provided, '
                             'categories will be joined according to "CATEGORIES_TO_JOIN".')
    
    args = parser.parse_args()
    
    results_df = count_samples(args.spectrograms_dir, args.config)
    
    if results_df is not None and not results_df.empty:
        if args.output:
            try:
                results_df.to_csv(args.output)
                print(f"\nResults table successfully saved to {args.output}")
            except Exception as e:
                print(f"Error saving to CSV: {e}")
        else:
            print("\n--- Dataset Composition ---")
            # Use to_string() to ensure the full table is printed without truncation
            print(results_df.to_string())
            print("\n-------------------------\n")

if __name__ == "__main__":
    main() 