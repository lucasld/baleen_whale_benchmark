#!/usr/bin/env python3
import pathlib
import json
import pandas as pd
import sys
import os
import shutil

# Add src to path
sys.path.insert(0, 'src')

import dataset

def recreate_yolo_labels(run_dir_path):
    run_dir = pathlib.Path(run_dir_path)
    
    # Load config from run_dir (shared across folds)
    config_path = run_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Initialize dataset once
    ds = dataset.SpectrogramDataSet(
        data_dir=config['DATA_DIR'],
        categories=config['CATEGORIES'],
        join_cat=config["CATEGORIES_TO_JOIN"],
        locations=config['LOCATIONS'],
        corrected=config['USE_CORRECTED_DATASET'],
        samples_per_class=config['SAMPLES_PER_CLASS']
    )
    
    # Find all fold directories
    fold_dirs = list(run_dir.glob('fold_*'))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {run_dir}")
    
    for fold_dir in fold_dirs:
        print(f"Processing fold: {fold_dir.name}")
        
        # Load paths_df from CSV
        csv_files = list(fold_dir.glob('data_used_fold_*.csv'))
        if not csv_files:
            print(f"Warning: No data_used_fold_*.csv found in {fold_dir}, skipping.")
            continue
        csv_path = csv_files[0]  # Take the first one
        paths_df = pd.read_csv(csv_path)
        
        # Ensure columns are correct
        if 'path' not in paths_df.columns or 'set' not in paths_df.columns:
            print(f"Warning: CSV in {fold_dir} missing 'path' or 'set' columns, skipping.")
            continue
        
        # Delete existing yolo_dataset if it exists
        yolo_dir = fold_dir / 'yolo_dataset'
        if yolo_dir.exists():
            shutil.rmtree(yolo_dir)
            print(f"Deleted existing yolo_dataset in {fold_dir}")
        
        # Create YOLO dataset
        ds.create_yolo_dataset(paths_df, 'train', yolo_dir)
        ds.create_yolo_dataset(paths_df, 'valid', yolo_dir)
        ds.create_yolo_dataset(paths_df, 'test', yolo_dir)
        
        print(f"YOLO labels recreated at {yolo_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python recreate_yolo_labels.py <run_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]
    recreate_yolo_labels(run_dir)
