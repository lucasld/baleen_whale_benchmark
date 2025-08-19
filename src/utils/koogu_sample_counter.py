#!/usr/bin/env python3
"""
Count samples from Koogu output by location and class.

This script analyzes the .npz files in the Koogu output directory to count
the actual number of samples produced for each class at each location.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import glob


def analyze_koogu_output(koogu_output_dir):
    """
    Analyze Koogu output to count samples by location and class.
    
    Each .npz file contains clips with labels. We need to count how many
    clips belong to each class for each location.
    """
    print("Analyzing Koogu output...")
    
    # Counters for samples by location and class
    location_class_counts = defaultdict(lambda: defaultdict(int))
    location_total_counts = defaultdict(int)
    
    # Overall statistics
    total_samples = 0
    class_totals = defaultdict(int)
    
    # Get all site directories
    koogu_path = Path(koogu_output_dir)
    site_dirs = [d for d in koogu_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(site_dirs)} sites with Koogu output")
    
    for site_dir in site_dirs:
        site_name = site_dir.name
        print(f"\nProcessing site: {site_name}")
        
        # Load class mapping for this site
        classes_file = site_dir / 'classes_list.json'
        if not classes_file.exists():
            # Try alternative name
            classes_file = site_dir / 'detected_classes.json'
            
        if not classes_file.exists():
            print(f"  Warning: No classes file found for {site_name}, skipping")
            continue
            
        try:
            with open(classes_file, 'r') as f:
                label_list = json.load(f)
            print(f"  Classes: {label_list}")
        except Exception as e:
            print(f"  Error reading classes file: {e}, skipping")
            continue
        
        # Create class name to index mapping
        class_to_idx = {class_name: idx for idx, class_name in enumerate(label_list)}
        
        # Process all .npz files for this site
        npz_files = list(site_dir.glob('*.npz'))
        site_sample_count = 0
        
        print(f"  Found {len(npz_files)} .npz files")
        
        for npz_file in npz_files:
            try:
                # Load the .npz file
                data = np.load(npz_file)
                
                # Get labels - this determines class assignment for each clip
                if 'labels' in data.files:
                    labels = data['labels']
                    
                    # Handle different label formats
                    if labels.ndim == 2 and labels.shape[1] == len(label_list):
                        # One-hot encoded labels
                        # Count clips assigned to each class
                        for i in range(labels.shape[0]):
                            # Find which classes this clip belongs to (where label = 1)
                            class_indices = np.where(labels[i] == 1)[0]
                            for class_idx in class_indices:
                                if class_idx < len(label_list):
                                    class_name = label_list[class_idx]
                                    location_class_counts[site_name][class_name] += 1
                                    class_totals[class_name] += 1
                                    site_sample_count += 1
                                    total_samples += 1
                    elif labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
                        # Single label per clip (direct class indices)
                        flat_labels = labels.flatten() if labels.ndim > 1 else labels
                        for label_idx in flat_labels:
                            if 0 <= label_idx < len(label_list):
                                class_name = label_list[int(label_idx)]
                                location_class_counts[site_name][class_name] += 1
                                class_totals[class_name] += 1
                                site_sample_count += 1
                                total_samples += 1
                    else:
                        print(f"  Warning: Unexpected label shape {labels.shape} in {npz_file.name}")
                else:
                    print(f"  Warning: No 'labels' array in {npz_file.name}")
                    
            except Exception as e:
                print(f"  Error processing {npz_file.name}: {e}")
                continue
        
        location_total_counts[site_name] = site_sample_count
        print(f"  Total samples for {site_name}: {site_sample_count}")
    
    print(f"\n=== KOOGU SAMPLE COUNT SUMMARY ===")
    print(f"Total samples across all sites: {total_samples:,}")
    
    # Create detailed dataframe
    if location_class_counts:
        # Convert to DataFrame
        df = pd.DataFrame(location_class_counts).T.fillna(0).astype(int)
        
        # Add totals
        df['Total'] = df.sum(axis=1)
        class_totals_series = pd.Series(class_totals)
        df.loc['Total'] = class_totals_series.reindex(df.columns, fill_value=0)
        
        print("\n--- KOOGU SAMPLE COUNTS BY LOCATION AND CLASS ---")
        print(df.to_string())
        
        # Show class distribution
        print("\n--- CLASS DISTRIBUTION ---")
        class_dist = pd.DataFrame(class_totals_series, columns=['Sample_Count'])
        class_dist['Percentage'] = (class_dist['Sample_Count'] / total_samples * 100).round(2)
        class_dist = class_dist.sort_values('Sample_Count', ascending=False)
        print(class_dist.to_string())
        
        # Show site distribution
        print("\n--- SITE DISTRIBUTION ---")
        site_dist = pd.DataFrame([(site, count) for site, count in location_total_counts.items()], 
                                columns=['Site', 'Sample_Count'])
        site_dist['Percentage'] = (site_dist['Sample_Count'] / total_samples * 100).round(2)
        site_dist = site_dist.sort_values('Sample_Count', ascending=False)
        print(site_dist.to_string(index=False))
        
        return df, class_dist, site_dist
    
    return None, None, None


def compare_with_annotations(annotations_dir, koogu_results_df):
    """
    Compare Koogu sample counts with original annotation counts.
    """
    if koogu_results_df is None:
        return
    
    print("\n=== COMPARISON WITH ORIGINAL ANNOTATIONS ===")
    
    # Load original annotation counts (simplified version)
    location_class_counts = defaultdict(lambda: defaultdict(int))
    
    annotations_path = Path(annotations_dir)
    if annotations_path.exists():
        site_dirs = [d for d in annotations_path.iterdir() if d.is_dir()]
        
        for site_dir in site_dirs:
            site_name = site_dir.name
            annotation_files = list(site_dir.glob('*_annotations.txt'))
            
            for ann_file in annotation_files:
                try:
                    df = pd.read_csv(ann_file, sep='\t')
                    # Count each row as an annotation event
                    if 'Tags' in df.columns:
                        for tags_str in df['Tags'].dropna():
                            for tag in tags_str.split(','):
                                clean_tag = tag.strip()
                                if clean_tag:
                                    location_class_counts[site_name][clean_tag] += 1
                except Exception as e:
                    print(f"Warning: Could not process {ann_file}: {e}")
    
    # Create annotation dataframe for comparison
    if location_class_counts:
        ann_df = pd.DataFrame(location_class_counts).T.fillna(0).astype(int)
        ann_df['Total'] = ann_df.sum(axis=1)
        
        # Get class totals for annotations
        ann_class_totals = ann_df.sum()[:-1]  # Exclude 'Total' column
        
        print("\n--- ORIGINAL ANNOTATION COUNTS ---")
        print(ann_df.to_string())
        
        # Compare totals
        print("\n--- AMPLIFICATION FACTOR (KOOGU SAMPLES / ORIGINAL ANNOTATIONS) ---")
        comparison_data = []
        
        # Get all sites that appear in either dataset
        all_sites = set(koogu_results_df.index) | set(ann_df.index)
        all_sites.discard('Total')  # Remove totals
        
        for site in sorted(all_sites):
            koogu_count = koogu_results_df.loc[site, 'Total'] if site in koogu_results_df.index else 0
            ann_count = ann_df.loc[site, 'Total'] if site in ann_df.index else 0
            
            if ann_count > 0 and koogu_count > 0:
                factor = koogu_count / ann_count
                comparison_data.append({
                    'Site': site,
                    'Original_Annotations': ann_count,
                    'Koogu_Samples': koogu_count,
                    'Amplification_Factor': f"{factor:.1f}x"
                })
            elif ann_count > 0:
                comparison_data.append({
                    'Site': site,
                    'Original_Annotations': ann_count,
                    'Koogu_Samples': 0,
                    'Amplification_Factor': "No Koogu data"
                })
            elif koogu_count > 0:
                comparison_data.append({
                    'Site': site,
                    'Original_Annotations': 0,
                    'Koogu_Samples': koogu_count,
                    'Amplification_Factor': "No annotation data"
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            print(comp_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Count samples from Koogu output by location and class.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('koogu_output_dir', type=str,
                        help='Path to the directory containing Koogu output '
                             'with site subdirectories and .npz files.')
    parser.add_argument('--annotations_dir', type=str, default=None,
                        help='Path to the directory containing original annotation files '
                             'for comparison (optional).')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save detailed results as CSV (optional).')
    
    args = parser.parse_args()
    
    # Analyze Koogu output
    koogu_df, class_dist, site_dist = analyze_koogu_output(args.koogu_output_dir)
    
    # Compare with original annotations if provided
    if args.annotations_dir and koogu_df is not None:
        compare_with_annotations(args.annotations_dir, koogu_df)
    
    # Save results if requested
    if args.output and koogu_df is not None:
        try:
            koogu_df.to_csv(args.output)
            print(f"\nDetailed results saved to {args.output}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")


if __name__ == "__main__":
    main()
