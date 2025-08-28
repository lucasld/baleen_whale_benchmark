#!/usr/bin/env python3
"""
Compare spectrogram counts with Koogu sample counts to identify any dropped samples.

This script analyzes whether the 1:1 mapping from Koogu samples to spectrograms
is actually maintained, or if samples are being dropped during spectrogram generation.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import glob


def count_koogu_samples(koogu_output_dir):
    """
    Count total samples from Koogu output by location and class.
    """
    print("Counting Koogu samples...")
    
    location_class_counts = defaultdict(lambda: defaultdict(int))
    location_total_counts = defaultdict(int)
    total_samples = 0
    
    koogu_path = Path(koogu_output_dir)
    site_dirs = [d for d in koogu_path.iterdir() if d.is_dir()]
    
    for site_dir in site_dirs:
        site_name = site_dir.name
        print(f"  Processing {site_name}...")
        
        # Load class mapping
        classes_file = site_dir / 'classes_list.json'
        if not classes_file.exists():
            classes_file = site_dir / 'detected_classes.json'
            
        if not classes_file.exists():
            print(f"    Warning: No classes file found for {site_name}, skipping")
            continue
            
        try:
            with open(classes_file, 'r') as f:
                label_list = json.load(f)
        except Exception as e:
            print(f"    Error reading classes file: {e}, skipping")
            continue
        
        # Process all .npz files for this site
        npz_files = list(site_dir.glob('*.npz'))
        site_sample_count = 0
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                if 'labels' in data.files:
                    labels = data['labels']
                    
                    # Handle different label formats
                    if labels.ndim == 2 and labels.shape[1] == len(label_list):
                        # One-hot encoded labels
                        for i in range(labels.shape[0]):
                            class_indices = np.where(labels[i] == 1)[0]
                            for class_idx in class_indices:
                                if class_idx < len(label_list):
                                    class_name = label_list[class_idx]
                                    location_class_counts[site_name][class_name] += 1
                                    site_sample_count += 1
                                    total_samples += 1
                    elif labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
                        # Single label per clip
                        flat_labels = labels.flatten() if labels.ndim > 1 else labels
                        for label_idx in flat_labels:
                            if 0 <= label_idx < len(label_list):
                                class_name = label_list[int(label_idx)]
                                location_class_counts[site_name][class_name] += 1
                                site_sample_count += 1
                                total_samples += 1
                                
            except Exception as e:
                print(f"    Error processing {npz_file.name}: {e}")
                continue
        
        location_total_counts[site_name] = site_sample_count
        print(f"    {site_name}: {site_sample_count} samples")
    
    print(f"Total Koogu samples: {total_samples:,}")
    return location_class_counts, location_total_counts


def count_spectrograms(spectrograms_dir):
    """
    Count total spectrograms by location and class.
    """
    print("Counting spectrograms...")
    
    location_class_counts = defaultdict(lambda: defaultdict(int))
    location_total_counts = defaultdict(int)
    total_spectrograms = 0
    
    spectrograms_path = Path(spectrograms_dir)
    class_dirs = [d for d in spectrograms_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        png_files = list(class_dir.glob('*.png'))
        count = len(png_files)
        total_spectrograms += count
        
        # Group by location_year extracted from filenames
        location_counts = defaultdict(int)
        for png_file in png_files:
            # Parse filename: {spec_ID}_{location_year}_{CLASSNAME}.png
            parts = png_file.stem.split('_')
            if len(parts) >= 3:
                location_year = '_'.join(parts[1:-1])
                location_class_counts[location_year][class_name] += 1
                location_counts[location_year] += 1
        
        print(f"  Class {class_name}: {count} spectrograms")
        for location, loc_count in location_counts.items():
            print(f"    {location}: {loc_count}")
    
    # Calculate totals per location
    for location, classes in location_class_counts.items():
        location_total_counts[location] = sum(classes.values())
    
    print(f"Total spectrograms: {total_spectrograms:,}")
    return location_class_counts, location_total_counts


def analyze_sample_loss(koogu_counts, spectrogram_counts, koogu_totals, spectrogram_totals):
    """
    Analyze where samples might be getting lost between Koogu and spectrogram generation.
    """
    print("\n=== SAMPLE LOSS ANALYSIS ===")
    
    # Get all locations that appear in either dataset
    all_locations = set(koogu_counts.keys()) | set(spectrogram_counts.keys())
    
    # Create comparison dataframe
    comparison_data = []
    
    for location in sorted(all_locations):
        koogu_total = koogu_totals.get(location, 0)
        spectrogram_total = spectrogram_totals.get(location, 0)
        
        if koogu_total > 0 and spectrogram_total > 0:
            loss_count = koogu_total - spectrogram_total
            loss_percentage = (loss_count / koogu_total * 100) if koogu_total > 0 else 0
            status = "✅ Complete" if loss_count == 0 else f"❌ Lost {loss_count:,} samples"
            
            comparison_data.append({
                'Location': location,
                'Koogu_Samples': koogu_total,
                'Spectrograms': spectrogram_total,
                'Samples_Lost': loss_count,
                'Loss_Percentage': f"{loss_percentage:.1f}%",
                'Status': status
            })
        elif koogu_total > 0:
            comparison_data.append({
                'Location': location,
                'Koogu_Samples': koogu_total,
                'Spectrograms': 0,
                'Samples_Lost': koogu_total,
                'Loss_Percentage': "100.0%",
                'Status': "❌ No spectrograms generated"
            })
        elif spectrogram_total > 0:
            comparison_data.append({
                'Location': location,
                'Koogu_Samples': 0,
                'Spectrograms': spectrogram_total,
                'Samples_Lost': 0,
                'Loss_Percentage': "0.0%",
                'Status': "⚠️ Spectrograms without Koogu data"
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        print(comp_df.to_string(index=False))
        
        # Summary statistics
        total_koogu = sum(koogu_totals.values())
        total_spectrograms = sum(spectrogram_totals.values())
        total_lost = total_koogu - total_spectrograms
        
        print(f"\n--- SUMMARY ---")
        print(f"Total Koogu samples: {total_koogu:,}")
        print(f"Total spectrograms: {total_spectrograms:,}")
        print(f"Total samples lost: {total_lost:,}")
        print(f"Overall loss rate: {(total_lost/total_koogu*100):.1f}%" if total_koogu > 0 else "N/A")
        
        # Sites with complete processing
        complete_sites = [row['Location'] for row in comparison_data if row['Samples_Lost'] == 0]
        incomplete_sites = [row['Location'] for row in comparison_data if row['Samples_Lost'] > 0]
        
        print(f"Sites with complete processing: {len(complete_sites)}")
        print(f"Sites with sample loss: {len(incomplete_sites)}")
        
        if incomplete_sites:
            print(f"Sites with sample loss: {', '.join(incomplete_sites)}")
    
    return comparison_data


def analyze_class_level_loss(koogu_counts, spectrogram_counts):
    """
    Analyze sample loss at the class level to see if certain classes are more affected.
    """
    print("\n=== CLASS-LEVEL SAMPLE LOSS ANALYSIS ===")
    
    # Get all classes that appear in either dataset
    all_classes = set()
    for location_data in koogu_counts.values():
        all_classes.update(location_data.keys())
    for location_data in spectrogram_counts.values():
        all_classes.update(location_data.keys())
    
    # Create class-level comparison
    class_comparison = []
    
    for class_name in sorted(all_classes):
        koogu_total = sum(location_data.get(class_name, 0) for location_data in koogu_counts.values())
        spectrogram_total = sum(location_data.get(class_name, 0) for location_data in spectrogram_counts.values())
        
        if koogu_total > 0:
            loss_count = koogu_total - spectrogram_total
            loss_percentage = (loss_count / koogu_total * 100) if koogu_total > 0 else 0
            
            class_comparison.append({
                'Class': class_name,
                'Koogu_Samples': koogu_total,
                'Spectrograms': spectrogram_total,
                'Samples_Lost': loss_count,
                'Loss_Percentage': f"{loss_percentage:.1f}%",
                'Status': "✅ Complete" if loss_count == 0 else f"❌ Lost {loss_count:,} samples"
            })
    
    if class_comparison:
        class_df = pd.DataFrame(class_comparison)
        print(class_df.to_string(index=False))
    
    return class_comparison


def main():
    parser = argparse.ArgumentParser(
        description='Compare Koogu sample counts with spectrogram counts to identify sample loss.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('koogu_output_dir', type=str,
                        default='datasets/preprocessed_dataset/koogu_output',
                        help='Path to the directory containing Koogu output '
                             'with site subdirectories and .npz files.')
    parser.add_argument('spectrograms_dir', type=str,
                        default="datasets/preprocessed_dataset/spectrograms",
                        help='Path to the directory containing spectrograms '
                             'sorted into class subdirectories.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save detailed comparison results as CSV (optional).')
    
    args = parser.parse_args()
    
    print("=== KOOGU vs SPECTROGRAM SAMPLE COUNT COMPARISON ===\n")
    
    # Count Koogu samples
    koogu_counts, koogu_totals = count_koogu_samples(args.koogu_output_dir)
    
    print()  # Empty line for readability
    
    # Count spectrograms
    spectrogram_counts, spectrogram_totals = count_spectrograms(args.spectrograms_dir)
    
    # Analyze sample loss
    comparison_data = analyze_sample_loss(koogu_counts, spectrogram_counts, koogu_totals, spectrogram_totals)
    
    # Analyze class-level loss
    class_comparison = analyze_class_level_loss(koogu_counts, spectrogram_counts)
    
    # Save results if requested
    if args.output and comparison_data:
        try:
            # Combine both analyses
            combined_data = []
            for row in comparison_data:
                combined_data.append({
                    'Location': row['Location'],
                    'Koogu_Samples': row['Koogu_Samples'],
                    'Spectrograms': row['Spectrograms'],
                    'Samples_Lost': row['Samples_Lost'],
                    'Loss_Percentage': row['Loss_Percentage'],
                    'Status': row['Status']
                })
            
            # Add class-level data
            for row in class_comparison:
                combined_data.append({
                    'Location': f"CLASS_{row['Class']}",
                    'Koogu_Samples': row['Koogu_Samples'],
                    'Spectrograms': row['Spectrograms'],
                    'Samples_Lost': row['Samples_Lost'],
                    'Loss_Percentage': row['Loss_Percentage'],
                    'Status': row['Status']
                })
            
            combined_df = pd.DataFrame(combined_data)
            combined_df.to_csv(args.output, index=False)
            print(f"\nDetailed comparison results saved to {args.output}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
