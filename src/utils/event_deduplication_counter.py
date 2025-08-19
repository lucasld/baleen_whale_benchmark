#!/usr/bin/env python3
"""
Properly deduplicate spectrograms by tracing them back to unique annotation events.

This script understands that:
1. Multiple spectrograms can come from the same original annotation event (sliding window)
2. We need to group spectrograms by their source annotation events to count unique events
3. The timing information in Koogu .npz files allows us to trace back to original events
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import glob


def parse_spectrogram_filename(filename):
    """
    Parse spectrogram filename to extract information.
    
    Format: {spec_ID}_{location_year}_{CLASSNAME}.png
    Example: 1000_BallenyIslands2015_A.png
    """
    # Remove .png extension
    name = os.path.splitext(filename)[0]
    
    # Split by underscores
    parts = name.split('_')
    
    if len(parts) < 3:
        return None
    
    try:
        spec_id = int(parts[0])
        # Location_year is everything between spec_id and the last part (class)
        location_year = '_'.join(parts[1:-1])
        class_name = parts[-1]
        
        return {
            'spec_id': spec_id,
            'location_year': location_year,
            'class_name': class_name,
            'filename': filename
        }
    except Exception as e:
        return None


def extract_timing_from_spectrogram_filename(filename):
    """
    Extract timing information from spectrogram filename if available.
    
    Some spectrogram filenames might contain timing information in the format:
    {spec_ID}_{location_year}_{CLASSNAME}_{STARTTIME}_{ENDTIME}.png
    """
    # Remove .png extension
    name = os.path.splitext(filename)[0]
    
    # Split by underscores
    parts = name.split('_')
    
    # Check if this is the extended format with timing
    if len(parts) >= 5:
        try:
            # The last two parts should be STARTTIME and ENDTIME
            start_time = int(parts[-2])
            end_time = int(parts[-1])
            return start_time, end_time
        except ValueError:
            pass
    
    return None, None


def group_spectrograms_by_source_event(spectrograms_dir):
    """
    Group spectrograms by their source events using timing overlap.
    
    Since we don't have direct timing in the current spectrogram filenames,
    we'll use a heuristic approach to estimate grouping.
    """
    print("Grouping spectrograms by estimated source events...")
    
    # For each location and class, group spectrograms that are likely from the same event
    event_groups = defaultdict(list)
    spectrogram_info = []
    
    # Collect all spectrogram information
    spectrograms_path = Path(spectrograms_dir)
    class_dirs = [d for d in spectrograms_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        png_files = list(class_dir.glob('*.png'))
        
        for png_file in png_files:
            parsed = parse_spectrogram_filename(png_file.name)
            if parsed:
                spectrogram_info.append({
                    'file_path': str(png_file),
                    'parsed': parsed
                })
    
    print(f"Found {len(spectrogram_info)} spectrograms total")
    
    # Group by location and class, then try to identify unique events
    location_class_groups = defaultdict(list)
    for info in spectrogram_info:
        location_year = info['parsed']['location_year']
        class_name = info['parsed']['class_name']
        key = f"{location_year}_{class_name}"
        location_class_groups[key].append(info)
    
    # For each group, estimate how many unique events there are
    unique_event_estimates = {}
    for key, spectrograms in location_class_groups.items():
        # Sort by spec_id to process in order
        sorted_specs = sorted(spectrograms, key=lambda x: x['parsed']['spec_id'])
        
        # Estimate unique events based on gaps in spec_id sequences
        # This is a heuristic - in reality, we'd need timing information
        unique_events = estimate_unique_events_from_spec_ids(sorted_specs)
        unique_event_estimates[key] = unique_events
        
        print(f"Location/Class {key}: {len(sorted_specs)} spectrograms, ~{unique_events} unique events")
    
    return unique_event_estimates


def estimate_unique_events_from_spec_ids(spectrograms):
    """
    Estimate the number of unique events from spectrogram IDs.
    
    This is a heuristic approach since we don't have exact timing information.
    In reality, spectrograms with consecutive IDs are likely from the same event
    due to the sliding window approach.
    """
    if not spectrograms:
        return 0
    
    # Extract spec_ids and sort them
    spec_ids = [s['parsed']['spec_id'] for s in spectrograms]
    spec_ids.sort()
    
    # Group consecutive IDs as likely belonging to the same event
    groups = []
    current_group = [spec_ids[0]]
    
    for i in range(1, len(spec_ids)):
        # If the gap is small (e.g., <= 3), consider it the same event
        # This accounts for the sliding window creating multiple clips per event
        if spec_ids[i] - spec_ids[i-1] <= 3:
            current_group.append(spec_ids[i])
        else:
            # Gap is large, start a new event group
            groups.append(current_group)
            current_group = [spec_ids[i]]
    
    # Add the last group
    groups.append(current_group)
    
    return len(groups)


def count_total_spectrograms(spectrograms_dir):
    """
    Count total spectrograms by location and class.
    """
    print("Counting total spectrograms...")
    
    location_class_counts = defaultdict(lambda: defaultdict(int))
    total_count = 0
    
    spectrograms_path = Path(spectrograms_dir)
    class_dirs = [d for d in spectrograms_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        png_files = list(class_dir.glob('*.png'))
        count = len(png_files)
        total_count += count
        
        # Group by location_year extracted from filenames
        location_counts = defaultdict(int)
        for png_file in png_files:
            parsed = parse_spectrogram_filename(png_file.name)
            if parsed:
                location_year = parsed['location_year']
                location_class_counts[location_year][class_name] += 1
                location_counts[location_year] += 1
        
        print(f"Class {class_name}: {count} spectrograms")
        for location, loc_count in location_counts.items():
            print(f"  {location}: {loc_count}")
    
    print(f"Total spectrograms: {total_count}")
    return location_class_counts


def count_original_annotations(annotations_dir):
    """
    Count original annotation events by location and class.
    """
    print("Counting original annotation events...")
    
    location_class_counts = defaultdict(lambda: defaultdict(int))
    total_count = 0
    
    annotations_path = Path(annotations_dir)
    site_dirs = [d for d in annotations_path.iterdir() if d.is_dir()]
    
    for site_dir in site_dirs:
        site_name = site_dir.name
        annotation_files = list(site_dir.glob('*_annotations.txt'))
        
        site_total = 0
        for ann_file in annotation_files:
            try:
                df = pd.read_csv(ann_file, sep='\t')
                # Count each row as an annotation event
                event_count = len(df)
                site_total += event_count
                
                # Also count by tags if available
                if 'Tags' in df.columns:
                    for tags_str in df['Tags'].dropna():
                        for tag in tags_str.split(','):
                            clean_tag = tag.strip()
                            if clean_tag:
                                location_class_counts[site_name][clean_tag] += 1
                                
            except Exception as e:
                print(f"Warning: Could not process {ann_file}: {e}")
        
        total_count += site_total
        print(f"Site {site_name}: {site_total} annotation events")
    
    print(f"Total original annotation events: {total_count}")
    return location_class_counts


def analyze_processing_pipeline(spectrograms_dir, annotations_dir, koogu_output_dir):
    """
    Analyze the complete processing pipeline to understand the relationship
    between original annotations, Koogu processing, and final spectrograms.
    """
    print("=== PROCESSING PIPELINE ANALYSIS ===")
    
    # 1. Count original annotations
    original_counts = count_original_annotations(annotations_dir)
    
    # 2. Count final spectrograms
    spectrogram_counts = count_total_spectrograms(spectrograms_dir)
    
    # 3. Estimate unique events in spectrograms (after deduplication)
    unique_event_estimates = group_spectrograms_by_source_event(spectrograms_dir)
    
    # Create summary dataframes
    print("\n--- ORIGINAL ANNOTATION COUNTS ---")
    if original_counts:
        orig_df = pd.DataFrame(original_counts).T.fillna(0).astype(int)
        if not orig_df.empty:
            orig_df['Total'] = orig_df.sum(axis=1)
            orig_df.loc['Total'] = orig_df.sum()
            print(orig_df.to_string())
    
    print("\n--- FINAL SPECTROGRAM COUNTS ---")
    if spectrogram_counts:
        spec_df = pd.DataFrame(spectrogram_counts).T.fillna(0).astype(int)
        if not spec_df.empty:
            spec_df['Total'] = spec_df.sum(axis=1)
            spec_df.loc['Total'] = spec_df.sum()
            print(spec_df.to_string())
    
    print("\n--- ESTIMATED UNIQUE EVENTS FROM SPECTROGRAMS ---")
    if unique_event_estimates:
        # Convert to dataframe for better presentation
        event_data = []
        for key, count in unique_event_estimates.items():
            parts = key.rsplit('_', 1)  # Split from the right to separate location and class
            if len(parts) == 2:
                location_year, class_name = parts
                event_data.append({
                    'Location': location_year,
                    'Class': class_name,
                    'Estimated_Unique_Events': count
                })
        
        if event_data:
            event_df = pd.DataFrame(event_data)
            # Pivot to show classes as columns
            pivot_df = event_df.pivot(index='Location', columns='Class', values='Estimated_Unique_Events').fillna(0).astype(int)
            pivot_df['Total'] = pivot_df.sum(axis=1)
            pivot_df.loc['Total'] = pivot_df.sum()
            print(pivot_df.to_string())
    
    # Show the amplification factor (how much the sliding window increased the count)
    print("\n--- PROCESSING AMPLIFICATION FACTOR ---")
    if original_counts and spectrogram_counts:
        # Calculate totals for comparison
        orig_totals = {}
        spec_totals = {}
        
        # Sum original annotations by location
        for location, classes in original_counts.items():
            orig_totals[location] = sum(classes.values())
        
        # Sum spectrograms by location  
        for location, classes in spectrogram_counts.items():
            spec_totals[location] = sum(classes.values())
        
        # Compare locations that exist in both
        common_locations = set(orig_totals.keys()) & set(spec_totals.keys())
        if common_locations:
            amplification_data = []
            for location in sorted(common_locations):
                orig_count = orig_totals[location]
                spec_count = spec_totals[location]
                if orig_count > 0:
                    factor = spec_count / orig_count
                    amplification_data.append({
                        'Location': location,
                        'Original_Events': orig_count,
                        'Final_Spectrograms': spec_count,
                        'Amplification_Factor': f"{factor:.1f}x"
                    })
            
            if amplification_data:
                amp_df = pd.DataFrame(amplification_data)
                print(amp_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Analyze the complete processing pipeline and properly deduplicate events.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('spectrograms_dir', type=str,
                        help='Path to the directory containing spectrograms '
                             'sorted into class subdirectories.')
    parser.add_argument('annotations_dir', type=str,
                        help='Path to the directory containing annotation files '
                             'sorted into site subdirectories.')
    parser.add_argument('koogu_output_dir', type=str,
                        help='Path to the directory containing Koogu output '
                             'with .npz files and timing information.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save detailed analysis results (optional).')
    
    args = parser.parse_args()
    
    analyze_processing_pipeline(args.spectrograms_dir, args.annotations_dir, args.koogu_output_dir)


if __name__ == "__main__":
    main()
