#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd
import argparse
from pathlib import Path


def create_wav_symlinks(original_wav_dir, new_wav_dir):
    """Create symbolic links to all WAV files in the original directory."""
    if not os.path.exists(original_wav_dir):
        print(f"    Warning: WAV directory not found: {original_wav_dir}")
        return 0
    
    os.makedirs(new_wav_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(original_wav_dir, '*.wav'))
    
    for wav_file in wav_files:
        wav_name = os.path.basename(wav_file)
        symlink_path = os.path.join(new_wav_dir, wav_name)
        
        # Remove existing symlink if it exists
        if os.path.islink(symlink_path):
            os.unlink(symlink_path)
        
        # Create new symlink
        os.symlink(wav_file, symlink_path)
    
    print(f"    Created {len(wav_files)} WAV symlinks")
    return len(wav_files)


def process_site(site_name, site_path, output_dir, file_tag_mapping):
    """Process all selection files in a site using the hardcoded file-tag mapping."""
    print(f"Processing site: {site_name}")
    
    # Get the file mapping for this site
    site_mapping = file_tag_mapping.get(site_name, {})
    if not site_mapping:
        print(f"  No mapping found for site {site_name}")
        return
    
    site_output_dir = os.path.join(output_dir, site_name)
    os.makedirs(site_output_dir, exist_ok=True)
    
    processed_files = 0
    site_tags = set()
    
    # Process each file in the mapping
    for file_name, tag in site_mapping.items():
        file_path = os.path.join(site_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}")
            continue
            
        # Skip files marked for ignoring
        if tag == "_IGNORE":
            print(f"  Skipping {file_name} (marked as _IGNORE)")
            continue
            
        site_tags.add(tag)
        
        try:
            # Try tab-separated first, then comma-separated
            try:
                df = pd.read_csv(file_path, sep='\t')
            except:
                try:
                    df = pd.read_csv(file_path, sep=',')
                except Exception as e:
                    print(f"  Error: Could not parse {file_name} as CSV or TSV: {e}")
                    continue
                
            # Add Tags column
            df['Tags'] = tag

            # --- FIX FOR GREENWICH64S2015 FILENAME SUFFIX ---
            if site_name.lower() == 'greenwich64s2015':
                def strip_suffix(filename):
                    if isinstance(filename, str) and filename.endswith('_AWI229-11_SV1057.wav'):
                        return filename.replace('_AWI229-11_SV1057.wav', '.wav')
                    return filename
                if 'Begin File' in df.columns:
                    df['Begin File'] = df['Begin File'].apply(strip_suffix)
                if 'End File' in df.columns:
                    df['End File'] = df['End File'].apply(strip_suffix)
            # --- END FIX ---
            
            # Save as .selections.tags.txt
            output_filename = os.path.basename(file_path).replace('.txt', '.selections.tags.txt')
            output_path = os.path.join(site_output_dir, output_filename)
            df.to_csv(output_path, sep='\t', index=False)
            processed_files += 1
            
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
    
    print(f"  Processed {processed_files} selection files with {len(site_tags)} unique tags")
    print(f"  Tags found: {sorted(site_tags)}")
    
    # Create WAV symlinks
    original_wav_dir = os.path.join(site_path, 'wav')
    new_wav_dir = os.path.join(site_output_dir, 'wav')
    wav_count = create_wav_symlinks(original_wav_dir, new_wav_dir)


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset using hardcoded file-tag mapping')
    parser.add_argument('--raw_data', required=True, help='Original raw data directory path')
    parser.add_argument('--output', required=True, help='Output directory for prepared dataset')
    parser.add_argument('--mapping', default=None, help='Path to file-tag mapping JSON (default: file_tag_mapping.json in the same directory as this script)')
    args = parser.parse_args()
    
    # Determine mapping file path
    if args.mapping:
        mapping_path = args.mapping
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.join(script_dir, 'file_tag_mapping.json')
    
    # Load the file-tag mapping
    try:
        with open(mapping_path, 'r') as f:
            file_tag_mapping = json.load(f)
            # Remove metadata keys
            file_tag_mapping.pop('_ABOUT', None)
            file_tag_mapping.pop('_NOTE', None)
    except Exception as e:
        print(f"Error loading mapping file {mapping_path}: {e}")
        return
    
    raw_data_dir = Path(args.raw_data)
    output_dir = Path(args.output)
    dataset_dir = output_dir / 'prepared_dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing dataset from: {raw_data_dir}")
    print(f"Output directory: {dataset_dir}")
    print(f"Using mapping file: {mapping_path}")
    print(f"This will create:")
    print(f"  - Selection files with Tags column")
    print(f"  - Symbolic links to original WAV files")
    
    # Process each site in the mapping
    for site_name in file_tag_mapping.keys():
        site_path = raw_data_dir / site_name
        if not site_path.exists():
            print(f"Warning: Site directory not found: {site_path}")
            continue
        
        process_site(site_name, site_path, dataset_dir, file_tag_mapping)
    
    print(f"\n{'='*60}")
    print(f"DATASET PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Sites processed: {len(file_tag_mapping)}")
    print(f"Prepared dataset location: {dataset_dir}")


if __name__ == "__main__":
    main() 