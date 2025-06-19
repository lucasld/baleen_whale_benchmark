#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd
import argparse
from pathlib import Path


def extract_tag_from_filename(filename):
    """Extract tag from selection filename pattern: Site.Tag.selections.txt"""
    basename = os.path.basename(filename)
    if '.selections.txt' in basename:
        parts = basename.replace('.selections.txt', '').split('.')
        if len(parts) >= 2:
            return '.'.join(parts[1:])  # Everything after site name
    return None


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


def process_site(site_path, output_dir):
    """Process all selection files in a site, add Tags column, and create WAV symlinks."""
    site_name = os.path.basename(site_path)
    print(f"Processing site: {site_name}")
    
    selection_files = glob.glob(os.path.join(site_path, '*.selections.txt'))
    if not selection_files:
        print(f"  No selection files found in {site_name}")
        return set()
    
    site_tags = set()
    site_output_dir = os.path.join(output_dir, site_name)
    os.makedirs(site_output_dir, exist_ok=True)
    
    # Process selection files
    for file_path in selection_files:
        tag = extract_tag_from_filename(file_path)
        if not tag:
            print(f"  Warning: Could not extract tag from {os.path.basename(file_path)}")
            continue
            
        site_tags.add(tag)
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            df['Tags'] = tag
            
            # Save as .selections.tags.txt
            output_filename = os.path.basename(file_path).replace('.selections.txt', '.selections.tags.txt')
            output_path = os.path.join(site_output_dir, output_filename)
            df.to_csv(output_path, sep='\t', index=False)
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
    
    print(f"  Processed {len(selection_files)} selection files with {len(site_tags)} unique tags")
    
    # Create WAV symlinks
    original_wav_dir = os.path.join(site_path, 'wav')
    new_wav_dir = os.path.join(site_output_dir, 'wav')
    wav_count = create_wav_symlinks(original_wav_dir, new_wav_dir)
    
    print(f"  Tags found: {sorted(site_tags)}")
    return site_tags


def create_tag_mapping_template(all_tags_by_site, output_path):
    """Create a tag mapping template with unique tags and their locations."""
    template = {
        "_ABOUT": "Template for tag mapping. Replace location lists with target categories.",
        "_NOTES": "Generated automatically from selection filenames"
    }
    
    # Create reverse mapping: tag -> list of sites
    tag_to_sites = {}
    for site_name, tags in all_tags_by_site.items():
        for tag in tags:
            if tag not in tag_to_sites:
                tag_to_sites[tag] = []
            tag_to_sites[tag].append(site_name)
    
    # Add unique tags with comma-separated site lists
    for tag in sorted(tag_to_sites.keys()):
        sites = sorted(set(tag_to_sites[tag]))  # Remove duplicates and sort
        template[tag] = ", ".join(sites)
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\nTag mapping template saved to: {output_path}")
    print(f"Total unique tags found: {len(tag_to_sites)}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset: extract tags, create selection files with Tags column, and symlink WAV files')
    parser.add_argument('--raw_data', required=True, help='Original raw data directory path')
    parser.add_argument('--output', required=True, help='Output directory for prepared dataset')
    args = parser.parse_args()
    
    raw_data_dir = Path(args.raw_data)
    output_dir = Path(args.output)
    dataset_dir = output_dir / 'prepared_dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing dataset from: {raw_data_dir}")
    print(f"Output directory: {dataset_dir}")
    print(f"This will create:")
    print(f"  - Selection files with Tags column")
    print(f"  - Symbolic links to original WAV files")
    print(f"  - Tag mapping template for scientist review")
    
    # Find all site directories
    site_dirs = [d for d in raw_data_dir.iterdir() 
                 if d.is_dir() and not d.name.startswith('.') 
                 and not d.name.startswith('0')]  # Skip docs/settings folders
    
    all_tags_by_site = {}
    total_wav_files = 0
    total_selection_files = 0
    
    for site_dir in sorted(site_dirs):
        site_tags = process_site(site_dir, dataset_dir)
        if site_tags:
            all_tags_by_site[site_dir.name] = site_tags
    
    # Create tag mapping template
    tag_mapping_path = output_dir / 'tag_mapping_template.json'
    create_tag_mapping_template(all_tags_by_site, tag_mapping_path)
    
    print(f"\n{'='*60}")
    print(f"DATASET PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Sites processed: {len(all_tags_by_site)}")
    print(f"Prepared dataset location: {dataset_dir}")
    print(f"Tag mapping template: {tag_mapping_path}")
    print(f"\nNext steps:")
    print(f"1. Review {tag_mapping_path}")
    print(f"2. Replace location lists with target categories:")
    print(f"   (20Plus, 20Hz, A, B, D, Dswp, Z, Noise, _IGNORE)")
    print(f"3. Save completed mapping as tag_mapping.json")
    print(f"4. Use prepared dataset for preprocessing workflow")


if __name__ == "__main__":
    main() 