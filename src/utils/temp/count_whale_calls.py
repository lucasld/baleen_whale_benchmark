#!/usr/bin/env python3
import os
import json
import pandas as pd
from collections import defaultdict
import glob

def count_whale_calls(dataset_root, tag_mapping_file):
    """
    Count whale calls for each location by counting rows in selection files.
    
    Args:
        dataset_root: Path to the dataset root directory
        tag_mapping_file: Path to the tag mapping JSON file
    
    Returns:
        Dictionary with counts per location and call type
    """
    # Load tag mapping
    with open(tag_mapping_file, 'r') as f:
        tag_mapping = json.load(f)
    
    # Initialize results dictionary
    results = {}
    
    # Get all location folders
    location_folders = [d for d in os.listdir(dataset_root) 
                        if os.path.isdir(os.path.join(dataset_root, d))]
    
    for location in location_folders:
        location_path = os.path.join(dataset_root, location)
        
        # Find all selection files (*.selections.tags.txt)
        selection_files = glob.glob(os.path.join(location_path, "*.selections.tags.txt"))
        
        # Initialize counters for this location
        call_counts = defaultdict(int)
        total_calls = 0
        
        for selection_file in selection_files:
            try:
                # Extract the call type from the filename
                filename = os.path.basename(selection_file)
                
                # Find the tag in the filename
                # The format is typically: LocationYear.Tag.selections.tags.txt
                parts = filename.split('.')
                file_tag = None
                
                # Look for known tags in the filename
                for part in parts:
                    if part in tag_mapping:
                        file_tag = part
                        break
                
                # If we couldn't find a direct match, look for partial matches
                if file_tag is None:
                    for tag in tag_mapping.keys():
                        if tag in filename:
                            file_tag = tag
                            break
                
                # If we still couldn't find a tag, use the second part as default
                if file_tag is None and len(parts) > 1:
                    file_tag = parts[1]
                
                # Map the tag if it exists in our mapping
                mapped_tag = tag_mapping.get(file_tag, file_tag) if file_tag else "Unknown"
                
                # Skip files with tags marked as _IGNORE
                if mapped_tag == "_IGNORE":
                    continue
                
                # Read the tab-separated file and count rows (excluding header)
                df = pd.read_csv(selection_file, sep='\t')
                num_calls = len(df)
                
                # Add to our counts
                call_counts[mapped_tag] += num_calls
                total_calls += num_calls
                
                print(f"Processed {filename}: {num_calls} calls of type {mapped_tag}")
                
            except Exception as e:
                print(f"Error processing {selection_file}: {str(e)}")
        
        # Store results for this location
        call_counts["Total"] = total_calls
        results[location] = dict(call_counts)
    
    return results

def main():
    # Adjust these paths as needed
    dataset_root = "../datasets/raw_dataset"
    tag_mapping_file = "../custom_preprocessing/tag_mapping.json"
    
    # Count calls
    results = count_whale_calls(dataset_root, tag_mapping_file)
    
    # Print results
    print("\nWhale Call Counts by Location:")
    print("============================")
    
    for location, counts in results.items():
        print(f"\n{location}:")
        print("-" * len(location))
        
        # Print counts for each call type
        for call_type, count in sorted(counts.items()):
            if call_type != "Total":
                print(f"  {call_type}: {count}")
        
        # Print total
        print(f"  Total: {counts['Total']}")
    
    # Save results to JSON
    output_file = "whale_call_counts.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 