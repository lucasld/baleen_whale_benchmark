import os
import pandas as pd
import glob
import shutil
import json
import traceback
# from multiprocessing import Process, Queue # REMOVED
from koogu.data.preprocess import from_selection_table_map


def load_tag_mapping(mapping_file=None):
    """Load tag mapping from JSON file or use default if not found."""
    if mapping_file is None:
        # Use default mapping file path in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_file = os.path.join(script_dir, 'tag_mapping.json')
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
                # Remove metadata keys that start with underscore
                return {k: v for k, v in mapping.items() if not k.startswith('_')}
        except Exception as e:
            print(f"Warning: Error loading tag mapping file {mapping_file}: {e}")
            return {}
    else:
        print(f"Warning: Tag mapping file not found at {mapping_file}. Tags will not be mapped.")
        return {}


def apply_tag_mapping(tags_str, tag_mapping):
    """Apply tag mapping to a comma-separated string of tags."""
    if not tag_mapping or pd.isna(tags_str):
        return tags_str
    
    result_tags = []
    for tag in tags_str.split(','):
        tag = tag.strip()
        if tag in tag_mapping:
            mapped_tag = tag_mapping[tag]
            # Convert _IGNORE to IGNORE instead of filtering out
            if mapped_tag == '_IGNORE':
                result_tags.append('IGNORE')
            else:
                result_tags.append(mapped_tag)
        else:
            # Keep unmapped tags as is
            result_tags.append(tag)
    
    return ','.join(result_tags)


def process_site(site_path, base_annotations_dir, base_output_dir, site_config=None, expected_categories=None, tag_mapping=None):
    """Processes annotations and extracts clips for a single site."""
    site_name = os.path.basename(site_path)
    print(f"\n--- Processing Site: {site_name} ---")

    # --- NEW: Get sample rate from site_config ---
    if site_config and site_name in site_config:
        sample_rate = site_config[site_name]
        print(f"Using sample rate {sample_rate} Hz for site {site_name} from config.")
    else:
        print(f"ERROR: Sample rate for site '{site_name}' not found in site_config.json. Skipping.")
        return False
    # --- END NEW ---

    wav_folder = os.path.join(site_path, 'wav')
    if not os.path.exists(wav_folder):
        print(f"WAV folder not found for site {site_name}, skipping.")
        return False # Indicate failure

    # Find all selection table files for the site
    selection_files = glob.glob(os.path.join(site_path, '*.selections.tags.txt'))
    if not selection_files:
        print(f"No selection files (*.selections.tags.txt) found for site {site_name}, skipping.")
        return False # Indicate failure
    
    print(f"Found selection files: {', '.join([os.path.basename(f) for f in selection_files])}")

    # Read and combine all selection tables
    all_annotations_list = []
    for f in selection_files:
        try:
            df = pd.read_csv(f, sep='\t')
            all_annotations_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not all_annotations_list:
        print("No valid annotation dataframes loaded, skipping site.")
        return False # Indicate failure
        
    all_annotations_df = pd.concat(all_annotations_list, ignore_index=True)

    # Preprocess combined annotations
    required_cols = ['Beg File Samp (samples)', 'End File Samp (samples)', 'Begin File', 'Tags',
                     'Low Freq (Hz)', 'High Freq (Hz)', 'Selection', 'View', 'Channel']
    all_annotations_df = all_annotations_df.dropna(subset=required_cols)
    if all_annotations_df.empty:
        print("No valid annotations after dropping NaNs, skipping site.")
        return False # Indicate failure

    all_annotations_df['Beg File Samp (samples)'] = pd.to_numeric(all_annotations_df['Beg File Samp (samples)'])
    all_annotations_df['End File Samp (samples)'] = pd.to_numeric(all_annotations_df['End File Samp (samples)'])

    # Use the sample rate loaded from config for this site
    all_annotations_df['Relative Begin Time (s)'] = all_annotations_df['Beg File Samp (samples)'] / sample_rate
    all_annotations_df['Relative End Time (s)'] = all_annotations_df['End File Samp (samples)'] / sample_rate

    # Prepare site-specific annotation and output directories
    site_annotations_dir = os.path.join(base_annotations_dir, site_name)
    site_output_dir = os.path.join(base_output_dir, site_name)
    os.makedirs(site_annotations_dir, exist_ok=True)
    os.makedirs(site_output_dir, exist_ok=True)

    # Apply tag mapping if provided
    original_tags = set()
    mapped_tags = set()
    
    if tag_mapping:
        print(f"Applying tag mapping from mapping file...")
        # Apply the mapping to the Tags column
        all_annotations_df['Original_Tags'] = all_annotations_df['Tags']  # Preserve original tags
        all_annotations_df['Tags'] = all_annotations_df['Tags'].apply(lambda x: apply_tag_mapping(x, tag_mapping))
        
        # Collect tag information for reporting
        for tags_str in all_annotations_df['Original_Tags']:
            if pd.notna(tags_str):
                for tag in tags_str.split(','):
                    tag = tag.strip()
                    if tag:
                        original_tags.add(tag)
        
        for tags_str in all_annotations_df['Tags']:
            if pd.notna(tags_str):
                for tag in tags_str.split(','):
                    tag = tag.strip()
                    if tag:
                        mapped_tags.add(tag)
        
        # Report tag mapping results
        print(f"Original tags found: {sorted(original_tags)}")
        print(f"Tags after mapping: {sorted(mapped_tags)}")
    
    # Validate class tags against expected categories if provided
    if expected_categories:
        unknown_tags = [tag for tag in mapped_tags if tag not in expected_categories and tag not in ['Noise', 'IGNORE']]
        if unknown_tags:
            # Raise an error as some mapped tags still don't match expected categories
            error_message = (
                f"ERROR: After tag mapping, site '{site_name}' still contains tags not in expected categories: {unknown_tags}. "
                f"Expected: {expected_categories}. Please check annotations, tag mapping, or config."
            )
            raise ValueError(error_message)

    # Validate WAV file references before processing
    unique_wav_files = sorted(list(set(all_annotations_df['Begin File'])))
    actual_wav_files = set(os.listdir(wav_folder)) if os.path.exists(wav_folder) else set()
    
    print(f"Validating WAV file references...")
    print(f"Found {len(actual_wav_files)} actual WAV files in {wav_folder}")
    print(f"Annotations reference {len(unique_wav_files)} unique WAV files")
    
    # To handle case-insensitivity for file extensions only, create a mapping
    # Key: filename with lowercase extension, Value: original filename
    def normalize_extension(filename):
        """Convert only the file extension to lowercase, keep the rest unchanged"""
        name, ext = os.path.splitext(filename)
        return name + ext.lower()
    
    actual_wav_files_map = {normalize_extension(f): f for f in actual_wav_files}
    
    print(f"DEBUG: Sample actual files (first 5): {list(actual_wav_files)[:5]}")
    print(f"DEBUG: Sample referenced files (first 5): {unique_wav_files[:5]}")
    
    # Check for missing WAV files (case-insensitive extensions only)
    missing_wav_files = []
    for wav_file in unique_wav_files:
        normalized_ref = normalize_extension(wav_file)
        if normalized_ref not in actual_wav_files_map:
            missing_wav_files.append(wav_file)
            print(f"DEBUG: Missing file '{wav_file}' -> looking for '{normalized_ref}'")
        else:
            # print(f"DEBUG: Found match for '{wav_file}' -> '{actual_wav_files_map[normalized_ref]}'")
            if len(missing_wav_files) == 0:  # Only show first few matches to avoid spam
                continue
            break
    
    if missing_wav_files:
        print(f"WARNING: {len(missing_wav_files)} WAV files referenced in annotations but not found:")
        for i, missing_file in enumerate(missing_wav_files):
            if i < 10:  # Show first 10
                print(f"  - {missing_file}")
            elif i == 10:
                print(f"  ... and {len(missing_wav_files) - 10} more")
        print(f"Filtering out annotations for missing files and continuing...")
        
        # Remove annotations for missing files
        missing_files_set = set(missing_wav_files)
        original_count = len(all_annotations_df)
        all_annotations_df = all_annotations_df[~all_annotations_df['Begin File'].isin(missing_files_set)]
        filtered_count = len(all_annotations_df)
        print(f"Removed {original_count - filtered_count} annotations for missing files.")
        print(f"Continuing with {filtered_count} annotations.")
        
        if filtered_count == 0:
            print("ERROR: No annotations remaining after filtering missing files.")
            return False
        
        # Update the unique WAV files list
        unique_wav_files = sorted(list(set(all_annotations_df['Begin File'])))
    
    print(f"✅ All {len(unique_wav_files)} referenced WAV files found")
    
    # Create combined annotation files per WAV file
    audio_seltab_list = []

    print(f"Generating combined annotation files in: {site_annotations_dir}")
    for wav_file in unique_wav_files:
        wav_df = all_annotations_df[all_annotations_df['Begin File'] == wav_file].copy()

        # Create the final DataFrame for saving
        new_df = pd.DataFrame()
        new_df['Selection'] = wav_df['Selection']
        new_df['View'] = wav_df['View']
        new_df['Channel'] = wav_df['Channel']
        new_df['Begin Time (s)'] = wav_df['Relative Begin Time (s)']
        new_df['End Time (s)'] = wav_df['Relative End Time (s)']
        new_df['Low Freq (Hz)'] = wav_df['Low Freq (Hz)']
        new_df['High Freq (Hz)'] = wav_df['High Freq (Hz)']
        new_df['Tags'] = wav_df['Tags'] # This now contains the mapped tags

        annotation_filename = f"{os.path.splitext(wav_file)[0]}_annotations.txt"
        annotation_path = os.path.join(site_annotations_dir, annotation_filename)
        new_df.to_csv(annotation_path, sep='\t', index=False, float_format='%.6f')
        audio_seltab_list.append((wav_file, annotation_filename))

    print(f"Generated {len(unique_wav_files)} annotation files.")

    # Define audio settings (adjust if needed)
    audio_settings = {
        'clip_length': 15.0,
        'clip_advance': 2.5,
        'desired_fs': 250
    }

    # Clean previous output for the site before running Koogu
    # print(f"Cleaning Koogu output directory for site: {site_output_dir}") # Optional: Keep or remove this line
    if os.path.exists(site_output_dir):
        shutil.rmtree(site_output_dir)
    os.makedirs(site_output_dir, exist_ok=True)

    # --- Run Koogu processing directly ---
    print(f"Running Koogu processing for site {site_name}...")
    try:
        result = from_selection_table_map(
            audio_settings=audio_settings,
            audio_seltab_list=audio_seltab_list,
            audio_root=wav_folder,
            seltab_root=site_annotations_dir,
            output_root=site_output_dir,
            ignore_zero_annot_files=0,
            negative_class_label='Noise',
            attempt_salvage=True
        )
        print(f"Koogu processing complete for {site_name}.")
        print("Result:", result)
    except Exception as e:
        print(f"ERROR: Koogu process failed for site {site_name} with an exception.")
        print(traceback.format_exc())
        return False
        
    # Validate Koogu output
    print(f"Validating Koogu output...")
    
    # Check for .npz files (the actual processed clips)
    npz_files = glob.glob(os.path.join(site_output_dir, "*.npz"))
    print(f"Generated {len(npz_files)} .npz files")
    
    if len(npz_files) == 0:
        print(f"ERROR: Koogu processing failed - no .npz files generated!")
        print(f"This indicates Koogu could not process any annotations.")
        print(f"Possible causes:")
        print(f"  - Audio file format incompatibility")
        print(f"  - Invalid annotation time ranges")
        print(f"  - Missing or corrupted audio files")
        print(f"  - Koogu library configuration issues")
        return False
    
    # Check for classes_list.json
    koogu_classes_file = os.path.join(site_output_dir, 'classes_list.json')
    if os.path.exists(koogu_classes_file):
        with open(koogu_classes_file, 'r') as f:
            koogu_classes = json.load(f)
        print(f"Koogu detected classes: {koogu_classes}")
    else:
        print(f"Warning: Koogu did not create classes_list.json in {site_output_dir}")
    
    # Report success with details
    total_size = sum(os.path.getsize(f) for f in npz_files if os.path.exists(f))
    print(f"✅ Koogu processing successful!")
    print(f"   Generated {len(npz_files)} clip files ({total_size / 1024 / 1024:.1f} MB)")
    
    return True # Indicate success


# Define a new function to orchestrate the processing for all sites
def run_all_sites_processing(raw_data_dir, annotations_output_dir, koogu_output_dir, config_path=None, tag_mapping_path=None, site_config_path=None):
    """Finds all sites and runs the processing pipeline for each."""
    print(f"Starting site processing.")
    print(f"Raw data source: {raw_data_dir}")
    print(f"Annotations output: {annotations_output_dir}")
    print(f"Koogu output: {koogu_output_dir}")

    # --- NEW: Load site config for sample rates ---
    site_config = None
    if site_config_path and os.path.exists(site_config_path):
        try:
            with open(site_config_path, 'r') as f:
                site_config = json.load(f)
            print(f"Loaded site configuration from {site_config_path}")
        except Exception as e:
            print(f"ERROR: Could not load site config file {site_config_path}: {e}. Aborting.")
            return [] # Cannot proceed without sample rates
    else:
        print(f"ERROR: Site config file not found at {site_config_path}. Aborting.")
        return []
    # --- END NEW ---

    # Load tag mapping
    tag_mapping = load_tag_mapping(tag_mapping_path)
    if tag_mapping:
        print(f"Loaded tag mapping with {len(tag_mapping)} entries")
    else:
        print("No tag mapping loaded. Original tags will be used.")

    # Load expected categories from config if provided
    expected_categories = None
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                expected_categories = config.get('CATEGORIES', [])
                if expected_categories:
                    print(f"Loaded {len(expected_categories)} expected categories from config: {expected_categories}")
        except Exception as e:
            print(f"Error loading config file '{config_path}': {e}. Proceeding without category validation.")

    # Find potential site directories
    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        return list(os.listdir('.'))  # Return all as failed if directory doesn't exist
        
    all_subdirs = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    site_subdirs = [d for d in all_subdirs if not d[0].isdigit()] # Filter out dirs starting with a digit

    if not site_subdirs:
        print(f"No site directories found in raw data path: {raw_data_dir}")
        return []  # No sites to process, so none failed

    print(f"Found potential sites: {', '.join(site_subdirs)}")

    failed_sites = []
    # Process each site
    for site_name in site_subdirs:
        site_path = os.path.join(raw_data_dir, site_name)
        try:
            success = process_site(
                site_path, 
                annotations_output_dir, 
                koogu_output_dir,
                site_config, # Pass the site config dictionary
                expected_categories,
                tag_mapping
            )
            if not success:
                failed_sites.append(site_name)
                print(f"Processing failed for site: {site_name}")
        except ValueError as ve:
             print(ve) # Print the specific validation error
             failed_sites.append(site_name)
             print(f"Validation error in site: {site_name} - continuing with other sites")
        except Exception as e:
            failed_sites.append(site_name)
            print(f"An unexpected error occurred while processing site {site_name}: {e}")

    if not failed_sites:
        print("\nAll sites processed successfully.")
    else:
        print(f"\nProcessing completed with {len(failed_sites)} failed sites.")
        
    return failed_sites


# Keep the __main__ block for potential direct execution, but it won't be used by the workflow script
if __name__ == "__main__":
    # Get inputs for direct execution
    raw_data_input = input('Base path for raw data sites: ') 
    # Use fixed relative paths for annotations/outputs when run directly
    annotations_output_input = os.path.join(os.getcwd(), 'outputs', 'annotations') 
    koogu_output_input = os.path.join(os.getcwd(), 'outputs', 'koogu_output')
    config_path_input = input('Path to config.json (optional, press Enter to skip): ')
    tag_mapping_path_input = input('Path to tag_mapping.json (optional, press Enter to use default): ')
    if not tag_mapping_path_input:
        tag_mapping_path_input = None  # Will use default path

    os.makedirs(annotations_output_input, exist_ok=True)
    os.makedirs(koogu_output_input, exist_ok=True)
    
    run_all_sites_processing(
        raw_data_input, 
        annotations_output_input, 
        koogu_output_input, 
        config_path_input,
        tag_mapping_path_input,
        # Add site_config_path for direct execution
        os.path.join(os.path.dirname(__file__), 'site_config.json')
    )
