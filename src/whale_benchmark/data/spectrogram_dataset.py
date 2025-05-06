"""
Dataset class for loading and preprocessing spectrograms for whale call classification.
"""
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Dict, List, Tuple, Union, Optional

# Constants from the legacy code
IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1
SHUFFLE_SEED = 42


class SpectrogramDataset:
    """
    Dataset class for loading and preprocessing spectrograms.
    
    Args:
        data_dir (str): Directory containing the spectrogram images
        categories (List[str]): List of category names
        join_cat (Dict[str, List[str]]): Dictionary mapping joined categories to their constituent categories
        locations (List[str]): List of recording locations
        corrected (bool): Whether to use corrected samples
        samples_per_class (Union[str, int]): Number of samples per class, or 'all' for all samples
    """
    
    def __init__(self, data_dir, categories, join_cat, locations, corrected, samples_per_class='all'):
        """Initialize the dataset."""
        self.locations = locations
        self.data_dir = data_dir
        self.categories = categories
        self.corrected = corrected
        self.join_cat = join_cat
        self.samples_per_class = samples_per_class
        
        # Create mappings for category names and indices
        self.map_join = {}
        self.classes2int = {}
        self.int2class = []
        
        # Process joined categories
        for join_class_name, classes_list in join_cat.items():
            self.classes2int[join_class_name] = len(self.classes2int.keys())
            self.int2class.append(join_class_name)
            for class_name in classes_list:
                self.map_join[class_name] = join_class_name
        
        # Process individual categories
        for cat_name in self.categories:
            if cat_name not in self.map_join.keys():
                self.map_join[cat_name] = cat_name
                self.classes2int[cat_name] = len(self.classes2int)
                self.int2class.append(cat_name)
        
        self.n_classes = len(self.classes2int)
        print('These are the classes: ', self.classes2int)
        print('Which are formed by doing: ', self.map_join)
    
    def how_many_samples(self):
        """
        Return the average number of samples per class if all samples are selected.
        
        Returns:
            float: Average number of samples per class
        """
        samples = 0
        for category in self.categories:
            # Skip Noise class for this calculation
            if category != 'Noise':
                path = os.path.join(self.data_dir, category)
                # Count number of files
                if os.path.exists(path):
                    samples += len(pd.Series(os.listdir(path)))
        
        # Return average per class (excluding Noise)
        non_noise_classes = len([c for c in self.int2class if c != 'Noise'])
        return samples / non_noise_classes if non_noise_classes > 0 else 0
    
    @staticmethod
    def reshape_images(images):
        """
        Reshape and normalize images to the required format.
        
        Args:
            images (List): List of images
            
        Returns:
            np.ndarray: Reshaped and normalized images
        """
        X = np.array(images).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS)
        x = X / 255.0
        return x
    
    @staticmethod
    def join_paths_to_df(paths_train, paths_valid, paths_test):
        """
        Join paths from different splits into a single DataFrame.
        
        Args:
            paths_train (List[str]): Paths for training set
            paths_valid (List[str]): Paths for validation set
            paths_test (List[str]): Paths for test set
            
        Returns:
            pd.DataFrame: DataFrame with paths and split information
        """
        train_df = pd.DataFrame({'path': paths_train})
        valid_df = pd.DataFrame({'path': paths_valid})
        test_df = pd.DataFrame({'path': paths_test})
        
        train_df = train_df.assign(set='train')
        valid_df = valid_df.assign(set='valid')
        test_df = test_df.assign(set='test')
        
        paths_df = pd.concat([train_df, valid_df, test_df])
        return paths_df
    
    def prepare_all_dataset(self, test_size, valid_size, noise_ratio):
        """
        Prepare the dataset with random splits.
        
        Args:
            test_size (float): Proportion of data to use for testing
            valid_size (float): Proportion of remaining data to use for validation
            noise_ratio (float): Ratio of noise samples to include
            
        Returns:
            pd.DataFrame: DataFrame with file paths and split information
        """
        paths_list = self.select_data(locations_to_exclude=None, noise_ratio=noise_ratio)
        x, y = self.load_from_file_list(file_list=paths_list)
        
        # Split into train/val/test
        x_model, x_test, y_model, y_test, paths_model, paths_test = train_test_split(
            x, y, paths_list, test_size=test_size, shuffle=True, random_state=SHUFFLE_SEED
        )
        
        x_train, x_valid, y_train, y_valid, paths_train, paths_valid = train_test_split(
            x_model, y_model, paths_model, test_size=valid_size, shuffle=True, random_state=SHUFFLE_SEED
        )
        
        paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)
        return paths_df
    
    def prepare_blocked_dataset(self, blocked_location, valid_size, noise_ratio, noise_ratio_test):
        """
        Prepare the dataset with a blocked location for testing.
        
        Args:
            blocked_location (str): Location to use for testing
            valid_size (float): Proportion of model data to use for validation
            noise_ratio (float): Ratio of noise samples to include in train/val
            noise_ratio_test (float): Ratio of noise samples to include in test
            
        Returns:
            pd.DataFrame: DataFrame with file paths and split information
        """
        selected_locs = list(set(self.locations) - {blocked_location})
        
        print('Selecting model data...')
        paths_list_model = self.select_data(locations_to_exclude=[blocked_location], noise_ratio=noise_ratio)
        y = self.read_labels_from_file_list(file_list=paths_list_model)
        
        paths_train, paths_valid = train_test_split(
            paths_list_model, stratify=y, test_size=valid_size, shuffle=True, random_state=SHUFFLE_SEED
        )
        
        print('Selecting test data...')
        paths_test = self.select_data(locations_to_exclude=selected_locs, noise_ratio=noise_ratio_test)
        
        paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)
        return paths_df
    
    def select_files_category(self, category, samples_to_load, locations_to_exclude=None, samples_to_exclude=None):
        """
        Select files for a specific category.
        
        Args:
            category (str): Category to select files for
            samples_to_load (Union[str, int]): Number of samples to load or 'all'
            locations_to_exclude (Optional[List[str]]): Locations to exclude
            samples_to_exclude (Optional[List[str]]): Sample paths to exclude
            
        Returns:
            np.ndarray: Array of file paths
        """
        locations_to_exclude = locations_to_exclude or []
        samples_to_exclude = samples_to_exclude or []
        
        # Get subcategories that map to this category
        subcats = dict((key, value) for (key, value) in self.map_join.items() if value == category)
        
        # Get number of samples available in each subcategory
        num_samples = []
        for subcat in subcats:
            path = os.path.join(self.data_dir, subcat)
            if os.path.exists(path):
                num_samples.append(len(os.listdir(path)))
            else:
                num_samples.append(0)
        num_samples = np.array(num_samples)
        
        # Determine how many samples to load per subcategory
        if samples_to_load != 'all':
            n_subcategories = len(subcats)
            samples_per_subcategory = round(samples_to_load / n_subcategories)
            
            # Check if we have enough samples
            if num_samples.sum() < samples_to_load:
                raise Exception('Samples per class too high for available dataset, please choose a lower number')
            
            # If we have enough samples in each subcategory
            elif all(num_samples >= samples_per_subcategory):
                samples_to_load_per_subcat = np.repeat(samples_per_subcategory, n_subcategories)
            
            # Handle case where some subcategories have fewer samples
            else:
                remaining_samples = samples_to_load
                samples_to_load_per_subcat = np.zeros(n_subcategories, dtype=int)
                
                # First, allocate what we can to each subcategory
                for i, subcat_samples in enumerate(num_samples):
                    if subcat_samples < samples_per_subcategory:
                        samples_to_load_per_subcat[i] = subcat_samples
                        remaining_samples -= subcat_samples
                    else:
                        samples_to_load_per_subcat[i] = samples_per_subcategory
                        remaining_samples -= samples_per_subcategory
                
                # Distribute remaining samples
                while remaining_samples > 0:
                    for i in range(n_subcategories):
                        if samples_to_load_per_subcat[i] < num_samples[i]:
                            samples_to_load_per_subcat[i] += 1
                            remaining_samples -= 1
                            if remaining_samples == 0:
                                break
        
        # Load all available samples if samples_to_load == 'all'
        else:
            samples_to_load_per_subcat = num_samples
        
        # Load files from each subcategory
        all_files = []
        for i, (subcat, samples_to_load_i) in enumerate(zip(subcats.keys(), samples_to_load_per_subcat)):
            subcat_path = os.path.join(self.data_dir, subcat)
            if not os.path.exists(subcat_path):
                continue
                
            files = pd.Series(os.listdir(subcat_path))
            files = [os.path.join(subcat_path, f) for f in files]
            
            # Filter by location if needed
            if locations_to_exclude:
                filtered_files = []
                for file in files:
                    exclude = False
                    for location in locations_to_exclude:
                        if location in file:
                            exclude = True
                            break
                    if not exclude:
                        filtered_files.append(file)
                files = filtered_files
            
            # Exclude specific samples if needed
            if samples_to_exclude:
                files = [f for f in files if f not in samples_to_exclude]
            
            # Select required number of samples
            if samples_to_load_i < len(files):
                files = files[:int(samples_to_load_i)]
            
            all_files.extend(files)
        
        return np.array(all_files)
    
    def select_data(self, noise_ratio, locations_to_exclude=None):
        """
        Select data with the specified noise ratio.
        
        Args:
            noise_ratio (float): Ratio of noise samples to include
            locations_to_exclude (Optional[List[str]]): Locations to exclude
            
        Returns:
            np.ndarray: Array of file paths
        """
        # Select files for each category
        all_files = []
        for category in self.categories:
            if category != 'Noise':
                files = self.select_files_category(
                    category=category,
                    samples_to_load=self.samples_per_class,
                    locations_to_exclude=locations_to_exclude
                )
                all_files.extend(files)
        
        # Add noise samples
        if noise_ratio > 0:
            n_files = len(all_files)
            n_noise = int(n_files / (1 - noise_ratio) * noise_ratio)
            noise_files = self.select_files_category(
                category='Noise',
                samples_to_load=n_noise,
                locations_to_exclude=locations_to_exclude
            )
            all_files.extend(noise_files)
        
        all_files = np.array(all_files)
        
        # Shuffle the data
        np.random.seed(SHUFFLE_SEED)
        np.random.shuffle(all_files)
        
        return all_files
    
    def read_labels_from_file_list(self, file_list):
        """
        Read labels from a list of file paths.
        
        Args:
            file_list (np.ndarray): Array of file paths
            
        Returns:
            np.ndarray: Array of labels
        """
        y = []
        for file_path in file_list:
            # Extract category from path
            category = os.path.dirname(file_path).split(os.sep)[-1]
            # Map to main category
            main_category = self.map_join[category]
            # Convert to label index
            y.append(self.classes2int[main_category])
        
        return np.array(y)
    
    def load_from_file_list(self, file_list):
        """
        Load images and labels from a list of file paths.
        
        Args:
            file_list (np.ndarray): Array of file paths
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (images, labels)
        """
        images = []
        for file_path in file_list:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
        
        x = self.reshape_images(images)
        y = self.read_labels_from_file_list(file_list)
        
        return x, y
    
    def load_set_from_df(self, paths_df, partition):
        """
        Load a specific partition of the dataset.
        
        Args:
            paths_df (pd.DataFrame): DataFrame with paths and partition information
            partition (str): Which partition to load ('train', 'valid', or 'test')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (images, labels)
        """
        file_list = paths_df.loc[paths_df['set'] == partition, 'path'].values
        return self.load_from_file_list(file_list)
    
    def select_more_noise(self, paths_df, new_noise_ratio, partition):
        """
        Add more noise samples to achieve a new noise ratio.
        
        Args:
            paths_df (pd.DataFrame): DataFrame with paths and partition information
            new_noise_ratio (Union[str, float]): New noise ratio or 'all'
            partition (str): Which partition to modify
            
        Returns:
            pd.DataFrame: Updated DataFrame
        """
        # If 'all', load all available noise samples
        if new_noise_ratio == 'all':
            noise_files = self.select_files_category(category='Noise', samples_to_load='all')
            # Convert to DataFrame
            noise_df = pd.DataFrame({'path': noise_files})
            noise_df = noise_df.assign(set=partition)
            # Remove any existing noise samples
            non_noise_df = paths_df[~paths_df['path'].str.contains('Noise')]
            # Combine
            return pd.concat([non_noise_df, noise_df])
        
        # Get current files for the partition
        partition_files = paths_df.loc[paths_df['set'] == partition, 'path'].values
        
        # Count non-noise files
        non_noise_files = [f for f in partition_files if 'Noise' not in f]
        n_non_noise = len(non_noise_files)
        
        # Calculate how many noise files we need
        n_noise_needed = int(n_non_noise / (1 - new_noise_ratio) * new_noise_ratio)
        
        # Count current noise files
        noise_files = [f for f in partition_files if 'Noise' in f]
        n_noise_current = len(noise_files)
        
        # If we need more noise files
        if n_noise_needed > n_noise_current:
            n_noise_to_add = n_noise_needed - n_noise_current
            # Exclude existing noise files
            new_noise_files = self.select_files_category(
                category='Noise',
                samples_to_load=n_noise_to_add,
                samples_to_exclude=noise_files
            )
            
            # Add to DataFrame
            new_noise_df = pd.DataFrame({'path': new_noise_files})
            new_noise_df = new_noise_df.assign(set=partition)
            return pd.concat([paths_df, new_noise_df])
        
        return paths_df
    
    def batch_load_from_df(self, data_split_df, data_split='test'):
        """
        Load data in batches from a DataFrame.
        
        Args:
            data_split_df (pd.DataFrame): DataFrame with paths and partition information
            data_split (str): Which partition to load
            
        Yields:
            Tuple: (x_batch, y_batch, dataset, images_for_test)
        """
        file_paths = data_split_df.loc[data_split_df['set'] == data_split, 'path'].values
        
        # Process in batches of 1000
        batch_size = 1000
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            x, y = self.load_from_file_list(batch_files)
            yield x, y, self, batch_files
