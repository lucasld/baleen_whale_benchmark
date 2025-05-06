"""
Test the SpectrogramDataset class.
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from PIL import Image
from whale_benchmark.data.spectrogram_dataset import SpectrogramDataset


class TestSpectrogramDataset:
    """Tests for the SpectrogramDataset class."""
    
    @pytest.fixture
    def mock_data_dir(self):
        """Create temporary directory with mock spectrogram files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create class directories
            class_names = ["Blue", "Fin", "Humpback", "Noise"]
            for class_name in class_names:
                os.makedirs(os.path.join(tmp_dir, class_name), exist_ok=True)
                
                # Create sample spectrogram images for each class
                for i in range(3):  # 3 samples per class
                    img = Image.new('L', (30, 90), color=100)  # 30x90 grayscale image
                    img_path = os.path.join(tmp_dir, class_name, f"{class_name}_{i}.png")
                    img.save(img_path)
            
            yield tmp_dir
    
    def test_init(self, mock_data_dir):
        """Test dataset initialization."""
        categories = ["Blue", "Fin", "Humpback", "Noise"]
        locations = ["Location1", "Location2"]
        join_cat = {
            "Whales": ["Blue", "Fin", "Humpback"]
        }
        
        dataset = SpectrogramDataset(
            data_dir=mock_data_dir,
            categories=categories,
            join_cat=join_cat,
            locations=locations,
            corrected=True,
            samples_per_class=250
        )
        
        # Check if categories are correctly processed
        assert len(dataset.int2class) == 2  # "Whales" and "Noise"
        assert "Whales" in dataset.classes2int
        assert "Noise" in dataset.classes2int
        assert dataset.n_classes == 2
        
        # Check mapping
        assert dataset.map_join["Blue"] == "Whales"
        assert dataset.map_join["Fin"] == "Whales"
        assert dataset.map_join["Humpback"] == "Whales"
        assert dataset.map_join["Noise"] == "Noise"
    
    def test_select_files(self, mock_data_dir):
        """Test file selection."""
        categories = ["Blue", "Fin", "Humpback", "Noise"]
        locations = ["Location1", "Location2"]
        join_cat = {}  # No joining, treat each class separately
        
        dataset = SpectrogramDataset(
            data_dir=mock_data_dir,
            categories=categories,
            join_cat=join_cat,
            locations=locations,
            corrected=True,
            samples_per_class="all"
        )
        
        # Test file selection for a category
        for category in categories:
            files = dataset.select_files_category(
                category=category, 
                samples_to_load="all"
            )
            assert len(files) == 3  # 3 samples per class
            
            # Check that all files belong to the correct category
            for file_path in files:
                assert category in file_path
    
    def test_load_images(self, mock_data_dir):
        """Test loading images."""
        categories = ["Blue", "Fin", "Humpback", "Noise"]
        locations = ["Location1", "Location2"]
        join_cat = {}
        
        dataset = SpectrogramDataset(
            data_dir=mock_data_dir,
            categories=categories,
            join_cat=join_cat,
            locations=locations,
            corrected=True,
            samples_per_class="all"
        )
        
        # Get all files for Blue class
        files = dataset.select_files_category("Blue", "all")
        
        # Load images
        images, labels = dataset.load_from_file_list(files)
        
        # Check shapes and types
        assert len(images) == len(files)
        assert len(labels) == len(files)
        assert images.shape[1:] == (30, 90, 1)  # [batch, width, height, channels]
        assert np.max(images) <= 1.0  # Check normalization
        
        # All labels should be the same (Blue class)
        assert np.all(labels == dataset.classes2int["Blue"])
    
    def test_train_val_test_split(self, mock_data_dir):
        """Test dataset splitting."""
        categories = ["Blue", "Fin", "Humpback", "Noise"]
        locations = ["Location1", "Location2"]
        join_cat = {}
        
        dataset = SpectrogramDataset(
            data_dir=mock_data_dir,
            categories=categories,
            join_cat=join_cat,
            locations=locations,
            corrected=True,
            samples_per_class="all"
        )
        
        # Prepare dataset with specific splits
        paths_df = dataset.prepare_all_dataset(
            test_size=0.2,
            valid_size=0.25,  # 25% of remaining 80% = 20% of total
            noise_ratio=0.3
        )
        
        # Check that all partitions exist
        assert set(paths_df['set'].unique()) == {'train', 'valid', 'test'}
        
        # Load partitions
        train_images, train_labels = dataset.load_set_from_df(paths_df, 'train')
        valid_images, valid_labels = dataset.load_set_from_df(paths_df, 'valid')
        test_images, test_labels = dataset.load_set_from_df(paths_df, 'test')
        
        # Check that we have data in each partition
        assert len(train_images) > 0
        assert len(valid_images) > 0
        assert len(test_images) > 0
        
        # Check shapes
        assert train_images.shape[1:] == (30, 90, 1)
        assert valid_images.shape[1:] == (30, 90, 1)
        assert test_images.shape[1:] == (30, 90, 1) 