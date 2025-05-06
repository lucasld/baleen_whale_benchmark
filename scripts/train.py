#!/usr/bin/env python
"""
Script for training a model.

Usage:
    python scripts/train.py --config configs/default.yaml

This script allows you to train a model using a specific configuration.
"""
import argparse
import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
import pathlib
from typing import Dict, Any

from whale_benchmark.data import SpectrogramDataset
from whale_benchmark.models import CNNModel
from whale_benchmark.training import Trainer
from whale_benchmark.utils import get_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to config file"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "train.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def prepare_dataset(config: Dict[str, Any]) -> tuple:
    """
    Prepare the dataset.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        tuple: (dataset, paths_df)
    """
    data_config = config.get("data", {})
    
    # Initialize dataset
    dataset = SpectrogramDataset(
        data_dir=data_config.get("dir", "data"),
        categories=data_config.get("categories", []),
        join_cat=data_config.get("join_cat", {}),
        locations=data_config.get("locations", []),
        corrected=data_config.get("use_corrected", True),
        samples_per_class=data_config.get("samples_per_class", "all")
    )
    
    # Prepare splits
    if data_config.get("blocked_location"):
        # Prepare dataset with blocked location
        paths_df = dataset.prepare_blocked_dataset(
            blocked_location=data_config.get("blocked_location"),
            valid_size=data_config.get("validation_split", 0.3),
            noise_ratio=data_config.get("noise_ratio", 0.5),
            noise_ratio_test=data_config.get("noise_ratio_test", 0.5)
        )
    else:
        # Prepare dataset with random splits
        paths_df = dataset.prepare_all_dataset(
            test_size=data_config.get("test_split", 0.2),
            valid_size=data_config.get("validation_split", 0.3),
            noise_ratio=data_config.get("noise_ratio", 0.5)
        )
    
    return dataset, paths_df


def create_data_loaders(dataset, paths_df):
    """
    Create data loaders.
    
    Args:
        dataset: Dataset instance
        paths_df: DataFrame with paths
        
    Returns:
        dict: Dictionary with data loaders
    """
    # Load data
    x_train, y_train = dataset.load_set_from_df(paths_df, 'train')
    x_valid, y_valid = dataset.load_set_from_df(paths_df, 'valid')
    
    # Create loaders
    class TrainLoader:
        def __init__(self, x, y, x_valid=None, y_valid=None, n_classes=None, int2class=None):
            self.x = x
            self.y_true = y
            self.x_valid = x_valid
            self.y_valid = y_valid
            self.n_classes = n_classes or len(np.unique(y))
            self.int2class = int2class
    
    train_loader = TrainLoader(
        x_train, y_train, x_valid, y_valid, 
        dataset.n_classes, dataset.int2class
    )
    
    loaders = {
        'train': train_loader
    }
    
    return loaders


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    logging.info(f"Training with config: {args.config}")
    
    # Prepare dataset
    dataset, paths_df = prepare_dataset(config)
    logging.info(f"Dataset prepared with {len(paths_df)} samples")
    logging.info(f"Classes: {dataset.int2class}")
    
    # Create data loaders
    loaders = create_data_loaders(dataset, paths_df)
    
    # Create model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "cnn")
    model_kwargs = model_config.get("params", {})
    
    # Add categories to model kwargs
    model_kwargs["categories"] = dataset.int2class
    
    # Create model from registry
    model = get_model(model_name, **model_kwargs)
    logging.info(f"Created model: {model_name}")
    
    # Create trainer
    training_config = config.get("training", {})
    trainer = Trainer(
        model=model,
        loaders=loaders,
        cfg=training_config
    )
    
    # Train model
    logging.info("Starting training...")
    history = trainer.train()
    logging.info("Training completed")
    
    # Save paths_df for reproducibility
    save_path = pathlib.Path(model.log_path)
    paths_df.to_csv(save_path / "data_splits.csv", index=False)
    
    # Print final metrics
    final_metrics = {k: v[-1] for k, v in history.items() 
                    if isinstance(v, list) and len(v) > 0}
    logging.info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
