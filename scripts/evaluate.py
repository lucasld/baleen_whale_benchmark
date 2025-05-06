#!/usr/bin/env python
"""
Script for evaluating a trained model.

Usage:
    python scripts/evaluate.py --model-path models/cnn_model/model --data-splits models/cnn_model/data_splits.csv

This script allows you to evaluate a trained model on test data.
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
from whale_benchmark.evaluation import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Path to the saved model"
    )
    parser.add_argument(
        "--data-splits", 
        type=str, 
        required=True, 
        help="Path to the data splits CSV file"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        help="Path to the data directory (overrides the one in data_splits)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results", 
        help="Directory to save evaluation results"
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Set up logging."""
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "evaluate.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def extract_dataset_info(data_splits_path: str) -> Dict[str, Any]:
    """
    Extract dataset information from the data splits file.
    
    Args:
        data_splits_path (str): Path to the data splits CSV file
        
    Returns:
        Dict[str, Any]: Dataset information
    """
    # Load the paths DataFrame
    paths_df = pd.read_csv(data_splits_path)
    
    # Extract first path to determine data structure
    first_path = paths_df["path"].iloc[0]
    path_parts = first_path.split(os.sep)
    
    # Extract data directory and categories
    data_dir = os.sep.join(path_parts[:-2])  # Remove filename and category
    
    # Extract categories
    categories = set()
    for path in paths_df["path"]:
        category = path.split(os.sep)[-2]  # Category is the second-to-last part of the path
        categories.add(category)
    
    return {
        "data_dir": data_dir,
        "categories": list(categories),
        "paths_df": paths_df
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info(f"Evaluating model: {args.model_path}")
    
    # Extract dataset info from data splits
    dataset_info = extract_dataset_info(args.data_splits)
    paths_df = dataset_info["paths_df"]
    
    # Override data directory if provided
    data_dir = args.data_dir or dataset_info["data_dir"]
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model type from path
    model_path = pathlib.Path(args.model_path)
    model_name = model_path.parent.name
    
    # Load the trained model
    logging.info(f"Loading model from {args.model_path}")
    model = CNNModel.load(args.model_path)
    
    # Initialize dataset with the same parameters used for training
    dataset = SpectrogramDataset(
        data_dir=data_dir,
        categories=dataset_info["categories"],
        join_cat={},  # We don't know the join_cat, so use default
        locations=[],  # We don't know the locations, so use default
        corrected=True,
        samples_per_class="all"
    )
    
    # Load test data
    logging.info("Loading test data")
    x_test, y_test = dataset.load_set_from_df(paths_df, 'test')
    
    # Create test loader
    class TestLoader:
        def __init__(self, x, y, class_names):
            self.x = x
            self.y_true = y
            self.int2class = class_names
    
    test_loader = TestLoader(x_test, y_test, dataset.int2class)
    
    # Evaluate model
    logging.info("Evaluating model")
    evaluator = Evaluator(model, test_loader)
    metrics = evaluator.compute_metrics()
    
    # Save results
    output_path = os.path.join(args.output_dir, model_name)
    evaluator.save_results(metrics, output_path)
    logging.info(f"Results saved to {output_path}")
    
    # Print key metrics
    logging.info(f"Accuracy: {metrics['accuracy']:.3f}")
    if "imbalanced_metric" in metrics:
        logging.info(f"Imbalanced Metric: {metrics['imbalanced_metric']:.3f}")
    if "call_avg_tpr" in metrics:
        logging.info(f"Call Average TPR: {metrics['call_avg_tpr']:.3f}")
    if "noise_misclass_rate" in metrics:
        logging.info(f"Noise Misclassification Rate: {metrics['noise_misclass_rate']:.3f}")


if __name__ == "__main__":
    main()
