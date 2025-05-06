"""
Evaluation utilities for whale call classification models.
"""
import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from whale_benchmark.utils.metrics import calculate_metrics, calculate_confusion_matrix
from typing import Dict, Any, Optional


class Evaluator:
    """
    Evaluator class for evaluating model performance.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with test data
    """
    
    def __init__(self, model, test_loader):
        """Initialize the evaluator."""
        self.model = model
        self.test_loader = test_loader
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics
        """
        # Get predictions and true values
        y_pred = self.model.predict(self.test_loader)
        y_true = self.test_loader.y_true
        
        # Compute metrics
        metrics = calculate_metrics(y_true, y_pred, self.test_loader.int2class)
        
        return metrics
    
    def evaluate_in_batches(self):
        """
        Evaluate model on test data in batches.
        
        Useful for large datasets that don't fit in memory.
        
        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics
        """
        all_y_pred = []
        all_y_true = []
        all_files = []
        
        # Process batches
        for x_batch, y_batch, _, files_batch in self.test_loader.batch_load_from_df(
            self.test_loader, data_split='test'
        ):
            y_pred_batch = self.model.predict(x_batch)
            all_y_pred.append(y_pred_batch)
            all_y_true.append(y_batch)
            all_files.extend(files_batch)
        
        # Concatenate results
        y_pred = np.concatenate(all_y_pred)
        y_true = np.concatenate(all_y_true)
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame(y_pred, columns=self.test_loader.int2class)
        pred_df['file'] = all_files
        
        # Compute metrics
        metrics = calculate_metrics(y_true, y_pred, self.test_loader.int2class)
        metrics['predictions_df'] = pred_df
        
        return metrics
    
    def plot_confusion_matrix(self, confusion_matrix, save_path=None):
        """
        Plot a confusion matrix.
        
        Args:
            confusion_matrix (pd.DataFrame): Confusion matrix
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_results(self, metrics, output_path):
        """
        Save evaluation results to disk.
        
        Args:
            metrics (Dict[str, Any]): Dictionary of evaluation metrics
            output_path (str): Base path for output files
        """
        output_path = pathlib.Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() 
                                 if k not in ['confusion_matrix', 'predictions_df', 
                                              'precision', 'recall', 'f1', 'support']})
        metrics_df.to_csv(f"{output_path}_metrics.csv", index=False)
        
        # Save class-specific metrics
        if 'precision' in metrics:
            class_metrics = pd.DataFrame({
                'class': list(metrics['precision'].keys()),
                'precision': list(metrics['precision'].values()),
                'recall': list(metrics['recall'].values()),
                'f1': list(metrics['f1'].values()),
                'support': list(metrics['support'].values())
            })
            class_metrics.to_csv(f"{output_path}_class_metrics.csv", index=False)
        
        # Save confusion matrix
        if 'confusion_matrix' in metrics:
            metrics['confusion_matrix'].to_csv(f"{output_path}_confusion_matrix.csv")
            # Plot and save confusion matrix
            self.plot_confusion_matrix(
                metrics['confusion_matrix'], 
                save_path=f"{output_path}_confusion_matrix.png"
            )
        
        # Save predictions
        if 'predictions_df' in metrics:
            metrics['predictions_df'].to_csv(f"{output_path}_predictions.csv", index=False)
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred, categories):
        """
        Get confusion matrix from predictions.
        
        Args:
            y_true (np.ndarray): Ground truth labels
            y_pred (np.ndarray): Predicted probabilities
            categories (List[str]): List of category names
            
        Returns:
            pd.DataFrame: Confusion matrix as DataFrame
        """
        return calculate_confusion_matrix(y_true, y_pred, categories)
