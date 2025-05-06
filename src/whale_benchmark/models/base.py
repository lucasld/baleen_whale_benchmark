"""
Base model interface for all whale call classification models.
"""
import abc
import os
import pathlib
import pickle


class BaseModel(abc.ABC):
    """
    Abstract base class for all models in the whale benchmark.
    
    All models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self):
        """Initialize the model."""
        pass
    
    @classmethod
    def from_config(cls, cfg):
        """
        Instantiate model from configuration.
        
        Args:
            cfg (dict): Model configuration
            
        Returns:
            BaseModel: Instantiated model
        """
        return cls(**cfg)
    
    @abc.abstractmethod
    def fit(self, train_loader):
        """
        Train the model on the provided data.
        
        Args:
            train_loader: DataLoader with training data
            
        Returns:
            dict: Training history with metrics
        """
        pass
    
    @abc.abstractmethod
    def predict(self, loader):
        """
        Generate predictions for the provided data.
        
        Args:
            loader: DataLoader with data to predict on
            
        Returns:
            numpy.ndarray: Model predictions
        """
        pass
    
    def evaluate(self, loader):
        """
        Evaluate model performance on the provided data.
        
        Args:
            loader: DataLoader with data to evaluate on
            
        Returns:
            dict: Evaluation metrics
        """
        # Default implementation uses predict and calculates metrics
        from sklearn.metrics import accuracy_score
        import numpy as np
        
        # Get predictions and ground truth
        y_pred = self.predict(loader)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = loader.y_true
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_classes)
        }
        
        return metrics
    
    def save(self, path):
        """
        Save model to disk.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: True if successful
        """
        # Create directory if it doesn't exist
        path = pathlib.Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        # Save model using pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        return True
    
    @classmethod
    def load(cls, path):
        """
        Load model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            BaseModel: Loaded model
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model
