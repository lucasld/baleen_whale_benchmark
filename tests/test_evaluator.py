"""
Test the Evaluator class.
"""
import pytest
import numpy as np
import pandas as pd
from whale_benchmark.evaluation.evaluator import Evaluator
from whale_benchmark.models.base import BaseModel


class MockModel(BaseModel):
    """A mock model for testing the Evaluator class."""
    
    def __init__(self):
        super().__init__()
        self.predict_called = False
        
    def predict(self, loader):
        self.predict_called = True
        # Return mock predictions for 3 classes
        return np.array([
            [0.7, 0.2, 0.1],  # Class 0
            [0.1, 0.8, 0.1],  # Class 1
            [0.2, 0.3, 0.5],  # Class 2
            [0.6, 0.3, 0.1],  # Class 0
            [0.3, 0.4, 0.3]   # Class 1
        ])


class MockDataLoader:
    """A mock data loader for testing."""
    
    def __init__(self):
        # Ground truth labels
        self.y_true = np.array([0, 1, 2, 0, 1])
        # Some test metadata
        self.metadata = pd.DataFrame({
            'filename': ['file1.png', 'file2.png', 'file3.png', 'file4.png', 'file5.png'],
            'label': ['class_0', 'class_1', 'class_2', 'class_0', 'class_1']
        })
        self.int2class = ['class_0', 'class_1', 'class_2']


def test_evaluator_init():
    """Test Evaluator initialization."""
    model = MockModel()
    test_loader = MockDataLoader()
    
    evaluator = Evaluator(model, test_loader)
    
    assert evaluator.model == model
    assert evaluator.test_loader == test_loader


def test_evaluator_compute_metrics():
    """Test Evaluator's compute_metrics method."""
    model = MockModel()
    test_loader = MockDataLoader()
    
    evaluator = Evaluator(model, test_loader)
    
    # Test computing metrics
    metrics = evaluator.compute_metrics()
    
    assert model.predict_called
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    assert metrics['accuracy'] > 0


def test_evaluator_confusion_matrix():
    """Test Evaluator's confusion matrix generation."""
    model = MockModel()
    test_loader = MockDataLoader()
    
    evaluator = Evaluator(model, test_loader)
    
    # Test computing confusion matrix
    metrics = evaluator.compute_metrics()
    confusion_matrix = metrics['confusion_matrix']
    
    # Check shape of confusion matrix (3x3 for 3 classes)
    assert confusion_matrix.shape == (3, 3)
    
    # Predicted vs actual should align with our mock data
    assert confusion_matrix[0, 0] == 2  # 2 correct predictions for class 0
    assert confusion_matrix[1, 1] == 2  # 2 correct predictions for class 1
    assert confusion_matrix[2, 2] == 1  # 1 correct prediction for class 2


def test_evaluator_save_results():
    """Test Evaluator's save_results method."""
    model = MockModel()
    test_loader = MockDataLoader()
    
    evaluator = Evaluator(model, test_loader)
    metrics = evaluator.compute_metrics()
    
    # Test saving results (we'll use a temporary path)
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "results")
        evaluator.save_results(metrics, output_path)
        
        # Check that the files were created
        assert os.path.exists(f"{output_path}_metrics.csv")
        assert os.path.exists(f"{output_path}_confusion_matrix.csv")
        assert os.path.exists(f"{output_path}_predictions.csv") 