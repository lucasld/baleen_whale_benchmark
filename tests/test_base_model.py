"""
Test the BaseModel class.
"""
import pytest
import torch
import numpy as np
from whale_benchmark.models.base import BaseModel


class MockModel(BaseModel):
    """A mock model for testing the BaseModel class."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.is_fit = False
        self.predict_called = False
        
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
    
    def fit(self, train_loader):
        self.is_fit = True
        return {"accuracy": 0.9, "loss": 0.1}
    
    def predict(self, loader):
        self.predict_called = True
        # Return mock predictions
        return np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    
    def save(self, path):
        # Just pretend to save
        return True
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.is_fit = True
        return model


def test_base_model_interface():
    """Test that the BaseModel interface works correctly."""
    model = MockModel()
    
    # Test fit method
    train_loader = ["mock_loader"]
    history = model.fit(train_loader)
    assert model.is_fit
    assert "accuracy" in history
    assert "loss" in history
    
    # Test predict method
    loader = ["mock_loader"]
    predictions = model.predict(loader)
    assert model.predict_called
    assert predictions.shape == (2, 3)
    
    # Test evaluate method
    metrics = model.evaluate(loader)
    assert isinstance(metrics, dict)
    
    # Test save and load methods
    assert model.save("dummy_path")
    loaded_model = MockModel.load("dummy_path")
    assert loaded_model.is_fit

    # Test from_config method
    config = {"param1": "value1"}
    model_from_config = MockModel.from_config(config)
    assert isinstance(model_from_config, MockModel) 