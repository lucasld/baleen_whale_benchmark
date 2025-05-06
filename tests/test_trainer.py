"""
Test the Trainer class.
"""
import pytest
import numpy as np
from whale_benchmark.training.trainer import Trainer
from whale_benchmark.models.base import BaseModel
from whale_benchmark.training.callbacks import EarlyStopping, ModelCheckpoint


class MockModel(BaseModel):
    """A mock model for testing the Trainer class."""
    
    def __init__(self):
        super().__init__()
        self.epochs_trained = 0
        self.train_called = False
        self.predict_called = False
        
    def fit(self, train_loader):
        self.train_called = True
        self.epochs_trained += 1
        # Return mock metrics that improve with each epoch
        return {
            "loss": 1.0 - (0.1 * self.epochs_trained),
            "accuracy": 0.5 + (0.05 * self.epochs_trained),
            "val_loss": 1.1 - (0.1 * self.epochs_trained),
            "val_accuracy": 0.45 + (0.05 * self.epochs_trained)
        }
    
    def predict(self, loader):
        self.predict_called = True
        return np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])


class MockDataLoader:
    """A mock data loader for testing."""
    
    def __init__(self, data):
        self.data = data


def test_trainer_init():
    """Test Trainer initialization."""
    model = MockModel()
    train_loader = MockDataLoader(["train_data"])
    val_loader = MockDataLoader(["val_data"])
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    cfg = {
        'max_epochs': 10,
        'early_stopping': {
            'patience': 3,
            'monitor': 'val_loss',
            'mode': 'min'
        },
        'checkpoint': {
            'monitor': 'val_loss',
            'mode': 'min',
            'save_best_only': True
        }
    }
    
    trainer = Trainer(model, loaders, cfg)
    
    assert trainer.model == model
    assert trainer.train_loader == train_loader
    assert trainer.val_loader == val_loader
    assert trainer.cfg == cfg
    assert len(trainer.callbacks) > 0


def test_trainer_train():
    """Test Trainer train method."""
    model = MockModel()
    train_loader = MockDataLoader(["train_data"])
    val_loader = MockDataLoader(["val_data"])
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    cfg = {
        'max_epochs': 5,
        'early_stopping': {
            'patience': 3,
            'monitor': 'val_loss',
            'mode': 'min'
        },
        'checkpoint': {
            'monitor': 'val_loss',
            'mode': 'min',
            'save_best_only': True
        }
    }
    
    trainer = Trainer(model, loaders, cfg)
    
    # Test normal training cycle
    history = trainer.train()
    
    assert model.train_called
    assert model.epochs_trained > 0
    assert 'loss' in history
    assert 'accuracy' in history
    assert 'val_loss' in history
    assert 'val_accuracy' in history
    
    # Test with early stopping
    model = MockModel()
    trainer = Trainer(model, loaders, cfg)
    # Manually set early stopping callback to stop after 2 epochs
    early_stopping = EarlyStopping(patience=1, monitor='val_loss', mode='min')
    trainer.callbacks = [early_stopping]
    
    history = trainer.train()
    assert model.epochs_trained <= 2  # Should stop early 