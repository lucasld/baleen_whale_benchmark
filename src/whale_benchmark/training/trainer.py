"""
Model trainer for training whale call classification models.
"""
import numpy as np
import pathlib
import os
import time
from typing import Dict, List, Any, Optional

from whale_benchmark.training.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint, CSVLogger, WandbLogger
)


class Trainer:
    """
    Trainer class for training models.
    
    Args:
        model: Model to train
        loaders: Dictionary containing 'train' and 'val' data loaders
        cfg: Configuration dictionary
    """
    
    def __init__(self, model, loaders, cfg):
        """Initialize the trainer."""
        self.model = model
        self.train_loader = loaders['train']
        self.val_loader = loaders.get('val')
        self.cfg = cfg
        self.callbacks = self._setup_callbacks()
        self.history = {'loss': [], 'val_loss': []}
    
    def _setup_callbacks(self) -> List[Callback]:
        """
        Set up the callbacks for training.
        
        Returns:
            List of callback objects
        """
        callbacks = []
        
        # Early stopping callback
        if 'early_stopping' in self.cfg:
            early_stopping_cfg = self.cfg['early_stopping']
            callbacks.append(EarlyStopping(
                monitor=early_stopping_cfg.get('monitor', 'val_loss'),
                patience=early_stopping_cfg.get('patience', 10),
                mode=early_stopping_cfg.get('mode', 'auto'),
                min_delta=early_stopping_cfg.get('min_delta', 0.0)
            ))
        
        # Model checkpoint callback
        if 'checkpoint' in self.cfg:
            checkpoint_cfg = self.cfg['checkpoint']
            log_dir = pathlib.Path(self.cfg.get('log_dir', 'logs'))
            os.makedirs(log_dir, exist_ok=True)
            
            filepath = log_dir / f"{self.cfg.get('model_name', 'model')}_checkpoint"
            callbacks.append(ModelCheckpoint(
                filepath=filepath,
                monitor=checkpoint_cfg.get('monitor', 'val_loss'),
                save_best_only=checkpoint_cfg.get('save_best_only', True),
                mode=checkpoint_cfg.get('mode', 'auto')
            ))
        
        # CSV logger callback
        if 'logging' in self.cfg and self.cfg['logging'].get('csv', True):
            log_dir = pathlib.Path(self.cfg.get('log_dir', 'logs'))
            os.makedirs(log_dir, exist_ok=True)
            
            filepath = log_dir / f"{self.cfg.get('model_name', 'model')}_logs.csv"
            callbacks.append(CSVLogger(
                filename=filepath,
                separator=',',
                append=False
            ))
        
        # Weights & Biases logger
        if 'logging' in self.cfg and self.cfg['logging'].get('wandb', False):
            callbacks.append(WandbLogger(
                project=self.cfg.get('wandb_project', 'whale_benchmark'),
                config=self.cfg
            ))
        
        return callbacks
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary containing training history
        """
        max_epochs = self.cfg.get('max_epochs', 100)
        
        # Set model reference in callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'model'):
                callback.model = self.model
        
        # Call on_train_begin for all callbacks
        logs = {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)
        
        # Main training loop
        for epoch in range(max_epochs):
            # Call on_epoch_begin for all callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, logs)
            
            # Train for one epoch
            epoch_start_time = time.time()
            epoch_logs = self._train_epoch()
            
            # Validate if validation data is available
            if self.val_loader is not None:
                val_logs = self._validate()
                epoch_logs.update(val_logs)
            
            epoch_time = time.time() - epoch_start_time
            epoch_logs['epoch_time'] = epoch_time
            
            # Update history
            for k, v in epoch_logs.items():
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)
            
            # Call on_epoch_end for all callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_logs)
            
            # Check if we should stop training
            if self._should_stop():
                break
        
        # Call on_train_end for all callbacks
        for callback in self.callbacks:
            callback.on_train_end(logs)
        
        return self.history
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with epoch metrics
        """
        return self.model.fit(self.train_loader)
    
    def _validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        metrics = self.model.evaluate(self.val_loader)
        
        # Prefix validation metrics with 'val_'
        val_metrics = {}
        for k, v in metrics.items():
            val_metrics[f'val_{k}'] = v
        
        return val_metrics
    
    def _should_stop(self) -> bool:
        """
        Check if training should be stopped.
        
        Returns:
            True if training should be stopped, False otherwise
        """
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping) and callback.should_stop_training():
                return True
        return False
