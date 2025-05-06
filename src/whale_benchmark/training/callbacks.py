"""
Callbacks for model training.

These callbacks are used to perform actions during training,
like early stopping, checkpointing, etc.
"""
import os
import pathlib
import numpy as np
import json


class Callback:
    """Base class for callbacks."""
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    
    Args:
        monitor (str): Quantity to monitor, e.g., 'val_loss'
        patience (int): Number of epochs with no improvement after which training will be stopped
        mode (str): One of {'min', 'max', 'auto'}. 'min' => monitored quantity should decrease, 
                   'max' => monitored quantity should increase
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
    """
    
    def __init__(self, monitor='val_loss', patience=0, mode='auto', min_delta=0.0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = None
        self.stopped_epoch = 0
        self.should_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:  # auto
            if 'loss' in monitor:
                self.monitor_op = np.less
                self.min_delta *= -1
            else:
                self.monitor_op = np.greater
                self.min_delta *= 1
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
    
    def should_stop_training(self):
        return self.should_stop


class ModelCheckpoint(Callback):
    """
    Save the model after every epoch or only the best model according to a monitored metric.
    
    Args:
        filepath (str): Path to save the model file
        monitor (str): Quantity to monitor
        save_best_only (bool): If True, only save the model when the monitored metric is improved
        mode (str): One of {'min', 'max', 'auto'}
    """
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=False, mode='auto'):
        super().__init__()
        self.filepath = pathlib.Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:  # auto
            if 'loss' in monitor:
                self.monitor_op = np.less
                self.best = np.Inf
            else:
                self.monitor_op = np.greater
                self.best = -np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        if not self.save_best_only:
            filepath = str(self.filepath).format(epoch=epoch, **logs)
            self.model.save(filepath)
        else:
            current = logs.get(self.monitor)
            if current is None:
                return
            
            if self.monitor_op(current, self.best):
                self.best = current
                filepath = str(self.filepath)
                self.model.save(filepath)


class CSVLogger(Callback):
    """
    Callback that streams epoch results to a CSV file.
    
    Args:
        filename (str): Path to the CSV file
        separator (str): String used to separate elements in the CSV file
        append (bool): Whether to append to an existing file
    """
    
    def __init__(self, filename, separator=',', append=False):
        super().__init__()
        self.filename = pathlib.Path(filename)
        self.separator = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.epoch = 0
        
        os.makedirs(self.filename.parent, exist_ok=True)
        self.file = None
    
    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.keys = f.readline().strip().split(self.separator)
                self.file = open(self.filename, 'a')
                return
        
        self.file = open(self.filename, 'w')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch
        
        if self.keys is None:
            self.keys = sorted(logs.keys())
            header = self.separator.join(self.keys) + '\n'
            self.file.write(header)
        
        row_dict = {key: logs[key] for key in self.keys}
        row = self.separator.join(str(row_dict[key]) for key in self.keys)
        self.file.write(row + '\n')
        self.file.flush()
    
    def on_train_end(self, logs=None):
        if self.file is not None:
            self.file.close()
            self.file = None


class WandbLogger(Callback):
    """
    Callback that logs metrics to Weights & Biases.
    
    Args:
        project (str): W&B project name
        config (dict): Additional configuration to log
    """
    
    def __init__(self, project, config=None):
        super().__init__()
        self.project = project
        self.config = config or {}
        self.run = None
    
    def on_train_begin(self, logs=None):
        try:
            import wandb
            self.run = wandb.init(project=self.project, config=self.config)
        except ImportError:
            print("wandb not installed. WandbLogger will not log metrics.")
            self.run = None
    
    def on_epoch_end(self, epoch, logs=None):
        if self.run is not None:
            import wandb
            wandb.log(logs, step=epoch)
    
    def on_train_end(self, logs=None):
        if self.run is not None:
            import wandb
            wandb.finish()
