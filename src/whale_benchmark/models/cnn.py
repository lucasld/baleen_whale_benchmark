"""
CNN model for whale call classification.
"""
import os
import pathlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import regularizers
from sklearn.utils.class_weight import compute_class_weight

from whale_benchmark.models.base import BaseModel
from whale_benchmark.utils.registry import register_model
from whale_benchmark.utils.metrics import ImbalancedDetectionMatrix, f1_score, recall_score
from whale_benchmark.data.spectrogram_dataset import IMAGE_WIDTH, IMAGE_HEIGHT


@register_model("cnn")
class CNNModel(BaseModel):
    """
    Convolutional Neural Network model for whale call classification.
    
    This model is a port of the legacy CNN model.
    """
    
    def __init__(self, 
                 save_path="models", 
                 categories=None, 
                 model_name="cnn_model",
                 learning_rate=0.001,
                 batch_size=32,
                 epochs=100,
                 early_stop=10,
                 monitoring_metric="val_loss",
                 monitoring_direction="min",
                 class_weights="balanced",
                 loss_function="categorical_crossentropy"):
        """
        Initialize the CNN model.
        
        Args:
            save_path (str): Path to save models and logs
            categories (list): List of category names
            model_name (str): Name of the model
            learning_rate (float): Initial learning rate
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs
            early_stop (int): Number of epochs with no improvement to stop
            monitoring_metric (str): Metric to monitor for early stopping
            monitoring_direction (str): Direction of improvement ('min', 'max', 'auto')
            class_weights (str): Type of class weighting ('balanced', 'None', or dict)
            loss_function (str): Loss function to use
        """
        super().__init__()
        self.save_path = pathlib.Path(save_path)
        self.model_name = model_name
        self.log_path = self.save_path.joinpath(model_name)
        self.categories = categories or []
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.monitoring_metric = monitoring_metric
        self.monitoring_direction = monitoring_direction
        self.class_weights = class_weights
        self.loss_function = loss_function
        
        # Create the log directory if it doesn't exist
        if not self.log_path.exists():
            os.makedirs(str(self.log_path), exist_ok=True)
        
        # Setup metrics
        self.setup_metrics()
        
        # Model to be created or loaded
        self.model = None
        self.history = None
    
    def setup_metrics(self):
        """Set up the model metrics."""
        if self.categories:
            detection_metrics = ImbalancedDetectionMatrix(
                noise_class_name='Noise', 
                classes_names=self.categories
            )
            
            self.metrics = [
                keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                f1_score,
                recall_score,
                detection_metrics.imbalanced_metric,
                detection_metrics.noise_misclas_rate,
                detection_metrics.call_avg_tpr
            ]
            
            self.metrics_dict = {
                'imbalanced_metric': detection_metrics.imbalanced_metric,
                'noise_misclas_rate': detection_metrics.noise_misclas_rate,
                'call_avg_tpr': detection_metrics.call_avg_tpr,
                'f1_score': f1_score,
                'recall_score': recall_score,
                'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
            }
        else:
            self.metrics = [
                keras.metrics.SparseCategoricalAccuracy(name='accuracy')
            ]
            self.metrics_dict = {
                'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
            }
    
    @classmethod
    def from_config(cls, cfg):
        """
        Create a model from configuration.
        
        Args:
            cfg (dict): Configuration dictionary
            
        Returns:
            CNNModel: Instantiated model
        """
        return cls(
            save_path=cfg.get('save_path', 'models'),
            categories=cfg.get('categories', []),
            model_name=cfg.get('model_name', 'cnn_model'),
            learning_rate=cfg.get('learning_rate', 0.001),
            batch_size=cfg.get('batch_size', 32),
            epochs=cfg.get('epochs', 100),
            early_stop=cfg.get('early_stop', 10),
            monitoring_metric=cfg.get('monitoring_metric', 'val_loss'),
            monitoring_direction=cfg.get('monitoring_direction', 'min'),
            class_weights=cfg.get('class_weights', 'balanced'),
            loss_function=cfg.get('loss_function', 'categorical_crossentropy')
        )
    
    def create(self, n_classes):
        """
        Create the CNN model.
        
        Args:
            n_classes (int): Number of classes
            
        Returns:
            tf.keras.Model: Created model
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(self.batch_size, kernel_size=(3, 3), activation='relu', padding="same",
                                         kernel_initializer='he_normal', input_shape=(IMAGE_WIDTH,
                                                                                      IMAGE_HEIGHT, 1)))

        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(self.batch_size, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(self.batch_size, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(
            tf.keras.layers.Conv2D(self.batch_size * 2, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Conv2D(self.batch_size * 4, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(self.batch_size * 2, kernel_regularizer=regularizers.l2(0.001)))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

        # Save model summary
        model_summary_path = self.log_path.joinpath('model_summary.txt')
        with open(model_summary_path, 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
        self.model = model
        return model
    
    def fit(self, train_loader):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader with training data
            
        Returns:
            dict: Training history
        """
        # If the model hasn't been created yet, create it now
        if self.model is None:
            self.create(n_classes=train_loader.n_classes)
        
        # Extract data
        x_train, y_train = train_loader.x, train_loader.y_true
        x_valid, y_valid = None, None
        if hasattr(train_loader, 'x_valid') and hasattr(train_loader, 'y_valid'):
            x_valid, y_valid = train_loader.x_valid, train_loader.y_valid
        
        # Compute class weights
        class_labels = np.unique(y_train)
        class_weights = compute_class_weight(self.class_weights, classes=class_labels, y=y_train)
        class_weights_dict = {}
        for label, weight in zip(class_labels, class_weights):
            class_weights_dict[label] = weight
        print('Used class weights:', class_weights_dict)
        
        # Setup learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=(len(x_train) / self.batch_size) * 10,
            decay_rate=0.5
        )
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        
        # Setup loss function
        loss_function = self.loss_function
        if loss_function == 'custom_cross_entropy':
            from whale_benchmark.models.custom_loss import custom_cross_entropy
            loss_function = custom_cross_entropy
        
        # Compile the model
        self.model.compile(
            loss=loss_function,
            optimizer=opt,
            metrics=self.metrics,
            jit_compile=True
        )
        
        # Setup callbacks
        model_save_filename = self.log_path.joinpath('checkpoints')
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor=self.monitoring_metric, 
            mode=self.monitoring_direction,
            patience=self.early_stop, 
            restore_best_weights=True
        )
        mdl_checkpoint_cb = keras.callbacks.ModelCheckpoint(
            model_save_filename, 
            save_weights_only=True,
            monitor=self.monitoring_metric, 
            save_best_only=True
        )
        history_cb = tf.keras.callbacks.CSVLogger(
            self.log_path.joinpath('logs.csv'), 
            separator=',', 
            append=False
        )
        
        # Train the model
        validation_data = None
        if x_valid is not None and y_valid is not None:
            validation_data = (x_valid, y_valid)
            
        history = self.model.fit(
            x_train, y_train, 
            batch_size=self.batch_size, 
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=[early_stopping_cb, mdl_checkpoint_cb, history_cb],
            class_weight=class_weights_dict
        )
        
        self.history = history.history
        self.plot_training_metrics()
        
        # Return metrics from the last epoch
        return {k: v[-1] for k, v in self.history.items()}
    
    def predict(self, loader):
        """
        Generate predictions for the provided data.
        
        Args:
            loader: DataLoader with data to predict on
            
        Returns:
            numpy.ndarray: Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call fit() or load() first.")
        
        if hasattr(loader, 'x'):
            # Predict on data in memory
            return self.model.predict(loader.x)
        else:
            # Batch prediction
            all_preds = []
            for batch in loader:
                batch_x = batch[0]
                preds = self.model.predict(batch_x)
                all_preds.append(preds)
            return np.concatenate(all_preds)
    
    def evaluate(self, loader):
        """
        Evaluate model performance.
        
        Args:
            loader: DataLoader with data to evaluate on
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call fit() or load() first.")
        
        if hasattr(loader, 'x') and hasattr(loader, 'y_true'):
            # Evaluate on data in memory
            scores = self.model.evaluate(loader.x, loader.y_true, verbose=0)
            metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics[metric_name] = scores[i]
            
            # Add confusion matrix
            y_pred = self.predict(loader)
            y_pred_classes = np.argmax(y_pred, axis=1)
            confusion_mat = tf.math.confusion_matrix(
                loader.y_true, 
                y_pred_classes, 
                num_classes=len(self.categories)
            ).numpy()
            
            metrics['confusion_matrix'] = pd.DataFrame(
                confusion_mat, 
                index=self.categories, 
                columns=self.categories
            )
            
            return metrics
        else:
            # Fall back to base implementation
            return super().evaluate(loader)
    
    def plot_training_metrics(self):
        """Plot training metrics."""
        if self.history is None:
            return
        
        # Extract metrics
        if self.monitoring_metric in self.history:
            monitoring_metric = self.monitoring_metric
            monitored_values = self.history[monitoring_metric]
            non_val_metric = monitoring_metric.split('val_')[-1]
            if non_val_metric in self.history:
                # Val and train versions of the same metric
                train_values = self.history[non_val_metric]
                
                # Plot monitored metric
                plt.figure(figsize=(8, 8))
                plt.plot(range(1, len(train_values) + 1), train_values, 'r--')
                plt.plot(range(1, len(monitored_values) + 1), monitored_values, 'b-')
                plt.legend([f'Training {non_val_metric}', f'Validation {non_val_metric}'])
                plt.xlabel('Epoch')
                plt.ylabel(non_val_metric)
                plt.yticks(np.arange(0, 1, 0.05))
                plt.xticks(np.arange(0, max(len(train_values), 20), 2))
                plt.grid()
                plt.savefig(self.log_path.joinpath(f'training_{monitoring_metric}_{self.model_name}.png'))
                plt.close()
        
        # Plot loss
        if 'loss' in self.history and 'val_loss' in self.history:
            plt.figure(figsize=(8, 8))
            plt.plot(range(1, len(self.history['loss']) + 1), self.history['loss'], 'g--')
            plt.plot(range(1, len(self.history['val_loss']) + 1), self.history['val_loss'], 'r-')
            plt.legend(['Training loss', 'Validation loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(np.arange(0, max(len(self.history['loss']), 20), 2))
            plt.grid()
            plt.savefig(self.log_path.joinpath(f'training_loss_{self.model_name}.png'))
            plt.close()
    
    def plot_confusion_matrix(self, confusion_matrix, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix (pd.DataFrame): Confusion matrix
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        hm = plt.imshow(confusion_matrix, cmap='Blues')
        plt.colorbar(hm)
        
        # Add labels
        categories = confusion_matrix.index
        plt.xticks(range(len(categories)), categories, rotation=45)
        plt.yticks(range(len(categories)), categories)
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(categories)):
                plt.text(j, i, confusion_matrix.iloc[i, j], 
                        ha="center", va="center", color="w" if confusion_matrix.iloc[i, j] > confusion_matrix.max().max()/2 else "k")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(self.log_path.joinpath(f'confusion_matrix_{self.model_name}.png'))
        
        plt.close()
    
    def save(self, path=None):
        """
        Save the model to disk.
        
        Args:
            path (str, optional): Path to save the model
        """
        if path is None:
            path = self.log_path.joinpath('model')
        
        if self.model is not None:
            self.model.save(path)
        
        return True
    
    @classmethod
    def load(cls, path):
        """
        Load a model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            CNNModel: Loaded model
        """
        path = pathlib.Path(path)
        
        # Determine model name and save path
        model_name = path.name
        save_path = path.parent
        
        # Create instance
        instance = cls(save_path=save_path, model_name=model_name)
        
        # Load the Keras model
        instance.model = tf.keras.models.load_model(
            path, 
            custom_objects=instance.metrics_dict
        )
        
        return instance
