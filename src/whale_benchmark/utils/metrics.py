"""
Metrics for evaluating whale call classification models.
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import confusion_matrix


class ImbalancedDetectionMatrix:
    """
    Specialized metrics for imbalanced detection tasks with a noise class.
    
    Args:
        noise_class_name (str): Name of the noise class
        classes_names (list): List of all class names
    """
    
    def __init__(self, noise_class_name, classes_names):
        """Initialize the metric."""
        self.classes_names = classes_names
        self.not_noise_indices = []
        for class_i, name in enumerate(classes_names):
            if name != noise_class_name:
                self.not_noise_indices.append([class_i])
            else:
                self.noise_class_index = [class_i]
    
    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            tf.Tensor: Confusion matrix
        """
        y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=len(self.classes_names))
        return cm
    
    def imbalanced_metric(self, y_true, y_pred):
        """
        Calculate a metric that balances call detection and noise misclassification.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            tf.Tensor: Imbalanced metric value
        """
        cm = self.confusion_matrix(y_true, y_pred)
        cm_calls = tf.gather_nd(cm, indices=self.not_noise_indices)
        cm_noise_total = tf.gather_nd(cm, indices=self.noise_class_index)
        cm_noise = tf.gather_nd(cm_noise_total, indices=self.not_noise_indices)
        diag_part = tf.linalg.diag_part(cm_calls)
        detections = tf.reduce_sum(cm_calls, 1) + tf.constant(1e-15)
        call_avg_tpr = tf.reduce_mean(diag_part / detections)
        noise_misclassification_rate = 1 - tf.reduce_sum(cm_noise) / (tf.reduce_sum(cm_noise_total, 0) +
                                                                      tf.constant(1e-15))
        imbalanced_metric = 2 * (call_avg_tpr * (noise_misclassification_rate**2)) / (call_avg_tpr +
                                                                                 noise_misclassification_rate)
        return imbalanced_metric
    
    def call_avg_tpr(self, y_true, y_pred):
        """
        Calculate the average true positive rate for call classes.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            tf.Tensor: Average TPR for call classes
        """
        cm = self.confusion_matrix(y_true, y_pred)
        
        cm_calls = tf.gather_nd(cm, indices=self.not_noise_indices)
        diag_part = tf.linalg.diag_part(cm_calls)
        detections = tf.reduce_sum(cm_calls, 1) + tf.constant(1e-15)
        call_avg_tpr = tf.reduce_mean(diag_part / detections)
        return call_avg_tpr
    
    def noise_misclas_rate(self, y_true, y_pred):
        """
        Calculate the noise misclassification rate.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            tf.Tensor: Noise misclassification rate
        """
        cm = self.confusion_matrix(y_true, y_pred)
        
        cm_noise_total = tf.gather_nd(cm, indices=self.noise_class_index)
        cm_noise = tf.gather_nd(cm_noise_total, indices=self.not_noise_indices)
        noise_misclassification_rate = tf.reduce_sum(cm_noise) / (tf.reduce_sum(cm_noise_total, 0) + tf.constant(1e-15))
        return noise_misclassification_rate


def recall_score(y_true, y_pred):
    """
    Calculate the recall score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        tf.Tensor: Recall score
    """
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall_val = true_positives / (all_positives + K.epsilon())
    return recall_val


def precision_score(y_true, y_pred):
    """
    Calculate the precision score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        tf.Tensor: Precision score
    """
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        tf.Tensor: F1 score
    """
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


def calculate_confusion_matrix(y_true, y_pred, class_names):
    """
    Calculate confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels (integer encoded)
        y_pred (np.ndarray): Predicted probabilities
        class_names (list): List of class names
        
    Returns:
        pd.DataFrame: Confusion matrix as DataFrame
    """
    import pandas as pd
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Convert to DataFrame
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    return cm_df


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate common evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels (integer encoded)
        y_pred (np.ndarray): Predicted probabilities
        class_names (list): List of class names
        
    Returns:
        dict: Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    # Calculate precision, recall, and F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None
    )
    
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred, class_names)
    
    # Check if "Noise" is in class_names
    if "Noise" in class_names:
        noise_index = class_names.index("Noise")
        total_noise = np.sum(cm.iloc[noise_index])
        misclassified_noise = total_noise - cm.iloc[noise_index, noise_index]
        noise_misclass_rate = misclassified_noise / total_noise if total_noise > 0 else 0
        
        # Calculate TPR for call classes (not Noise)
        call_tpr = []
        for i, name in enumerate(class_names):
            if name != "Noise":
                class_tpr = cm.iloc[i, i] / np.sum(cm.iloc[i]) if np.sum(cm.iloc[i]) > 0 else 0
                call_tpr.append(class_tpr)
        
        call_avg_tpr = np.mean(call_tpr) if call_tpr else 0
        
        # Calculate imbalanced metric
        if call_avg_tpr + (1 - noise_misclass_rate) > 0:
            imbalanced_metric = 2 * (call_avg_tpr * ((1 - noise_misclass_rate)**2)) / (call_avg_tpr + (1 - noise_misclass_rate))
        else:
            imbalanced_metric = 0
    else:
        noise_misclass_rate = None
        call_avg_tpr = None
        imbalanced_metric = None
    
    metrics = {
        'accuracy': accuracy,
        'precision': {class_names[i]: precision[i] for i in range(len(class_names))},
        'recall': {class_names[i]: recall[i] for i in range(len(class_names))},
        'f1': {class_names[i]: f1[i] for i in range(len(class_names))},
        'support': {class_names[i]: support[i] for i in range(len(class_names))},
        'confusion_matrix': cm
    }
    
    if "Noise" in class_names:
        metrics.update({
            'noise_misclass_rate': noise_misclass_rate,
            'call_avg_tpr': call_avg_tpr,
            'imbalanced_metric': imbalanced_metric
        })
    
    return metrics
