"""
Custom loss function for whale call classification.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses

# Risk scenarios and weights
# If prediction matches reality... it is good news so assign a standard weight of 1
# If call types are confused with each other we want e medium penalty
# If noise is predicted as call types we want a high penalty
# If call types are predicted as noise we want a low penalty
SCENARIO_RISK = {
    'prediction_matches_reality': 1.0,
    'call_type_confusion': 3.0,
    'noise_as_call_confusion': 15.0,
    'calls_as_noise_confusion': 1.5
}

# Normalize risk values
WORST_CASE_SCENARIO_RISK = max(SCENARIO_RISK.values())
SCENARIO_RISK_NORMALIZED = {
    key: value / WORST_CASE_SCENARIO_RISK for key, value in SCENARIO_RISK.items()
}


@tf.function
def _convert_scenario_risk_to_tensor(scenario):
    """
    Look up how risky the scenario is when comparing the model prediction to reality,
    and return a weight score indicating how much emphasis the model should place on
    learning from that particular observation.
    
    Args:
        scenario (str): Risk scenario key
        
    Returns:
        tf.Tensor: Weight for the specified scenario
    """
    return tf.cast(x=tf.constant(SCENARIO_RISK_NORMALIZED[scenario]), dtype=tf.float32)


@tf.function
def _get_loss_adjustment_for_scenario(actual_vs_predicted_class):
    """
    Return a weight according to how damaging the scenario is.
    
    Args:
        actual_vs_predicted_class: Tuple of [actual_class, predicted_class]
        
    Returns:
        tf.Tensor: Weight for the scenario
    """
    actual_class = actual_vs_predicted_class[0]
    predicted_class = actual_vs_predicted_class[1]
    
    # Retrieve the appropriate weighting based on how the prediction compares with reality
    return tf.case(
        [
            # If prediction matches reality
            (tf.equal(actual_class, predicted_class),
             lambda: _convert_scenario_risk_to_tensor('prediction_matches_reality')),
            
            # If call types are confused with each other (both not noise, class 3)
            (tf.logical_and(
                tf.logical_and(
                    tf.math.not_equal(actual_class, 3),
                    tf.math.not_equal(predicted_class, 3)
                ),
                tf.math.not_equal(predicted_class, actual_class)
            ),
             lambda: _convert_scenario_risk_to_tensor('call_type_confusion')),
            
            # If call types are predicted as noise
            (tf.logical_and(
                tf.math.not_equal(actual_class, 3),
                tf.equal(predicted_class, 3)
            ),
             lambda: _convert_scenario_risk_to_tensor('calls_as_noise_confusion')),
            
            # If noise is predicted as call types
            (tf.logical_and(
                tf.equal(actual_class, 3),
                tf.math.not_equal(predicted_class, 3)
            ),
             lambda: _convert_scenario_risk_to_tensor('noise_as_call_confusion')),
        ]
    )


@tf.function
def number_comparison(actual, predicted):
    """
    Alternative simpler method for calculating loss adjustment.
    
    Args:
        actual: Actual class
        predicted: Predicted class
        
    Returns:
        tf.Tensor: Weights for each example
    """
    new_tensor = tf.math.subtract(tf.math.exp(actual), tf.math.exp(predicted))
    weightings = tf.zeros([tf.size(new_tensor)])
    weightings = tf.where(new_tensor == 0, 0.07, weightings)
    weightings = tf.where(
        tf.less(new_tensor, 7) & 
        tf.math.not_equal(new_tensor, 0) & 
        tf.greater(new_tensor, -10), 
        0.2, 
        weightings
    )
    weightings = tf.where(new_tensor < -10, 0.1, weightings)
    weightings = tf.where(new_tensor > 10, 1.0, weightings)
    
    return weightings


@tf.function
def custom_cross_entropy(y_actual, y_prediction):
    """
    Calculate cross entropy loss, but weighted according to how risky the scenario is.
    
    Args:
        y_actual: Ground truth labels (sparse)
        y_prediction: Predicted probabilities
        
    Returns:
        tf.Tensor: Weighted cross-entropy loss
    """
    # Calculate standard cross-entropy
    standard_cross_entropy = losses.sparse_categorical_crossentropy(
        y_true=y_actual, 
        y_pred=y_prediction
    )
    
    # Get predicted class
    predicted_class = tf.math.argmax(input=y_prediction, axis=1)
    predicted_class = tf.cast(x=predicted_class, dtype=tf.float32)
    
    # Convert y_actual to float tensor
    actual_class = tf.cast(x=y_actual, dtype=tf.float32)
    
    # Calculate weights
    weighting = number_comparison(actual_class, predicted_class)
    
    # Apply weighting to cross-entropy
    return tf.math.multiply(standard_cross_entropy, weighting) 