"""
Model architecture factory for the baleen whale benchmark.
Provides build_model_architecture() and get_available_models() functions.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# Import dataset constants - use the values directly to avoid import issues
IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1


def build_efficientnet_b0(n_classes, batch_size):
    """
    EfficientNet-B0 implementation with input resizing to meet minimum size requirements.
    Loads ImageNet pretrained weights and adapts for grayscale input.
    Resizes 30x90 input to 32x96 to maintain aspect ratio while meeting EfficientNet's 32x32 minimum.
    """
    # Create base EfficientNet-B0 with minimum required input size (32x96 maintains ~3:1 aspect ratio)
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 96, 3)  # Minimum size that preserves aspect ratio
    )
    
    # Create our model
    model = keras.Sequential([
        # Convert grayscale to RGB by repeating the channel
        keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), 
                          input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
        
        # Resize from 30x90 to 32x96 to meet EfficientNet minimum requirements
        keras.layers.Lambda(lambda x: tf.image.resize(x, [32, 96])),
        
        # Add the pretrained base
        base_model,
        
        # Add our classifier head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu', 
                          kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    # Freeze the base model initially for transfer learning
    base_model.trainable = False
    
    return model


def build_idil_cnn(n_classes, batch_size):
    """
    Original IdilCNN implementation (fallback).
    """
    previously_batch_size = 16
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(previously_batch_size, kernel_size=(3, 3), activation='relu', padding="same",
                                     kernel_initializer='he_normal', input_shape=(IMAGE_WIDTH,
                                                                                  IMAGE_HEIGHT, 1)))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(previously_batch_size, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(previously_batch_size, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(
        tf.keras.layers.Conv2D(previously_batch_size * 2, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.Conv2D(previously_batch_size * 4, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(previously_batch_size * 2, kernel_regularizer=regularizers.l2(0.001)))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model


def build_mobilenet_v2(n_classes, batch_size):
    """
    Lightweight MobileNetV2 implementation with far fewer parameters than EfficientNet.
    Uses ImageNet pretrained weights and adapts for grayscale input.
    Resizes 30x90 input to 32x96 to meet minimum requirements while maintaining aspect ratio.
    """
    # Create base MobileNetV2 with minimum required input size (32x96 maintains ~3:1 aspect ratio)
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 96, 3),  # Minimum size that preserves aspect ratio
        alpha=1.0  # Width multiplier (use full model)
    )
    
    # Create our model
    model = keras.Sequential([
        # Convert grayscale to RGB by repeating the channel
        keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), 
                          input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
        
        # Resize from 30x90 to 32x96 to meet MobileNet minimum requirements
        keras.layers.Lambda(lambda x: tf.image.resize(x, [32, 96])),
        
        # Add the pretrained base
        base_model,
        
        # Add our classifier head (simpler than EfficientNet)
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu', 
                          kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    # Freeze the base model initially for transfer learning
    base_model.trainable = False
    
    return model


def build_mobilenet_v3_small(n_classes, batch_size):
    """
    MobileNetV3Small - Very lightweight pretrained model with minimal parameters.
    Much smaller than MobileNetV2 while still providing pretrained ImageNet features.
    """
    # Create base MobileNetV3Small with ImageNet weights
    base_model = keras.applications.MobileNetV3Small(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 96, 3),  # Use same size as EfficientNet for consistency
        minimalistic=False,  # Use full model, not minimalistic version
    )
    
    # Create our model
    model = keras.Sequential([
        # Convert grayscale to RGB by repeating the channel
        keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), 
                          input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
        
        # Resize from 30x90 to 32x96 for consistency with other models
        keras.layers.Lambda(lambda x: tf.image.resize(x, [32, 96])),
        
        # Add the pretrained base
        base_model,
        
        # Add our classifier head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),  # Lower dropout for smaller model
        keras.layers.Dense(64, activation='relu',  # Smaller dense layer
                          kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    # Freeze the base model initially for transfer learning
    base_model.trainable = False
    
    return model


def build_simple_cnn(n_classes, batch_size):
    """
    Very simple CNN with minimal parameters for comparison.
    Designed to be extremely lightweight while still functional.
    """
    model = tf.keras.models.Sequential([
        # First conv block
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',
                              input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second conv block
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu',
                             kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    return model


# Registry of available models
MODEL_REGISTRY = {
    'IdilCNN': build_idil_cnn,
    'EfficientNet-B0': build_efficientnet_b0,
    'MobileNet-V2': build_mobilenet_v2,
    'MobileNet-V3-Small': build_mobilenet_v3_small,
    'SimpleCNN': build_simple_cnn,
}


def build_model_architecture(model_architecture, n_classes, batch_size):
    """
    Build a model architecture by name.
    
    Args:
        model_architecture: Name of the architecture (e.g., "IdilCNN", "EfficientNet-B0")
        n_classes: Number of output classes
        batch_size: Batch size (used by some architectures)
        
    Returns:
        Keras model instance
        
    Raises:
        ValueError: If the architecture name is not found
    """
    if model_architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture: {model_architecture}")
    
    builder_func = MODEL_REGISTRY[model_architecture]
    return builder_func(n_classes, batch_size)


def get_available_models():
    """
    Get list of available model architectures.
    
    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())
