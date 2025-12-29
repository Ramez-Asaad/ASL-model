"""
CNN Model architecture for ASL Detection.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten

from .config import IMAGE_SIZE, NUM_CLASSES


def create_asl_model(num_classes=NUM_CLASSES):
    """
    Create the CNN model for ASL classification.
    
    Architecture:
    - Conv2D(32, 5x5) + ReLU + MaxPool(2x2)
    - Conv2D(64, 3x3) + ReLU + MaxPool(2x2)  
    - Conv2D(64, 3x3) + ReLU + MaxPool(2x2)
    - Flatten + Dense(128) + Dense(num_classes)
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        # First conv block
        Conv2D(32, (5, 5), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Second conv block
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Third conv block
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Fully connected layers
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with Adam optimizer.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for Adam optimizer
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_model(model_path):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to .h5 model file
    
    Returns:
        Loaded Keras model
    """
    from tensorflow.keras.models import load_model as keras_load_model
    return keras_load_model(model_path)
