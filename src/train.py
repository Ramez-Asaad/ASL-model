"""
Training utilities for ASL Detection model.
"""

import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .config import EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE


def get_callbacks(model_save_path, early_stop_patience=EARLY_STOP_PATIENCE):
    """
    Get training callbacks.
    
    Args:
        model_save_path: Path to save best model
        early_stop_patience: Patience for early stopping
    
    Returns:
        List of Keras callbacks
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=EPOCHS, batch_size=BATCH_SIZE, 
                model_save_path='models/asl_cnn.h5'):
    """
    Train the model.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        model_save_path: Path to save best model
    
    Returns:
        Training history
    """
    callbacks = get_callbacks(model_save_path)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return history


def save_model(model, path):
    """
    Save model to disk.
    
    Args:
        model: Trained Keras model
        path: Save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")
