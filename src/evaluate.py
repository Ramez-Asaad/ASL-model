"""
Evaluation and metrics for ASL Detection model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .config import CLASS_LABELS


def evaluate_model(model, X_test, y_test_cat, verbose=True):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test_cat: One-hot encoded test labels
        verbose: Whether to print results
    
    Returns:
        loss, accuracy tuple
    """
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    if verbose:
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    return loss, accuracy


def get_predictions(model, X_test):
    """
    Get model predictions.
    
    Args:
        model: Trained Keras model
        X_test: Test images
    
    Returns:
        Predicted class indices
    """
    predictions = model.predict(X_test, verbose=0)
    return np.argmax(predictions, axis=1)


def print_classification_report(y_true, y_pred, labels=CLASS_LABELS):
    """
    Print classification report.
    
    Args:
        y_true: True labels (not one-hot)
        y_pred: Predicted labels
        labels: Class label names
    """
    # Get unique classes present in the data
    unique_classes = sorted(set(y_true) | set(y_pred))
    present_labels = [labels[i] for i in unique_classes]
    
    report = classification_report(y_true, y_pred, labels=unique_classes, target_names=present_labels)
    print("Classification Report:")
    print(report)
    return report


def plot_confusion_matrix(y_true, y_pred, labels=CLASS_LABELS, 
                         figsize=(14, 12), save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class label names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Get unique classes present in the data
    unique_classes = sorted(set(y_true) | set(y_pred))
    present_labels = [labels[i] for i in unique_classes]
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_labels, yticklabels=present_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Keras training history object
        save_path: Optional path to save figure
    """
    metrics = pd.DataFrame(history.history)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(metrics['loss'], label='Training Loss')
    axes[0].plot(metrics['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(metrics['accuracy'], label='Training Accuracy')
    axes[1].plot(metrics['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()
