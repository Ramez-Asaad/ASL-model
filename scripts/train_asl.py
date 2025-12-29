#!/usr/bin/env python3
"""
Training script for ASL Detection model.

Usage:
    python scripts/train_asl.py --data-dir ./data/asl_alphabet_train --epochs 50
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_images_from_folder, prepare_data
from src.model import create_asl_model, compile_model
from src.train import train_model
from src.evaluate import evaluate_model, get_predictions, print_classification_report, plot_training_history


def main():
    parser = argparse.ArgumentParser(description='Train ASL Detection Model')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output', type=str, default='models/asl_cnn.h5',
                        help='Output model path')
    args = parser.parse_args()
    
    print("=" * 50)
    print("ASL Detection Model Training")
    print("=" * 50)
    
    # Load data
    print(f"\n[1/4] Loading data from {args.data_dir}...")
    X, y = load_images_from_folder(args.data_dir)
    print(f"Loaded {len(X)} images")
    
    # Prepare data
    print("\n[2/4] Preparing data (splitting & encoding)...")
    X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = prepare_data(X, y)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Clean up to save memory
    del X, y
    
    # Create model
    print("\n[3/4] Creating model...")
    model = create_asl_model()
    model = compile_model(model)
    model.summary()
    
    # Train
    print(f"\n[4/4] Training for {args.epochs} epochs...")
    history = train_model(
        model, 
        X_train, y_train_cat,
        X_test, y_test_cat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.output
    )
    
    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    evaluate_model(model, X_test, y_test_cat)
    
    predictions = get_predictions(model, X_test)
    print_classification_report(y_test, predictions)
    
    # Plot history
    plot_training_history(history, save_path='models/training_history.png')
    
    print("\nTraining complete!")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
