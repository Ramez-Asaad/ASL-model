#!/usr/bin/env python3
"""
Evaluation script for ASL Detection model.

Usage:
    python scripts/evaluate_model.py --model ./models/asl_cnn.h5 --data-dir ./data/asl_alphabet_test
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_images_from_folder, prepare_data
from src.model import load_model
from src.evaluate import (
    evaluate_model, 
    get_predictions, 
    print_classification_report,
    plot_confusion_matrix
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASL Detection Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.h5)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--save-confusion', type=str, default=None,
                        help='Path to save confusion matrix image')
    args = parser.parse_args()
    
    print("=" * 50)
    print("ASL Detection Model Evaluation")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model)
    
    # Load data
    print(f"Loading test data from {args.data_dir}...")
    X, y = load_images_from_folder(args.data_dir)
    
    # Prepare (we only need test set, but function expects split)
    _, X_test, _, y_test_cat, _, y_test = prepare_data(X, y, test_size=1.0)
    
    # Actually use all data as test
    from tensorflow.keras.utils import to_categorical
    from src.config import NUM_CLASSES
    y_test_cat = to_categorical(y, NUM_CLASSES)
    X_test = X
    y_test = y
    
    print(f"Evaluating on {len(X_test)} samples...")
    
    # Evaluate
    evaluate_model(model, X_test, y_test_cat)
    
    # Get predictions
    predictions = get_predictions(model, X_test)
    
    # Classification report
    print_classification_report(y_test, predictions)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, predictions, save_path=args.save_confusion)
    

if __name__ == "__main__":
    main()
