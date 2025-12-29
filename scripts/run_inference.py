#!/usr/bin/env python3
"""
Run real-time inference from webcam.

Usage:
    python scripts/run_inference.py --model ./models/asl_cnn.h5
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference import run_webcam_inference


def main():
    parser = argparse.ArgumentParser(description='Run ASL Inference from Webcam')
    parser.add_argument('--model', type=str, default='models/asl_cnn.h5',
                        help='Path to trained model (.h5)')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using: python scripts/train_asl.py")
        sys.exit(1)
    
    run_webcam_inference(args.model)


if __name__ == "__main__":
    main()
