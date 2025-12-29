# ASL Detection - American Sign Language Recognition

A CNN-based deep learning model for real-time American Sign Language (ASL) alphabet recognition with 99% accuracy.

## Features

- 🎯 **29 Classes**: A-Z letters + `del`, `nothing`, `space`
- 🧠 **CNN Architecture**: 3 convolutional blocks with max pooling
- 📷 **Real-time Inference**: Webcam-based gesture recognition
- 📊 **Comprehensive Evaluation**: Confusion matrix, classification reports

## Project Structure

```
ASL-model/
├── src/
│   ├── __init__.py         # Package exports
│   ├── config.py           # Configuration constants
│   ├── data_loader.py      # Data loading & preprocessing
│   ├── model.py            # CNN model architecture
│   ├── train.py            # Training utilities
│   ├── evaluate.py         # Evaluation & metrics
│   └── inference.py        # Real-time inference
├── scripts/
│   ├── train_asl.py        # Training entry point
│   ├── evaluate_model.py   # Evaluation entry point
│   └── run_inference.py    # Webcam inference
├── models/                  # Saved models
├── data/                    # Training data
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train_asl.py --data-dir ./data/asl_alphabet_train --epochs 50
```

### Evaluation

```bash
python scripts/evaluate_model.py --model ./models/asl_cnn.h5 --data-dir ./data/asl_alphabet_test
```

### Real-time Inference

```bash
python scripts/run_inference.py --model ./models/asl_cnn.h5
```

## Dataset

This model is trained on the [ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet) from Kaggle:
- 87,000 images (200x200 pixels)
- 29 classes (A-Z + del/nothing/space)
- 3,000 images per class

## Model Architecture

```
Conv2D(32, 5x5) → ReLU → MaxPool(2x2)
Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
Flatten → Dense(128) → Dense(29, softmax)
```

## Credits

Based on the [ASL Detection notebook](https://www.kaggle.com/namanmanchanda/asl-detection-99-accuracy) by Naman Manchanda.