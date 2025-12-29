"""
Data loading and preprocessing utilities for ASL Detection.
"""

import os
import numpy as np
import cv2
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from .config import (
    IMAGE_SIZE, NUM_CLASSES, TEST_SIZE, RANDOM_STATE, LABEL_TO_IDX
)


def load_images_from_folder(folder_path, verbose=True):
    """
    Load all images from a folder structure where each subfolder is a class.
    
    Args:
        folder_path: Path to the training data directory
        verbose: Whether to print progress
    
    Returns:
        X: numpy array of images (N, IMAGE_SIZE, IMAGE_SIZE, 3)
        y: numpy array of labels (N,)
    """
    # Count total images first
    total_images = 0
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_full_path) and not folder_name.startswith('.'):
            total_images += len([f for f in os.listdir(folder_full_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if verbose:
        print(f"Found {total_images} images to load...")
    
    X = np.empty((total_images, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    y = np.empty((total_images,), dtype=np.int32)
    
    cnt = 0
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)
        
        if not os.path.isdir(folder_full_path) or folder_name.startswith('.'):
            continue
            
        # Get label index
        if folder_name in LABEL_TO_IDX:
            label = LABEL_TO_IDX[folder_name]
        else:
            if verbose:
                print(f"Warning: Unknown class '{folder_name}', skipping...")
            continue
        
        # Load all images in this class folder
        for image_filename in os.listdir(folder_full_path):
            if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(folder_full_path, image_filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize and normalize
                img_resized = resize(img, (IMAGE_SIZE, IMAGE_SIZE, 3), 
                                    anti_aliasing=True, preserve_range=False)
                X[cnt] = img_resized.astype(np.float32)
                y[cnt] = label
                cnt += 1
                
                if verbose and cnt % 5000 == 0:
                    print(f"Loaded {cnt}/{total_images} images...")
    
    # Trim arrays if we loaded fewer images than expected
    X = X[:cnt]
    y = y[:cnt]
    
    if verbose:
        print(f"Successfully loaded {cnt} images.")
    
    return X, y


def prepare_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data and apply one-hot encoding.
    
    Args:
        X: Image data
        y: Labels
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test: Split and encoded data
    """
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    return X_train, X_test, y_train_cat, y_test_cat, y_train, y_test


def preprocess_single_image(image):
    """
    Preprocess a single image for inference.
    
    Args:
        image: BGR image from cv2.imread or webcam
    
    Returns:
        Preprocessed image ready for model prediction
    """
    img_resized = resize(image, (IMAGE_SIZE, IMAGE_SIZE, 3), 
                        anti_aliasing=True, preserve_range=False)
    return np.expand_dims(img_resized.astype(np.float32), axis=0)
