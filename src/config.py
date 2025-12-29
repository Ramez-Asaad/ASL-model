"""
Configuration constants for ASL Detection model.
"""

# Image settings
IMAGE_SIZE = 64
TARGET_DIMS = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Model settings
NUM_CLASSES = 29
BATCH_SIZE = 64

# Training settings
EPOCHS = 50
TEST_SIZE = 0.3
RANDOM_STATE = 42
EARLY_STOP_PATIENCE = 2

# Class labels (A-Z + del, nothing, space)
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Label to index mapping
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_LABELS)}

# Paths
DEFAULT_DATA_DIR = 'data/asl_alphabet_train'
DEFAULT_MODEL_PATH = 'models/asl_cnn.h5'
