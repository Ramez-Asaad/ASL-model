"""
ASL Detection - Modular package for American Sign Language recognition.

Import specific modules as needed:
    from src.config import CLASS_LABELS
    from src.model import create_asl_model
    from src.inference import ASLInferenceService
"""

# Only expose config without TensorFlow dependency
from .config import CLASS_LABELS, IDX_TO_LABEL, LABEL_TO_IDX, IMAGE_SIZE, NUM_CLASSES

__all__ = [
    'CLASS_LABELS',
    'IDX_TO_LABEL', 
    'LABEL_TO_IDX',
    'IMAGE_SIZE',
    'NUM_CLASSES',
]

# Lazy imports for TensorFlow-dependent modules
def __getattr__(name):
    """Lazy import for TensorFlow-dependent modules."""
    if name in ('create_asl_model', 'compile_model', 'load_model'):
        from .model import create_asl_model, compile_model, load_model
        return locals()[name]
    elif name in ('load_images_from_folder', 'prepare_data', 'preprocess_single_image'):
        from .data_loader import load_images_from_folder, prepare_data, preprocess_single_image
        return locals()[name]
    elif name in ('train_model', 'save_model'):
        from .train import train_model, save_model
        return locals()[name]
    elif name in ('evaluate_model', 'get_predictions'):
        from .evaluate import evaluate_model, get_predictions
        return locals()[name]
    elif name == 'ASLInferenceService':
        from .inference import ASLInferenceService
        return ASLInferenceService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
