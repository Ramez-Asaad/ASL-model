"""
Real-time inference for ASL Detection.
"""

import cv2
import numpy as np

from .config import IMAGE_SIZE, IDX_TO_LABEL
from .data_loader import preprocess_single_image
from .model import load_model


class ASLInferenceService:
    """Real-time ASL gesture recognition service."""
    
    def __init__(self, model_path):
        """
        Initialize the inference service.
        
        Args:
            model_path: Path to trained .h5 model
        """
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        print("Model loaded successfully.")
    
    def predict(self, image):
        """
        Predict ASL gesture from an image.
        
        Args:
            image: BGR image from cv2
        
        Returns:
            (label, confidence) tuple
        """
        # Preprocess
        processed = preprocess_single_image(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        
        # Get top prediction
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        label = IDX_TO_LABEL[predicted_idx]
        
        return label, confidence
    
    def predict_top_k(self, image, k=3):
        """
        Get top-k predictions.
        
        Args:
            image: BGR image from cv2
            k: Number of top predictions to return
        
        Returns:
            List of (label, confidence) tuples
        """
        processed = preprocess_single_image(image)
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top-k indices
        top_indices = np.argsort(predictions)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((IDX_TO_LABEL[idx], predictions[idx]))
        
        return results


def run_webcam_inference(model_path):
    """
    Run real-time inference from webcam.
    
    Args:
        model_path: Path to trained model
    """
    service = ASLInferenceService(model_path)
    
    cap = cv2.VideoCapture(0)
    print("\n=== ASL Gesture Recognition ===")
    print("Press 'q' to quit\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get prediction
        label, confidence = service.predict(frame)
        
        # Color based on confidence
        if confidence >= 0.7:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw UI
        text = f"{label}: {confidence*100:.0f}%"
        
        # Background rectangle
        cv2.rectangle(frame, (5, 5), (200, 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (200, 50), color, 2)
        
        # Text
        cv2.putText(frame, text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw ROI hint (center square)
        h, w = frame.shape[:2]
        roi_size = min(h, w) // 2
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        cv2.rectangle(frame, (x1, y1), (x1 + roi_size, y1 + roi_size), 
                     (100, 100, 100), 2)
        cv2.putText(frame, "Show gesture here", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        cv2.imshow('ASL Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import os
    model_path = os.path.join(os.path.dirname(__file__), '../models/asl_cnn.h5')
    run_webcam_inference(model_path)
