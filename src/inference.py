
import cv2
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from feature_extractor import FeatureExtractor
from classifier import ASLClassifier

class ASLInferenceService:
    def __init__(self, model_path, classes, sequence_length=30):
        self.classes = classes
        self.sequence_length = sequence_length
        self.extractor = FeatureExtractor()
        
        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ASLClassifier(input_size=42, num_classes=len(classes)).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model file not found at {model_path}. Running with uninitialized weights.")

        self.sequence_buffer = []

    def predict(self, image):
        """
        Takes an image, extracts features, updates buffer, and returns prediction.
        """
        landmarks = self.extractor.extract_landmarks(image)
        # Check if landmarks are all zeros (no hand)
        if np.all(landmarks == 0):
             return None, 0.0

        self.sequence_buffer.append(landmarks)
        
        # Keep only the last 'sequence_length' frames
        self.sequence_buffer = self.sequence_buffer[-self.sequence_length:]
        
        if len(self.sequence_buffer) == self.sequence_length:
            input_seq = np.array([self.sequence_buffer]) # Add batch dim: (1, 30, 42)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                return self.classes[predicted_idx.item()], confidence.item()
        
        return None, 0.0

if __name__ == "__main__":
    # IPNHand Classes (13 distinct gestures)
    classes = ['0.0', '1.0', '10.0', '11.0', '12.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
    
    # Path relative to this script
    model_path = os.path.join(os.path.dirname(__file__), '../asl_model.pth')
    
    service = ASLInferenceService(model_path, classes)
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        label, conf = service.predict(frame)
        
        # Display
        text = f"Pred: {label} ({conf:.2f})" if label else "Waiting for buffer..."
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ASL Inference', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
