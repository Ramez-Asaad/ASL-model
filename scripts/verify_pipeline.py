
import os
import shutil
import numpy as np
import sys
import torch

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../data_test')
model_path = os.path.join(base_dir, '../model_test.pth')

sys.path.append(os.path.join(base_dir, '../src'))
from classifier import ASLClassifier
from feature_extractor import FeatureExtractor
# We can't import collect_data directly as it uses cv2.VideoCapture which fails, 
# but we can simulate data creation.

def generate_dummy_data():
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    classes = ['class_a', 'class_b']
    
    for i, label in enumerate(classes):
        os.makedirs(os.path.join(data_dir, label), exist_ok=True)
        # Create 10 sequences per class
        for j in range(10):
            # 30 frames, 63 features
            # Give them slightly different means so they are separable
            data = np.random.randn(30, 63) + (i * 5) 
            np.save(os.path.join(data_dir, label, f"{j}.npy"), data)
            
    print("Dummy data generated.")

def test_training():
    print("Testing training script logic...")
    # Manually calling train logic to avoid import issues or argparse
    from train_model import train_model
    train_model(data_dir, model_path, epochs=5)
    
    if os.path.exists(model_path):
        print("Training successful, model file created.")
    else:
        print("Training failed, no model file.")
        sys.exit(1)

def test_inference():
    print("Testing inference logic...")
    from inference import ASLInferenceService
    
    # Mock Feature Extractor to return random data instead of using MediaPipe/CV2
    # because we don't have a screen/camera here
    real_extractor = FeatureExtractor
    
    class MockExtractor:
        def extract_landmarks(self, image):
            return np.random.randn(63)
            
    # Swap it out for testing
    import inference
    inference.FeatureExtractor = MockExtractor
    
    service = inference.ASLInferenceService(model_path, ['class_a', 'class_b'])
    
    # Run a few predictions
    for _ in range(35):
        pred, conf = service.predict(np.zeros((100,100,3), dtype=np.uint8))
        if pred:
            print(f"Prediction: {pred}, Conf: {conf}")

    print("Inference test complete.")

def main():
    try:
        generate_dummy_data()
        test_training()
        test_inference()
        print("\nALL VERIFICATION CHECKS PASSED.")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    main()
