
import cv2
import numpy as np
import os
import argparse
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from feature_extractor import FeatureExtractor

def collect_data(action_name, num_sequences=30, sequence_length=30, output_dir='../data'):
    """
    Collects data for a specific action/sign.
    Args:
        action_name: The name of the sign (e.g., "hello", "thank you").
        num_sequences: How many examples to collect.
        sequence_length: Frames per example.
        output_dir: Where to save the data.
    """
    
    data_path = os.path.join(output_dir, action_name)
    os.makedirs(data_path, exist_ok=True)
    
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting collection for action: '{action_name}'")
    print(f"Collecting {num_sequences} sequences of {sequence_length} frames each.")
    print("Press 'q' to quit early.")
    
    # Wait a bit before starting
    print("Get ready! Starting in 3 seconds...")
    time.sleep(3)

    for sequence in range(num_sequences):
        frames = []
        print(f"Recording sequence {sequence + 1}/{num_sequences}...")
        
        for frame_num in range(sequence_length):
            ret, image = cap.read()
            if not ret:
                print("Error reading frame.")
                break

            # Feature extraction
            landmarks = extractor.extract_landmarks(image)
            frames.append(landmarks)
            
            # Visual feedback
            cv2.putText(image, f'{action_name}: {sequence + 1}/{num_sequences} | Frame: {frame_num}', (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Save the sequence
        npy_path = os.path.join(data_path, f"{sequence}.npy")
        np.save(npy_path, np.array(frames))
        
        # Brief pause between sequences
        print("Sequence saved. Next in 1 second...")
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection for '{action_name}' complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Data Collection Tool")
    parser.add_argument("--action", type=str, required=True, help="Name of the sign to collect")
    parser.add_argument("--count", type=int, default=30, help="Number of sequences to collect")
    parser.add_argument("--len", type=int, default=30, help="Number of frames per sequence")
    args = parser.parse_args()
    
    # Ensure data directory exists
    base_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(base_data_dir, exist_ok=True)
    
    collect_data(args.action, args.count, args.len, base_data_dir)
