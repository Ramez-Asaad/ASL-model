
import os
import os
import numpy as np

data_dir = 'data/processed'

def analyze_data():
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Total Classes: {len(classes)}")
    
    seq_lengths = []
    class_counts = []
    
    for label in classes:
        class_path = os.path.join(data_dir, label)
        files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        class_counts.append(len(files))
        
        for f in files:
            path = os.path.join(class_path, f)
            data = np.load(path)
            # Check non-zero length
            non_zero = data[~np.all(data == 0, axis=1)]
            seq_lengths.append(len(non_zero))

    seq_lengths = np.array(seq_lengths)
    class_counts = np.array(class_counts)
    
    print(f"Total Samples: {np.sum(class_counts)}")
    print(f"Samples per Class - Mean: {np.mean(class_counts):.2f}, Min: {np.min(class_counts)}, Max: {np.max(class_counts)}")
    print(f"Sequence Lengths (Frames) - Mean: {np.mean(seq_lengths):.2f}, Median: {np.median(seq_lengths)}, Min: {np.min(seq_lengths)}, Max: {np.max(seq_lengths)}")
    print(f"Number of very short sequences (< 5 frames): {np.sum(seq_lengths < 5)}")

if __name__ == "__main__":
    analyze_data()
