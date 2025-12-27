
import numpy as np
import sys

file_path = 'data/processed/class_47/755.npy'

def inspect_npy():
    try:
        data = np.load(file_path)
        print(f"Shape: {data.shape}")
        print(f"Type: {data.dtype}")
        print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
        
        # Check for zeros
        zeros = np.sum(data == 0)
        total = data.size
        print(f"Zeros: {zeros} / {total} ({100*zeros/total:.2f}%)")
        
        # Print a non-zero sample if exists
        if zeros < total:
            print("Non-zero sample (first 10 elements where not 0):")
            flat = data.flatten()
            non_zeros = flat[flat != 0]
            print(non_zeros[:10])
        else:
            print("ALL ZEROS!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_npy()
