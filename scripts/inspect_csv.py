
import pandas as pd
import ast
import numpy as np

csv_path = 'data/inputs/WLASL100/MediaPipe_data/x-y/WLASL100_train.csv'

def inspect():
    df = pd.read_csv(csv_path, nrows=5)
    print("Columns:", df.columns.tolist())
    
    # Check first row
    row = df.iloc[0]
    print(f"\nLabel: {row['labels']}")
    
    # Inspect one cell (RHx0)
    rhx0_str = row['RHx0']
    print(f"\nRHx0 type: {type(rhx0_str)}")
    print(f"RHx0 content (first 50 chars): {rhx0_str[:50]}...")
    
    # Parse it
    try:
        rhx0_val = ast.literal_eval(rhx0_str)
        print(f"Parsed RHx0 length: {len(rhx0_val)}")
        print(f"First 5 val: {rhx0_val[:5]}")
    except Exception as e:
        print(f"Parse error: {e}")

if __name__ == "__main__":
    inspect()
