
import csv
import ast
import sys

csv_path = 'data/inputs/WLASL100/MediaPipe_data/x-y/WLASL100_train.csv'

def inspect_csv():
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Header length: {len(header)}")
            print(f"Sample columns: {header[:10]} ... {header[-5:]}")
            
            row = next(reader)
            print(f"\nRow length: {len(row)}")
            
            # Find label index
            try:
                label_idx = header.index('labels')
                print(f"Label: {row[label_idx]}")
            except ValueError:
                print("Label column not found")

            # Inspect RHx0
            try:
                rhx0_idx = header.index('RHx0')
                rhx0_str = row[rhx0_idx]
                print(f"\nRHx0 raw (first 50): {rhx0_str[:50]}...")
                
                # It looks like a string representation of a list
                rhx0_val = ast.literal_eval(rhx0_str)
                print(f"RHx0 parsed type: {type(rhx0_val)}")
                print(f"RHx0 length (frames): {len(rhx0_val)}")
                print(f"RHx0 values: {rhx0_val[:5]}")
            except Exception as e:
                print(f"Error parsing RHx0: {e}")

    except FileNotFoundError:
        print("File not found.")

if __name__ == "__main__":
    inspect_csv()
