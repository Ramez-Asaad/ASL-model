
import csv
import ast
import sys

csv_path = 'data/inputs/WLASL100/MediaPipe_data/x-y/WLASL100_train.csv'

def inspect_more():
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            rh_idx = header.index('RHx0')
            lh_idx = header.index('LHx0')
            lbl_idx = header.index('labels')
            
            # Read first 100 rows to see unique labels
            unique_labels = set()
            
            for i in range(50):
                try:
                    row = next(reader)
                except StopIteration:
                    break
                    
                lbl_raw = row[lbl_idx]
                # It's a string "[0,0,0...]"
                lbl_val = ast.literal_eval(lbl_raw)
                # Take the first one, assuming frame-consistency
                if len(lbl_val) > 0:
                    unique_labels.add(lbl_val[0])
                
                if i == 0:
                    lh_val = ast.literal_eval(row[lh_idx])
                    print(f"Row 0 LHx0: {lh_val[:5]}")
                    print(f"Row 0 Label: {lbl_val[0]} (from list of len {len(lbl_val)})")
            
            print(f"Unique labels in first 50 rows: {unique_labels}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_more()
