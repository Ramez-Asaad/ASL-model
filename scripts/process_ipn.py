
import csv
import ast
import os
import numpy as np
import shutil

# Config
CSV_PATH = os.path.join(os.path.dirname(__file__), '../data/inputs/IPNHand/MediaPipe_data/x-y/IPNHand_train.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data/processed_ipn')

def process_ipn():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Processing {CSV_PATH}...")
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # In inspect_ipn, header was ['x0', 'y0', ... 'x20', 'y20', ... 'labels', 'video_name']
        # We need x0...x20 and y0...y20 (42 cols)
        
        feature_cols = []
        for i in range(21):
            feature_cols.append(f'x{i}')
            feature_cols.append(f'y{i}')
            
        col_indices = []
        for col in feature_cols:
            try:
                col_indices.append(header.index(col))
            except ValueError:
                print(f"Missing column {col} in CSV!")
                return
        
        try:
            label_idx = header.index('labels')
        except ValueError:
            print("Missing labels column")
            return

        # Process rows
        count = 0
        total_rows = 0
        
        for row in reader:
            total_rows += 1
            try:
                # Get label
                label_list = ast.literal_eval(row[label_idx])
                if not label_list:
                    continue
                label = label_list[0] # Assume video-level label (first label)
                
                # Get lengths from first column to determine T
                # Inspect showed IPNHand stores lists in cells
                first_col_data = ast.literal_eval(row[col_indices[0]])
                T = len(first_col_data)
                
                if T == 0:
                    continue

                # Extract all features: Shape (42, T)
                features_T = []
                for idx in col_indices:
                    val_list = ast.literal_eval(row[idx])
                    if len(val_list) != T:
                        val_list = val_list[:T] 
                    features_T.append(val_list)
                
                # Transpose to (T, 42)
                data_matrix = np.array(features_T).T
                
                # Save
                class_dir = os.path.join(OUTPUT_DIR, f"{label}") # IPN labels are strings/names likely? Or ints? Inspect showed strings.
                os.makedirs(class_dir, exist_ok=True)
                
                # Unique filename
                file_name = f"{count}.npy"
                np.save(os.path.join(class_dir, file_name), data_matrix)
                
                count += 1
                if count % 200 == 0:
                    print(f"Processed {count} samples...")
                    
            except Exception as e:
                print(f"Skipping row due to error: {e}")
                
    print(f"Done. Processed {count}/{total_rows} samples.")

if __name__ == "__main__":
    process_ipn()
