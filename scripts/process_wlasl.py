
import csv
import ast
import os
import numpy as np
import shutil

# Config
CSV_PATH = os.path.join(os.path.dirname(__file__), '../data/inputs/WLASL100/MediaPipe_data/x-y/WLASL100_train.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')

def process_wlasl():
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
        
        # Map indices
        # We need to construct columns in order: RHx0, RHy0, RHx1, RHy1 ... LHx0 ...
        # The CSV has RHx0, RHy0, etc. so we can just look them up.
        
        feature_cols = []
        # Right hand (21 landmarks)
        for i in range(21):
            feature_cols.append(f'RHx{i}')
            feature_cols.append(f'RHy{i}')
        # Left hand (21 landmarks)
        for i in range(21):
            feature_cols.append(f'LHx{i}')
            feature_cols.append(f'LHy{i}')
            
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
        for row in reader:
            try:
                # Get label
                label_list = ast.literal_eval(row[label_idx])
                if not label_list:
                    continue
                label = label_list[0] # Assume video-level label
                
                # Get lengths from first column to determine T
                first_col_data = ast.literal_eval(row[col_indices[0]])
                T = len(first_col_data)
                
                # Extract all features: Shape (84, T)
                features_T = []
                for idx in col_indices:
                    val_list = ast.literal_eval(row[idx])
                    # Pad or truncate if length mismatch (shouldn't happen in clean data)
                    if len(val_list) != T:
                        val_list = val_list[:T] # simplistic handling
                    features_T.append(val_list)
                
                # Transpose to (T, 84)
                # features_T is list of 84 lists, each len T.
                # data[t][f] = features_T[f][t]
                
                data_matrix = np.array(features_T).T
                
                # Save
                class_dir = os.path.join(OUTPUT_DIR, f"class_{label}")
                os.makedirs(class_dir, exist_ok=True)
                
                # Unique filename
                file_name = f"{count}.npy"
                np.save(os.path.join(class_dir, file_name), data_matrix)
                
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} samples...")
                    
            except Exception as e:
                print(f"Skipping row {count} due to error: {e}")
                
    print(f"Done. Processed {count} samples.")

if __name__ == "__main__":
    process_wlasl()
