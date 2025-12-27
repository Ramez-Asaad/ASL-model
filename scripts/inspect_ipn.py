
import csv
import ast
import sys

csv_path = 'data/inputs/IPNHand/MediaPipe_data/x-y/IPNHand_train.csv'

def inspect_ipn():
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Header length: {len(header)}")
            print(f"Sample header columns: {header[:10]}")
            
            # Check for same columns
            rhx0_present = 'RHx0' in header
            labels_present = 'labels' in header
            print(f"Has RHx0: {rhx0_present}, Has labels: {labels_present}")
            
            # Read all rows to count samples/classes
            classes = set()
            count = 0
            for row in reader:
                count += 1
                if labels_present:
                    lbl_idx = header.index('labels')
                    lbl_raw = row[lbl_idx]
                    try:
                        lbl_val = ast.literal_eval(lbl_raw)
                        if lbl_val:
                            classes.add(lbl_val[0])
                    except:
                        pass
            
            print(f"Total rows: {count}")
            print(f"Total classes: {len(classes)}")
            if len(classes) > 0:
                print(f"Avg samples/class: {count / len(classes):.2f}")
                
    except FileNotFoundError:
        print("File not found.")

if __name__ == "__main__":
    inspect_ipn()
