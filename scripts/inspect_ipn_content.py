
import csv
import ast
import sys

csv_path = 'data/inputs/IPNHand/MediaPipe_data/x-y/IPNHand_train.csv'

def inspect_ipn_content():
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            row = next(reader)
            
            # Inspect first column 'x0'
            print(f"Column 'x0' raw type: {type(row[0])}")
            print(f"Column 'x0' raw content (first 50 chars): {row[0][:50]}...")
            
            try:
                val = ast.literal_eval(row[0])
                print(f"Parsed type: {type(val)}")
                if isinstance(val, list):
                    print(f"List len: {len(val)}")
                    print(f"First 5 elements: {val[:5]}")
            except Exception as e:
                print(f"Parse error: {e}")

    except Exception as e:
         print(f"Error: {e}")

if __name__ == "__main__":
    inspect_ipn_content()
