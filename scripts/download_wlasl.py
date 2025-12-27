
import requests
import zipfile
import os
import sys
from tqdm import tqdm

DATA_URL = "https://drive.upm.es/s/AsErgLlRn5WJ0zM/download"
DEST_DIR = os.path.join(os.path.dirname(__file__), '../data')
ZIP_FILE = os.path.join(DEST_DIR, 'wlasl_landmarks.zip')

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return False
    return True

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    if os.path.exists(ZIP_FILE):
        print(f"File {ZIP_FILE} already exists. Skipping download.")
    else:
        print(f"Downloading WLASL landmarks to {ZIP_FILE}...")
        if not download_file(DATA_URL, ZIP_FILE):
            sys.exit(1)
            
    extract_zip(ZIP_FILE, DEST_DIR)
    
    # List contents to understand structure
    print("\nDataset contents:")
    for root, dirs, files in os.walk(DEST_DIR):
        level = root.replace(DEST_DIR, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        # Limit printed files
        for f in files[:5]:
            print('{}{}'.format(subindent, f))
        if len(files) > 5:
            print('{}(... and {} more)'.format(subindent, len(files)-5))

if __name__ == "__main__":
    main()
