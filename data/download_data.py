import os
import requests
from zipfile import ZipFile

# Example: a public dataset URL (replace with yours if you have a specific dataset)
DATASET_URL = "https://github.com/karpathy/char-rnn/raw/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, save_path):
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"Saved dataset to {save_path}")

if __name__ == "__main__":
    save_path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    download_file(DATASET_URL, save_path)
