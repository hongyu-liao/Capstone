# Code to count the number of images in a Docling-generated JSON file (looks for 'pictures' key)

import json

# Path to your JSON file
json_file_path = r"C:\Users\Hongyu\OneDrive - Northwestern University\NU\Capstone\output_lmstudio_conversion\1-s2.0-S1385110124000054-main-google_gemma-3-12b-it-gguf_nlp_ready.json"

# Try to open and load the JSON file using utf-8 encoding
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except UnicodeDecodeError:
    print("UnicodeDecodeError: Could not decode the file with utf-8 encoding. Try opening with a different encoding (e.g., 'gbk').")
    data = None
except Exception as e:
    print(f"An error occurred while loading the JSON file: {e}")
    data = None

# Check for the 'pictures' key, which is used in Docling-generated JSON files
if isinstance(data, dict) and 'pictures' in data and isinstance(data['pictures'], list):
    num_images = len(data['pictures'])
    print(f"Number of images in the JSON file: {num_images}")
elif data is not None:
    print("Could not find a list of images under the 'pictures' key in the JSON file.")
