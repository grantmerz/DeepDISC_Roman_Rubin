import os
import json

def load_and_combine_json_files(directory_path, output_file):
    combined_data = []
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json') and filename.startswith('dc2'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                combined_data.extend(data)
    
    # Save the combined data into a single JSON file
    with open(output_file, 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)

# Example usage
# directory_path = '../roman_data/annotations'
# output_file = '../roman_data/annotations/all_metadata.json'
directory_path = 'lsst_data/annotations'
output_file = 'lsst_data/annotations/all_metadata.json'
load_and_combine_json_files(directory_path, output_file)
