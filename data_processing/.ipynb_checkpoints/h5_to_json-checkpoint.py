import pandas as pd
import numpy as np
import argparse
import multiprocessing as mp
from functools import partial
import os
from tqdm import tqdm
import json
import gc

def process_chunk(chunk_data, temp_dir):
    """
    Process a single chunk and save it to a temporary file.
    
    Parameters:
    -----------
    chunk_data : tuple
        Contains (chunk_id, start_idx, end_idx, h5_file, key)
    temp_dir : str
        Directory for temporary files
    """
    chunk_id, start, end, h5_file, key = chunk_data
    try:
        # Read the chunk from HDF5
        chunk = pd.read_hdf(h5_file, key=key, start=start, stop=end)
        
        # Save to temporary file
        temp_file = os.path.join(temp_dir, f'chunk_{chunk_id}.json')
        chunk.to_json(temp_file, orient='records')
        
        # Clear memory
        del chunk
        gc.collect()
        
        return temp_file
    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {str(e)}")
        return None

def merge_json_files(temp_files, output_file, chunk_size=1000):
    """
    Merge temporary JSON files into final output file while managing memory.
    """
    with open(output_file, 'w') as outfile:
        outfile.write('[')
        first_chunk = True
        
        for temp_file in tqdm(temp_files, desc="Merging files"):
            if temp_file is None:
                continue
                
            try:
                with open(temp_file, 'r') as infile:
                    data = json.load(infile)
                    
                    # Write records in smaller chunks
                    for i in range(0, len(data), chunk_size):
                        chunk = data[i:i + chunk_size]
                        
                        if not first_chunk:
                            outfile.write(',')
                        first_chunk = False
                        
                        json.dump(chunk, outfile)
                        del chunk
                        
                    del data
                    gc.collect()
                
                # Remove temporary file
                os.remove(temp_file)
                
            except Exception as e:
                print(f"Error processing {temp_file}: {str(e)}")
                continue
                
        outfile.write(']')

def convert_h5_to_json(h5_file, output_file, chunk_size=10000, num_processes=None):
    """
    Convert HDF5 file to JSON format using multiprocessing with improved memory management.
    """
    # Create temporary directory
    temp_dir = output_file + '_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    # Get total size and key from HDF5 file
    with pd.HDFStore(h5_file, mode='r') as store:
        key = store.keys()[0]
        total_rows = store.get_storer(key).nrows
    
    # Calculate chunks
    chunks = []
    for i, start in enumerate(range(0, total_rows, chunk_size)):
        end = min(start + chunk_size, total_rows)
        chunks.append((i, start, end, h5_file, key))
    
    print(f"Processing {len(chunks)} chunks using {num_processes} processes...")
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        temp_files = list(tqdm(
            pool.imap(partial(process_chunk, temp_dir=temp_dir), chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
    
    # Merge temporary files
    print("Merging temporary files...")
    merge_json_files(temp_files, output_file)
    
    # Clean up
    try:
        os.rmdir(temp_dir)
    except:
        print(f"Warning: Could not remove temporary directory: {temp_dir}")
    
    print(f"Successfully converted {h5_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 file to JSON format using multiprocessing')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output JSON file')
    parser.add_argument('--chunk-size', type=int, default=50000,
                      help='Number of rows to process at once')
    parser.add_argument('--processes', type=int, default=None,
                      help='Number of processes to use (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    # If running on a cluster with SLURM, use the allocated CPUs
    args.processes = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    convert_h5_to_json(args.input, args.output, args.chunk_size, args.processes)

if __name__ == "__main__":
    main()