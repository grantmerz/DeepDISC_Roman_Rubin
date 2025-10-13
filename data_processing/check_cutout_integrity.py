"""
Check all Roman data cutouts for empty or NaN values.
This script scans ~157k cutouts (512x512x3 float32 arrays) and identifies any
with empty data or NaN values in any filter band.
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# extract ra_dec from directory names
RA_DEC_PATTERN = re.compile(r'(\d+\.\d+)_(\-?\d+\.\d+)')

def check_cutout_for_issues(cutout_path):
    """
    Check a single cutout for empty or NaN values.
    
    Args:
        cutout_path: Path to the .npy cutout file
        
    Returns:
        dict with keys:
            'has_issues': bool
            'empty_bands': list of band idxs that are empty (all zeros)
            'nan_bands': list of band idxs that contain NaN values
            'shape': tuple of cutout shape
    """
    try:
        cutout = np.load(cutout_path, mmap_mode='r')
        result = {
            'has_issues': False,
            'empty_bands': [],
            'nan_bands': [],
            'shape': cutout.shape,
            'all_zero': False
        }
        if np.all(cutout == 0):
            result['has_issues'] = True
            result['all_zero'] = True
            result['empty_bands'] = [0, 1, 2]
            return result
        
        # do each band separately (Y106, J129, H158)
        band_names = ['Y106', 'J129', 'H158']
        for band_idx in range(cutout.shape[0]):
            band_data = cutout[band_idx]
            
            if np.any(np.isnan(band_data)):
                result['has_issues'] = True
                result['nan_bands'].append(band_idx)
            
            if np.all(band_data == 0):
                result['has_issues'] = True
                result['empty_bands'].append(band_idx)
        
        return result
        
    except Exception as e:
        return {
            'has_issues': True,
            'error': str(e),
            'empty_bands': [],
            'nan_bands': [],
            'shape': None
        }

def find_all_cutouts(base_dir):
    """
    Generate all expected cutout filenames based on tile directories and a fixed number of cutouts.
    
    Args:
        base_dir: Base directory containing ra_dec subdirectories
        
    Returns:
        list of all cutout file paths
    """
    print("Generating expected cutout filenames (225 cutouts per tile)...")
    base_path = Path(base_dir)
    all_cutout_files = []
    # finding all tile directories
    tile_dirs = [d for d in base_path.iterdir() if d.is_dir() and RA_DEC_PATTERN.search(d.name)]
    print(f"Found {len(tile_dirs)} tile directories. Ex: {tile_dirs[:3]}")
    if not tile_dirs:
        print("Warning: No tile directories found. Searching for all cutouts recursively.")
        return sorted(base_path.rglob('c*_*.npy'))

    for tile_dir in tqdm(tile_dirs, desc="Generating filenames"):
        tile_name = tile_dir.name
        for i in range(225):
            cutout_filename = f"c{i}_{tile_name}.npy"
            all_cutout_files.append(tile_dir / cutout_filename)
            
    return all_cutout_files

def main(base_dir, output_file, max_workers):
    """
    Main function to check all cutouts and save corrupt data info.
    
    Args:
        base_dir: Base directory containing cutouts
        output_file: Path to save the corruption report
    """
    print(f"Scanning for cutout files in {base_dir}...")
    cutout_files = find_all_cutouts(base_dir)
    
    print(f"Found {len(cutout_files)} cutout files")
    print(f"Expected shape: (3, 512, 512)")
    print(f"Expected dtype: float32")
    print(f"Bands: Y106 (0), J129 (1), H158 (2)")
    print(f"Using {max_workers} worker processes.")
    print("\nStarting integrity check...\n")    
    corrupt_cutouts = {}
    stats = {
        'total_cutouts': len(cutout_files),
        'corrupt_cutouts': 0,
        'cutouts_with_nans': 0,
        'cutouts_with_empty_bands': 0,
        'cutouts_all_zero': 0,
        'band_nan_counts': {'Y106': 0, 'J129': 0, 'H158': 0},
        'band_empty_counts': {'Y106': 0, 'J129': 0, 'H158': 0}
    }
    
    band_names = ['Y106', 'J129', 'H158']
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # dict to map futures to their file paths
        future_to_path = {executor.submit(check_cutout_for_issues, path): path for path in cutout_files}
        for future in tqdm(as_completed(future_to_path), total=len(cutout_files), desc="Checking cutouts", unit="cutout"):
            cutout_path = future_to_path[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'has_issues': True, 'error': str(exc)}

            if result['has_issues']:
                stats['corrupt_cutouts'] += 1
                
                cutout_info = {
                    'filename': cutout_path.name,
                    'full_path': str(cutout_path),
                    'shape': result.get('shape'),
                }
                
                if 'error' in result:
                    cutout_info['error'] = result['error']
                else:
                    if result.get('all_zero'):
                        cutout_info['all_zero'] = True
                        stats['cutouts_all_zero'] += 1
                    
                    if result.get('nan_bands'):
                        cutout_info['nan_bands'] = [band_names[i] for i in result['nan_bands']]
                        stats['cutouts_with_nans'] += 1
                        for band_idx in result['nan_bands']:
                            stats['band_nan_counts'][band_names[band_idx]] += 1
                    
                    if result.get('empty_bands') and not result.get('all_zero'):
                        cutout_info['empty_bands'] = [band_names[i] for i in result['empty_bands']]
                        stats['cutouts_with_empty_bands'] += 1
                        for band_idx in result['empty_bands']:
                            stats['band_empty_counts'][band_names[band_idx]] += 1
                
                corrupt_cutouts[cutout_path.name] = cutout_info

    report = {
        'summary': stats,
        'corrupt_cutouts': corrupt_cutouts
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print("\n" + "="*60)
    print("CUTOUT INTEGRITY CHECK SUMMARY")
    print("="*60)
    print(f"Total cutouts checked:        {stats['total_cutouts']:>10,}")
    print(f"Corrupt cutouts found:        {stats['corrupt_cutouts']:>10,} ({stats['corrupt_cutouts']/stats['total_cutouts']*100:.2f}%)")
    print(f"  - All zero cutouts:         {stats['cutouts_all_zero']:>10,}")
    print(f"  - Cutouts with NaN values:  {stats['cutouts_with_nans']:>10,}")
    print(f"  - Cutouts with empty bands: {stats['cutouts_with_empty_bands']:>10,}")
    print("\nPer-band statistics:")
    print(f"  Y106 (band 0):")
    print(f"    - NaN count:              {stats['band_nan_counts']['Y106']:>10,}")
    print(f"    - Empty count:            {stats['band_empty_counts']['Y106']:>10,}")
    print(f"  J129 (band 1):")
    print(f"    - NaN count:              {stats['band_nan_counts']['J129']:>10,}")
    print(f"    - Empty count:            {stats['band_empty_counts']['J129']:>10,}")
    print(f"  H158 (band 2):")
    print(f"    - NaN count:              {stats['band_nan_counts']['H158']:>10,}")
    print(f"    - Empty count:            {stats['band_empty_counts']['H158']:>10,}")
    print(f"\nDetailed report saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check Roman data cutouts for empty or NaN values"
    )
    parser.add_argument(
        "--base_dir", 
        type=str,
        default="/u/yse2/roman_data/truth/",
        help="Base directory containing ra_dec subdirectories with cutouts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/u/yse2/roman_cutout_integrity_report.json",
        help="Output file path for the corruption report (JSON format)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Maximum number of worker processes to use for checking cutouts"
    )
    args = parser.parse_args()
    main(args.base_dir, args.output, args.max_workers)
