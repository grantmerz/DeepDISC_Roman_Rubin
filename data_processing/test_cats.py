import os
import gc
import json
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import multiprocessing as mp
from functools import partial
from astropy.table import Table

class CatalogProcessor:
    def __init__(self, survey_type, test_data_file):
        """
        Initialize catalog processor for either LSST or Roman data
        
        Parameters:
        -----------
        survey_type : str
            Either 'lsst' or 'roman'
        test_data_file : str
            Path to test data JSON file
        """
        self.survey_type = survey_type.lower()
        with open(test_data_file, 'r') as f:
            self.test_data = json.load(f)
        self.bounds = self._calculate_bounds()
    
    def _calculate_bounds(self):
        """Calculate spatial bounds from test data"""
        all_bounds = []
        truth_info_cache = {}
        
        for d in self.test_data:
            imid = d['image_id']
            subpatch = d['subpatch']
            
            if subpatch not in truth_info_cache:
                truth_info_filename = f'./lsst_data/truth/{subpatch}/{subpatch}_info.json'
                with open(truth_info_filename) as json_data:
                    truth_info_cache[subpatch] = json.load(json_data)
            
            truth_info = truth_info_cache[subpatch]
            entry = next(entry for entry in truth_info if entry['image_id'] == imid)
            wcs = WCS(entry['wcs'])
            height = entry['height']
            width = entry['width']
            
            corners = wcs.pixel_to_world([0, width-1, 0, width-1], 
                                       [0, 0, height-1, height-1])
            all_bounds.append(corners)
            
        return all_bounds
    
    def _get_coords_from_catalog(self, catalog, cat_type):
        """Get coordinates from catalog based on survey and catalog type"""
        if self.survey_type == 'lsst':
            if cat_type == 'det':
                return SkyCoord(ra=catalog['ra']*u.deg, 
                              dec=catalog['dec']*u.deg)
            else:  # truth
                return SkyCoord(ra=catalog['ra_truth_merged']*u.deg, 
                              dec=catalog['dec_truth_merged']*u.deg)
        else:  # roman
            if cat_type == 'det':
                return SkyCoord(ra=catalog['alphawin_j2000']*u.deg, 
                              dec=catalog['deltawin_j2000']*u.deg)
            else:  # truth
                return SkyCoord(ra=catalog['ra']*u.deg, 
                              dec=catalog['dec']*u.deg)
    
    def _get_duplicate_subset(self, catalog, cat_type):
        """Get column subset for deduplication based on survey and catalog type"""
        if self.survey_type == 'lsst':
            if cat_type == 'det':
                return ['ra', 'dec']
            else:
                return ['ra_truth_merged', 'dec_truth_merged']
        else:  # roman
            if cat_type == 'det':
                return ['alphawin_j2000', 'deltawin_j2000']
            else:
                return ['ra', 'dec']
    
    def spatial_filter(self, catalog, cat_type):
        """Filter catalog based on spatial bounds"""
        coords = self._get_coords_from_catalog(catalog, cat_type)
        
        mask = np.zeros(len(coords), dtype=bool)
        for bounds in self.bounds:
            ra_min, ra_max = np.min(bounds.ra), np.max(bounds.ra)
            dec_min, dec_max = np.min(bounds.dec), np.max(bounds.dec)
            
            mask |= ((coords.ra >= ra_min) & (coords.ra <= ra_max) &
                     (coords.dec >= dec_min) & (coords.dec <= dec_max))
        
        filtered_cat = catalog[mask]
        dedup_subset = self._get_duplicate_subset(filtered_cat, cat_type)
        return filtered_cat.drop_duplicates(subset=dedup_subset)
    
    def match_catalogs(self, det_catalog, truth_catalog, max_sep=0.5*u.arcsec):
        """Match detection catalog to truth catalog"""
        det_coords = self._get_coords_from_catalog(det_catalog, 'det')
        truth_coords = self._get_coords_from_catalog(truth_catalog, 'truth')
        
        # Match catalogs
        idx_det, idx_truth, d2d, _ = det_coords.search_around_sky(truth_coords, max_sep)
        
        if len(idx_det) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Create dictionary of matches with distances
        matches = {}
        for det_idx, truth_idx, dist in zip(idx_det, idx_truth, d2d):
            if det_idx not in matches:
                matches[det_idx] = []
            matches[det_idx].append((truth_idx, dist.to(u.arcsec).value))
        
        # Find best matches
        best_matches = {}
        used_truth = set()
        
        # Sort detections by best match distance
        det_best_distances = {det_idx: min(match_list, key=lambda x: x[1])[1] 
                            for det_idx, match_list in matches.items()}
        sorted_dets = sorted(det_best_distances.keys(), 
                           key=lambda x: det_best_distances[x])
        
        # Assign best matches
        for det_idx in sorted_dets:
            sorted_matches = sorted(matches[det_idx], key=lambda x: x[1])
            for truth_idx, dist in sorted_matches:
                if truth_idx not in used_truth:
                    best_matches[det_idx] = (truth_idx, dist)
                    used_truth.add(truth_idx)
                    break
        
        # Create matched catalogs
        matched_det_indices = list(best_matches.keys())
        matched_truth_indices = [best_matches[det_idx][0] for det_idx in matched_det_indices]
        
        matched_det_catalog = det_catalog.iloc[matched_det_indices].copy()
        matched_truth_catalog = truth_catalog.iloc[matched_truth_indices].copy()
        
        # Add match distances
        match_distances = [best_matches[det_idx][1] for det_idx in matched_det_indices]
        matched_det_catalog['match_distance'] = match_distances
        matched_truth_catalog['match_distance'] = match_distances
        
        return matched_det_catalog, matched_truth_catalog

def process_catalog(cat_file, truth_file, test_data_file, output_det, output_truth, 
                   survey_type='lsst', chunksize=50000):
    """Process catalog with spatial filtering and truth matching"""
    processor = CatalogProcessor(survey_type, test_data_file)
    
    if survey_type.lower() == 'lsst':
        # For LSST, load entire catalogs
        print("Processing LSST catalogs...")
        det_catalog = pd.read_json(cat_file)
        truth_catalog = pd.read_json(truth_file)
        
        # Spatial filtering
        print("Applying spatial filtering...")
        filtered_det = processor.spatial_filter(det_catalog, 'det')
        filtered_truth = processor.spatial_filter(truth_catalog, 'truth')
        
        # Match catalogs
        print("Matching catalogs...")
        matched_det, matched_truth = processor.match_catalogs(filtered_det, filtered_truth)
        
    else:  # Roman - use chunked processing
        print("Processing Roman catalogs...")
        # [Previous Roman processing code remains the same]
        # ... 
    
    # Save results
    if len(matched_det) > 0:
        print(f"Saving {len(matched_det)} matched pairs...")
        matched_det.to_json(output_det)
        matched_truth.to_json(output_truth)
        return matched_det, matched_truth
    
    print("No matches found.")
    return pd.DataFrame(), pd.DataFrame()