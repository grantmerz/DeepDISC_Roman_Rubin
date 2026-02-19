import json, os
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D


def create_cutouts(patch_id, patch_wcs, cutout_size=128):
    patch_cutout = f'full_p{patch_id}.npy'
    full_img_data = np.load(patch_cutout)
    
    # read in patch catalog and then turn coords into SkyCoords
    full_cat = pd.read_json(f'full_p{patch_id}.json', orient='records')
    coords = SkyCoord(ra=full_cat['coord_ra'].values * u.deg,
                      dec=full_cat['coord_dec'].values * u.deg)
    
    coadd_size = full_img_data.shape[1:] 
    overlap_pixels = 200
    
    # only use the core region avoiding 200px overlap on each edge
    usable_width = coadd_size[1] - 2 * overlap_pixels  # 3000
    usable_height = coadd_size[0] - 2 * overlap_pixels  # 3000
    start_x, start_y = overlap_pixels, overlap_pixels  # 200, 200
    
    nx_cutouts = usable_width // cutout_size # 3000 // 128 = 23
    ny_cutouts = usable_height // cutout_size # 3000 // 128 = 23
        
    # calc spacing to distribute cutouts evenly
    if nx_cutouts > 1:
        x_spacing = (usable_width - cutout_size) // (nx_cutouts - 1)
    else:
        x_spacing = 0
    
    if ny_cutouts > 1:
        y_spacing = (usable_height - cutout_size) // (ny_cutouts - 1)
    else:
        y_spacing = 0
    
    print(f"Image size: {coadd_size[1]}x{coadd_size[0]}")
    print(f"Usable region: {usable_width}x{usable_height} (starting at {start_x},{start_y})")
    print(f"Cutouts: {nx_cutouts}x{ny_cutouts} = {nx_cutouts * ny_cutouts} total")
    print(f"Spacing: x={x_spacing}, y={y_spacing}")
    
    # List to store metadata for all cutouts
    cutout_md = []
    counter = 0
    for i in range(ny_cutouts):
        for j in range(nx_cutouts):
            # cutout center pos
            if ny_cutouts == 1:
                y_center = start_y + cutout_size // 2
            else:
                y_center = start_y + cutout_size // 2 + i * y_spacing
            
            if nx_cutouts == 1:
                x_center = start_x + cutout_size // 2
            else:
                x_center = start_x + cutout_size // 2 + j * x_spacing
            
            full_cutout_path = f'p{patch_id}/full_c{counter}.npy'
            
            if not os.path.exists(full_cutout_path):
                raw_cutout_u = Cutout2D(full_img_data[0], position=(x_center, y_center), 
                                        size=cutout_size, wcs=patch_wcs, mode='partial', fill_value=0)
                raw_cutout_g = Cutout2D(full_img_data[1], position=(x_center, y_center), 
                                        size=cutout_size, wcs=patch_wcs, mode='partial', fill_value=0)
                raw_cutout_r = Cutout2D(full_img_data[2], position=(x_center, y_center), 
                                        size=cutout_size, wcs=patch_wcs, mode='partial', fill_value=0)
                raw_cutout_i = Cutout2D(full_img_data[3], position=(x_center, y_center), 
                                        size=cutout_size, wcs=patch_wcs, mode='partial', fill_value=0)
                raw_cutout_z = Cutout2D(full_img_data[4], position=(x_center, y_center), 
                                        size=cutout_size, wcs=patch_wcs, mode='partial', fill_value=0)
                raw_cutout_y = Cutout2D(full_img_data[5], position=(x_center, y_center), 
                                        size=cutout_size, wcs=patch_wcs, mode='partial', fill_value=0)
                full_raw_cutout = np.stack((
                    raw_cutout_u.data,
                    raw_cutout_g.data,
                    raw_cutout_r.data,
                    raw_cutout_i.data,
                    raw_cutout_z.data,
                    raw_cutout_y.data
                ))
                np.save(full_cutout_path, full_raw_cutout)
                cutout_wcs = raw_cutout_i.wcs
            else:
                # If cutout already exists, recreate WCS for metadata
                raw_cutout_i = Cutout2D(full_img_data[3], position=(x_center, y_center), 
                                        size=cutout_size, wcs=patch_wcs, mode='partial', fill_value=0)
                cutout_wcs = raw_cutout_i.wcs
            
            # Convert catalog RA/Dec to pixels using cutout wcs
            patch_xpix, patch_ypix = cutout_wcs.world_to_pixel(coords)
            # Check if within cutout bounds (0 to cutout_size)
            in_cutout = ((patch_xpix >= 0) & (patch_xpix < cutout_size) & 
                         (patch_ypix >= 0) & (patch_ypix < cutout_size))
            cutout_xpix, cutout_ypix = patch_xpix[in_cutout], patch_ypix[in_cutout]
            n_catalog_objects = len(cutout_xpix)
            overlaps_with_hst = n_catalog_objects > 0
            # Save metadata for this cutout
            metadata = {
                "file_name": full_cutout_path,
                "image_id": counter,
                "height": cutout_size,
                "width": cutout_size,
                "patch": patch_id,
                "tract": 5063,
                "cutout_id": counter,
                "overlaps_with_hst": overlaps_with_hst,
                "wcs": cutout_wcs.to_header_string(),
                "annotations": []
            }
            cutout_md.append(metadata)
            # debug info for first few cutouts
            if counter < 3:
                print(f"  Cutout {counter}: center=({x_center}, {y_center}), "
                      f"objects={n_catalog_objects}")
            
            counter += 1
    
    print(f"Created and saved {counter} cutouts for patch {patch_id}!")    
    return cutout_md

patch_ids = [24, 25, 34, 35, 44, 45]
all_cutout_md = []

for patch_id in patch_ids:
    wcs = WCS(all_wcs[f'{patch_id}'])
    patch_md = create_cutouts(patch_id, wcs)
    all_cutout_md.extend(patch_md)
    print(f"Total cutouts so far: {len(all_cutout_md)}\n")

# Save all metadata to a single file
with open('test.json', 'w') as f:
    json.dump(all_cutout_md, f, indent=2)

print(f"DONE: Saved metadata for {len(all_cutout_md)} cutouts to test.json")