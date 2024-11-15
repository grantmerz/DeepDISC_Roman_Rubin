import re, os, glob
import json
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import pandas as pd
import numpy as np
from detectron2.structures import BoxMode
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Attach magnitudes to annotations from truth catalogs")
    parser.add_argument('--ann-file', type=str, help='File containing the annotations you\'d like to attach the magnitudes to')
    parser.add_argument('--dir', type=str, help='Directory containing all the annotation files you\'d like to attach mags to')
    args = parser.parse_args()
    return args

def add_mags(old_ann_filename):
    with open(old_ann_filename, 'r') as f:
        data = json.load(f)
    updated_data = []
    for i, d in enumerate(data):
        file_name = d['file_name']
        truth_patch_dir = (re.search(r'truth/(.+)/', file_name)).group(1)
        truth_info_filename = f'roman_data/truth/{truth_patch_dir}/{truth_patch_dir}_info.json'
        with open(truth_info_filename, 'r') as f:
            all_truth_info = json.load(f)
        for truth_img in all_truth_info:
            if truth_img['file_name'] == file_name:
                wcs = WCS(truth_img['wcs'])
                xs = []
                ys = []
                # extract the centers of the gt boxes directly from annotations
                anns = d['annotations']
                for a in anns:
                    box = a['bbox']
                    transformed_box = BoxMode.convert(box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    w = transformed_box[2] - transformed_box[0]
                    h = transformed_box[3] - transformed_box[1]
                    x = transformed_box[0] + w // 2
                    y = transformed_box[1] + h // 2
                    xs.append(x)
                    ys.append(y)
                coords = wcs.pixel_to_world(xs, ys)
                # reading in all objs present in patch
                all_objs_df = pd.read_json(truth_img['objects_info'])
                # find closest object match to pixel coord
                obj_coords = SkyCoord(ra = all_objs_df['ra'], dec = all_objs_df['dec'], unit='deg')
                idx, d2d, d3d = coords.match_to_catalog_sky(obj_coords)

                # find that matched object's mags
                matched_objs = all_objs_df.iloc[idx]
                mag_F184 = matched_objs['mag_F184']
                mag_Y106 = matched_objs['mag_Y106']
                mag_J129 = matched_objs['mag_J129']
                mag_H158 = matched_objs['mag_H158']

                for idx, ann in enumerate(anns):
                    ann['fmag'] = mag_F184.iloc[idx]
                    ann['ymag'] = mag_Y106.iloc[idx]
                    ann['jmag'] = mag_J129.iloc[idx]
                    ann['hmag'] = mag_H158.iloc[idx]
                d_new = d.copy()
                d_new['annotations'] = anns
                updated_data.append(d_new)
                break
    new_ann_filename = old_ann_filename[:-5] + "_updated.json"
    with open(new_ann_filename, 'w') as f:
        json.dump(updated_data, f)

def main(args):
    if not args.ann_file and not args.dir:
        print("Please provide either an annotation file or directory that has annotations")
    if args.ann_file:
        old_ann_filename = args.ann_file
        add_mags(old_ann_filename)
    else:
        path = args.dir
        pattern = os.path.join(path, 'dc2*.json')
        ann_files = glob.glob(pattern)
        for filename in ann_files:
            add_mags(filename)
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)