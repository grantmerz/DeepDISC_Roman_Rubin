import json
import copy
import numpy as np
import os
from tqdm import tqdm
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from deepdisc.data_format.image_readers import DC2ImageReader, HSCImageReader, RomanImageReader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from deepdisc.astrodet.visualizer import ColorMode
from detectron2.structures import BoxMode
from deepdisc.astrodet.visualizer import Visualizer
from astropy.visualization import make_lupton_rgb
import multiprocessing as mp
from functools import partial

# load in all the metadata
all_metadata_file = './lsst_data/annotations-old/all_metadata.json'
with open(all_metadata_file, 'r') as f:
    all_metadata = json.load(f)
    
# load in all the train, val, test
train_file = './lsst_data/annotations-old/train.json'
test_file = './lsst_data/annotations-old/test.json'
val_file = './lsst_data/annotations-old/val.json'
with open(train_file, 'r') as f:
    trainmd = json.load(f)
with open(test_file, 'r') as f:
    testmd = json.load(f)
with open(val_file, 'r') as f:
    valmd = json.load(f)

reader = RomanImageReader()

red = np.array(colors.to_rgb('red'))*255
white = np.array(colors.to_rgb('white'))*255
blue = np.array(colors.to_rgb('blue'))*255
green = np.array(colors.to_rgb('green'))*255
astrotest_metadata = MetadataCatalog.get("astro_test").set(thing_classes=["galaxy", "star"]).set(thing_colors=[green, blue])

def plot_new_anns(ddict, new_img, kind='HWC'):
    # to visualize transformed annotations
    if kind == 'CHW':
        b1 = new_img[2, :, :]
        b2 = new_img[1, :, :]
        b3 = new_img[0, :, :]        
    else:
        b1 = new_img[:, :, 2]
        b2 = new_img[:, :, 1]
        b3 = new_img[:, :, 0]        
    vis_img = make_lupton_rgb(b1, b2, b3, minimum=0, stretch=0.5, Q=10)
    v0 = Visualizer(
        vis_img,
        metadata=astrotest_metadata,
        scale=1,
        instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    groundTruth = v0.draw_dataset_dict(ddict, lf=False, alpha=0.1, boxf=True)# boxf turns off  
    ax1 = plt.subplot(1, 1, 1)
    plt.figure(figsize=(7,7))
    ax1.imshow(groundTruth.get_image())

def transform(ddict, new_filename):#, save_cutout=False):
    ddict = copy.deepcopy(ddict) # segmentation masks were causing problems with shallow copy  
    img = reader._read_image(ddict['file_name']) # changes to (Height, Width, Channels)
    lsst_shape = img.shape[:-1] # height, width
    roman_shape = (512, 512) # (Channels, Height, width)
    # from https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.ResizeTransform
    # using bicubic instead of Lanczos interpolation as lanzcos isn't supported by
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html#torch.nn.functional.interpolate which is what
    # detectron2 uses
    transform_list = [
        T.ResizeTransform(lsst_shape[0], lsst_shape[1], roman_shape[0], roman_shape[1], interp=Image.BICUBIC) 
    ]
    
    new_img, transforms = T.apply_transform_gens(transform_list, img)   
    annos = [
        utils.transform_instance_annotations(obj, transforms, new_img.shape[:-1])
        for obj in ddict.pop("annotations")
    ]
    # convert bboxes back to XYWH_ABS and set bbox_mode
    for ann in annos:
        bbox = BoxMode.convert(ann["bbox"].tolist(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) # convert XYXY_ABS to XYWH_ABS
        ann["bbox"] = bbox
        ann["bbox_mode"] = 1
        ann["segmentation"] = [seg.tolist() for seg in ann['segmentation']] # to make it JSON serializable
    
    ddict["annotations"] = annos 
#     plot_new_anns(ddict, new_img)
    ddict["height"] = new_img.shape[:-1][0]
    ddict["width"] = new_img.shape[:-1][1]
    ddict["file_name"] = new_filename
    # annotations are fully updated now with this new cutout; they just need to be saved onto disk
#     if save_cutout:
#         # make sure to change back the axes of lsst_img so RomanImageReader can read it properly during training
#         new_img = np.transpose(new_img, axes=(2, 0, 1)).astype(np.float32) # (channels, height, width)
#         # combine roman with the new image
#         combined_data = np.concatenate((new_img, roman_img), axis=0)
#         np.save(new_filename, combined_data) # cutouts are saved onto disk
#     else:
#         pass
#         print("Combined cutouts already saved!")
    return ddict

def process_entry(entry, new_dir):
    old_lsst_filename = entry['file_name']
    new_filename = old_lsst_filename.replace('truth', new_dir)
    new_ddict = transform(entry, new_filename)
    return new_ddict

def process_all_metadata(metadata, metadata_file, new_dir):
    num_cpus = 20
    pool = mp.Pool(processes=num_cpus)
    process_func = partial(process_entry, new_dir=new_dir)
    
    # slightly diff than before since using tqdm
    # use imap so we get an iter that tqdm can use
    new_metadata = list(tqdm(
        pool.imap(process_func, metadata),
        total=len(metadata),
        desc="Processing entries"
    ))
    
    pool.close() # close and join to make sure all processes are done and we're back on main
    pool.join()

    output_file = metadata_file.replace('annotations', 'annotations' + new_dir[5:])
    with open(output_file, 'w') as output_file:
        json.dump(new_metadata, output_file, indent=4)

# serial - mainly used to test whole process on a single object
def process_all_metadata_serial(metadata, metadata_file, new_dir):
    new_metadata = []
    for entry in tqdm(metadata, desc="Processing each entry"):
        entry_filename = entry['file_name']
        print("Given:", entry_filename)
        new_filename = entry_filename.replace('truth', new_dir)
        print("New Filename for ann: ", new_filename)
        new_ddict = transform(entry, new_filename)
        print("New anns:", new_ddict)
        new_metadata.append(new_ddict)
        break
    
    output_file = metadata_file.replace('annotations', 'annotationsc-ups')
    print(output_file)
    with open(output_file, 'w') as output_file:
        json.dump(new_metadata, output_file, indent=4)

# saving each subpatch's entries into its own JSON file
def separate_metadata(metadata_file, output_dir):
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)
    
    subpatch_dict = {}
    # grouping entries by subpatch
    for entry in tqdm(all_metadata, desc="Grouping by subpatch"):
        subpatch = entry['subpatch']
        if subpatch not in subpatch_dict:
            subpatch_dict[subpatch] = []
        subpatch_dict[subpatch].append(entry)
    
    # save each subpatch's entries to disk
    for subpatch, entries in subpatch_dict.items():
        output_file_path = os.path.join(output_dir, f"{subpatch}.json")
        with open(output_file_path, 'w') as output_file:
            json.dump(entries, output_file, indent=4)

# process the all_metadata file first
# process_all_metadata(all_metadata, all_metadata_file, 'truthc-ups')

new_metadata_file = './lsst_data/annotationsc-ups-old/all_metadata.json'

if os.path.exists(new_metadata_file):
    print(f"{new_metadata_file} exists!\n Separating into each individual subpatch:")
    separate_metadata(new_metadata_file, './lsst_data/annotationsc-ups-old')
#     print("Now, we will be processing the train, val and test annotations!\n")
#     process_all_metadata(trainmd, train_file, 'truthc-ups')
#     process_all_metadata(valmd, val_file, 'truthc-ups')
#     process_all_metadata(testmd, test_file, 'truthc-ups')
    print("All done!")
else:
    print(f"{new_metadata_file} does not exist. Something went wrong")
