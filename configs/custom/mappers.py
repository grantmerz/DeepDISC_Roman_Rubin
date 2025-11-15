import copy
import torch
from deepdisc.model.loaders import DataMapper, DictMapper
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

class FileNameMapper(DictMapper):
    """Adds file names in the dataset.
    Args:
        DictMapper (DictMapper): Base class for mapping COCO-formatted dictionary data.
    """
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)
    
    def map_data(self, dataset_dict):
        # calling the map_data method from the parent class (DictMapper)
        # This will do all the image loading, augmentations, and instance transformations
        d = super().map_data(dataset_dict)
        d['file_name'] = dataset_dict['file_name']
        return d

class FileNameWCSMapper(DictMapper):
    """Adds file names and WCS info in the dataset.
    Args:
        DictMapper (DictMapper): Base class for mapping COCO-formatted dictionary data.
    """
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)
    
    def map_data(self, dataset_dict):
        # calling the map_data method from the parent class (DictMapper)
        # This will do all the image loading, augmentations, and instance transformations
        d = super().map_data(dataset_dict)
        d['file_name'] = dataset_dict['file_name']
        d['wcs'] = dataset_dict['wcs']
        return d
