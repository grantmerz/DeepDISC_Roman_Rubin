import copy

import detectron2.data as data
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data import detection_utils as utils

import deepdisc.astrodet.astrodet as toolkit
import deepdisc.astrodet.detectron as detectron_addons
from deepdisc.model.loaders import DataMapper






class MocoDictMapper(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        lsst_key,roman_key = self.km(dataset_dict)
        lsst_image = self.IR(lsst_key)
        roman_image = self.IR(roman_key)
        # Data Augmentation (do we want to do this?)
        auginput_lsst = T.AugInput(lsst_image)
        auginput_roman = T.AugInput(roman_image)

        # Transformations to model shapes
        if self.augmentations is not None:
            lsst_augs = self.augmentations(lsst_image)
            roman_augs = self.augmentations(roman_image)
        else:
            augs = T.AugmentationList([])

        transform_lsst = lsst_augs(auginput_lsst)
        lsst_image = torch.from_numpy(auginput_lsst.image.copy().transpose(2, 0, 1))

        transform_roman = roman_augs(auginput_roman)
        roman_image = torch.from_numpy(auginput_roman.image.copy().transpose(2, 0, 1))


        annos = [
            utils.transform_instance_annotations(annotation, [transform_lsst], lsst_image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            #if annotation["redshift"] != 0.0
        ]

        instances = utils.annotations_to_instances(annos, lsst_image.shape[1:])

        #instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])
        
        #instances.gt_objid = torch.tensor([a["objectId"] for a in annos])

        instances = utils.filter_empty_instances(instances)
        
        
        return {
            # create the format that the model expects
            "image_lsst": lsst_image,
            "image_roman": roman_image,
            "height": lsst_image.shape[1],
            "width": lsst_image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            #"annotations": annos
            #"wcs": wcs
        }
    