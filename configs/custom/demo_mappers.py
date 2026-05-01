import copy
import torch
import torch.utils.data as torchdata
import warnings
from deepdisc.model.loaders import DataMapper, DictMapper
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.data.build import trivial_batch_collator, get_detection_dataset_dicts
import os
from .transforms import LanczosResize
from astropy.wcs import WCS
import numpy as np



class DictMapper(DataMapper):
    """Class that will map COCO dictionary data to the format necessary for the model"""

    def __init__(self, keypoints=False, *args, **kwargs):
        # Pass arguments to the parent function.
        self.keypoints = keypoints
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        if self.keypoints:
            annos = [
                utils.transform_instance_annotations(annotation, [transform], image.shape[1:],keypoint_hflip_indices=[0])
                for annotation in dataset_dict.pop("annotations")
            ]
        else:
            annos = [
                utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
                for annotation in dataset_dict.pop("annotations")
            ]


        instances = utils.annotations_to_instances(annos, image.shape[1:])
        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
        }



class SupConDictMapper(DataMapper):
    """Class that will map COCO dictionary data to the format necessary for the model
       Adds ground truth object IDs to allow for supervised contrastive learning.
       Requires annotations to have an "obj_id" key.
    """

    def __init__(self, keypoints=False, *args, **kwargs):
        # Pass arguments to the parent function.
        self.keypoints = keypoints
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        if self.keypoints:
            annos = [
                utils.transform_instance_annotations(annotation, [transform], image.shape[1:],keypoint_hflip_indices=[0])
                for annotation in dataset_dict.pop("annotations")
            ]
        else:
            annos = [
                utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
                for annotation in dataset_dict.pop("annotations")
            ]


        instances = utils.annotations_to_instances(annos, image.shape[1:])
        # add obj_id field so we can track which objs match to which proposals across modalities
        # label_and_sample_propsals will then copy it onto sampled proposals thru:
        #  props_per_img.set(trg_name, trg_vale[sampled_targets])
        # for any field starting with gt_ that isn't already set on proposals
        instances.gt_objid = torch.tensor([ann['obj_id'] for ann in annos], dtype=torch.int64)
        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
        }


# worker_init_fn sets 'file_system' sharing strategy inside each DataLoader worker process.
# This is necessary because DataLoader workers are spawned (not forked), so they don't inherit
# the sharing strategy from the parent GPU process so it must be set explicitly in each worker.
# https://docs.pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
# 'file_system' uses filenames (via shm_open) to identify the shared memory region instead of caching file descriptors,
# which avoids hitting the per-process FD limit (ulimit -n) when running many workers.
def worker_init_fn(worker_id):
    torch.multiprocessing.set_sharing_strategy('file_system')

def return_custom_test_loader(cfg, mapper):
    """Returns a test loader with configurable batch size, persistent workers,
    and file_system sharing strategy set in each worker.

    Parameters
    ----------
    cfg : LazyConfig
        The lazy config, which contains data loader config values
        including batch size, num_workers, and persistent_workers
    mapper : callable
        A callable which takes a sample (dict) from dataset and returns
        the format to be consumed by the model.

    Returns
    -------
        a test loader
    """
    batch_size = getattr(cfg.dataloader.test, 'total_batch_size', 1)
    num_workers = getattr(cfg.dataloader.test, 'num_workers', 0)
    persistent_workers = getattr(cfg.dataloader.test, 'persistent_workers', False)
    # Set filter_empty=False to allow images without annotations (for inference on unlabeled data)
    dataset = get_detection_dataset_dicts(cfg.DATASETS.TEST, filter_empty=False)
    # loader w/ explicit params so we can bypass @configurable for detectron2's build_detection_test_loader()
    # detectron2-style type checking
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        sampler = None
    else:
        sampler = InferenceSampler(len(dataset))

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0, # only enable persistent workers if u have more than one worker
    )