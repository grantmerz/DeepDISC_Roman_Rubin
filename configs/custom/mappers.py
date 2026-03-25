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
        # Delete the uncompressed image array if it exists since we don't need it
        # using .pop(key, default) safely removes it without throwing an error if it's missing
        d.pop('image_shaped', None)
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
        # Delete the uncompressed image array if it exists since we don't need it
        # using .pop(key, default) safely removes it without throwing an error if it's missing
        d.pop('image_shaped', None)
        d['file_name'] = dataset_dict['file_name']
        d['wcs'] = dataset_dict['wcs']
        return d

# for synchronized augmentations across paired images, we need to rebuild the transforms for the new image size
def rescale_transform(tfm, shape):
    """
    Reinitialize a transform applied to image A so it is correctly rescaled 
    to the size of a new image B. 
    Handles HFlip, VFlip, and Rotation. Passthrough for others.
    Assumes the new image B is spatially aligned with the old image A.

    Parameters
    ----------
    tfm: detectron2.data.transforms.Transform
        the transform to apply (e.g. HFlipTransform,
        RotationTransform)
    shape: Tuple (H,W,C)
        The shape of the image to apply the transform to
    Returns
    -------
    tfm: detectron2.data.transforms.Transform
        The new rescaled transform
    """
    h, w = shape[:2]
    cx = w // 2
    cy = h // 2
    if isinstance(tfm, T.HFlipTransform):
        return T.HFlipTransform(w)
    elif isinstance(tfm, T.VFlipTransform):
        return T.VFlipTransform(h)
    elif isinstance(tfm, T.RotationTransform):
        return T.RotationTransform(
            h=h,
            w=w,
            angle=tfm.angle,         
            expand=tfm.expand,
            center=(cx, cy),
            interp=tfm.interp,
        )
    else:
        warnings.warn(
            f"rescale_transform: unsupported transform type "
            f"{type(tfm).__name__}. Only HFlipTransform, VFlipTransform "
            f"and RotationTransform are supported.",
            category=UserWarning
        )
        return tfm



class ResizeCombinedMapper(DataMapper):
    """
    Data mapper for upsampling LSST images and 
    combining with Roman.  Also upsamples annotations.
    """
    def __init__(self, keypoint_hflip_indices=None, cache_dir='.', *args, **kwargs):
        # Pickling with multiprocessing can cause issues
        # so this ensures the keypoint_hflip_indices is a list 
        if keypoint_hflip_indices is not None:
            keypoint_hflip_indices = list(keypoint_hflip_indices)
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.cache_dir = cache_dir
        super().__init__(*args, **kwargs)

 
    def map_data(self, dataset_dict):
        """
        Map COCO dict data to format for CLIP + detection training
        
        Parameters
        ----------
        dataset_dict: dict
            COCO formatted metadata with LSST paths
            
        Returns
        -------
        dict with 'rubin_image', 'roman_image', and optionally 'instances'
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        rubin_fns = self.km(dataset_dict)
        roman_fns = rubin_fns.replace('lsst_data', 'truth-roman').replace('/truth/','/')
        filenames = {
            'rubin_path': rubin_fns,
            'roman_path': roman_fns
        }
        # loading in both imgs using DualRomanRubinImageReader
        rubin_imgs, roman_imgs = self.IR(filenames)
        C_rubin=rubin_imgs.shape[-1]
        C_roman=roman_imgs.shape[-1]
        comb_img = torch.empty((C_rubin+C_roman, 512, 512), dtype=torch.float32)

        #wcs roman has been saved in the lsst dict
        wcs_rubin = WCS(dataset_dict['wcs'])
        wcs_roman = WCS(dataset_dict['wcs_roman'])
        new_h, new_w = 512, 512
        initial_resize = LanczosResize(new_h, new_w,wcs_rubin,wcs_roman,rubin_fns,roman_fns,self.cache_dir)
        

        # --- Synchronized spatial augmentations ---
        # Sample augmentations ONCE from Rubin, then rebuild the same
        # logical operations for Roman's dimensions so that ROIs stay
        # spatially aligned across modalities
        rubin_auginput = T.AugInput(rubin_imgs)
        if self.augmentations is not None:
            rubin_augs = self.augmentations(rubin_imgs)
        else:
            rubin_augs = T.AugmentationList([])
        # Apply to Rubin (samples random choices & modifies auginput in place)
        # Add on the initial resize transform
        transform_rubin = initial_resize(rubin_auginput)         
        transform_rubin+=rubin_augs(rubin_auginput)
        #Speed optimization
        rubin_img = torch.as_tensor(np.ascontiguousarray(rubin_auginput.image.transpose(2, 0, 1)))

        # Rebuild the same transforms for Roman's (h, w) and apply
        # by iterating over transforms in rubin's transform list
        rescaled_transforms = [
            rescale_transform(tfm, roman_imgs.shape)
            for tfm in transform_rubin.transforms[1:]  #[1:] there to avoid the resize trasnform
        ]
        # new transform list resized for Roman that we can apply to roman imgs 
        transform_roman = T.TransformList(rescaled_transforms)         
        roman_img_transformed = transform_roman.apply_image(roman_imgs)
        roman_img = torch.as_tensor(np.ascontiguousarray(roman_img_transformed.transpose(2, 0, 1)))

        comb_img[:C_rubin] = rubin_img
        comb_img[C_rubin:] = roman_img

        #transform annotations
        annos = [
            utils.transform_instance_annotations(
                annotation, [transform_rubin], comb_img.shape[1:], keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for annotation in dataset_dict.pop('annotations')
        ] 

        instances = utils.annotations_to_instances(annos, comb_img.shape[1:])
        instances = utils.filter_empty_instances(instances)
        result = {
            'image': comb_img,
            #'image_roman': roman_img,
            'height': comb_img.shape[1],
            'width': comb_img.shape[2],
            'image_id': dataset_dict['image_id'],
            'instances': instances,
            'annotations':annos
        }

        return result




# Mapper to resize Rubin and combine with Roman
class ResizeCombinedandLSSTMapper(DataMapper):
    """
    Data mapper for upsampling LSST images and 
    combining with Roman.  Also upsamples annotations.

    Includes the native-scale LSST image

    """
    def __init__(self, keypoint_hflip_indices=None, cache_dir='.', *args, **kwargs):
        # Pickling with multiprocessing can cause issues
        # so this ensures the keypoint_hflip_indices is a list 
        if keypoint_hflip_indices is not None:
            keypoint_hflip_indices = list(keypoint_hflip_indices)
        self.keypoint_hflip_indices = keypoint_hflip_indices
        #cache dir points to where the cached pixel maps are for cv2 to use on the fly 
        self.cache_dir = cache_dir
        super().__init__(*args, **kwargs)

 
    def map_data(self, dataset_dict):
        """
        Map COCO dict data to format for CLIP + detection training
        
        Parameters
        ----------
        dataset_dict: dict
            COCO formatted metadata with LSST paths
            
        Returns
        -------
        dict with 'rubin_image', 'roman_image', and optionally 'instances'
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        rubin_fns = self.km(dataset_dict)
        roman_fns = rubin_fns.replace('lsst_data', 'truth-roman').replace('/truth/','/')
        filenames = {
            'rubin_path': rubin_fns,
            'roman_path': roman_fns
        }
        # loading in both imgs using DualRomanRubinImageReader
        rubin_imgs, roman_imgs = self.IR(filenames)
        C_rubin=rubin_imgs.shape[-1]
        C_roman=roman_imgs.shape[-1]
        comb_img = torch.empty((C_rubin+C_roman, 512, 512), dtype=torch.float32)

        #wcs roman has been saved in the lsst dict
        wcs_rubin = WCS(dataset_dict['wcs'])
        wcs_roman = WCS(dataset_dict['wcs_roman'])
        new_h, new_w = 512, 512
        initial_resize = LanczosResize(new_h, new_w,wcs_rubin,wcs_roman,rubin_fns,roman_fns,self.cache_dir)
        

        # --- Synchronized spatial augmentations ---
        # Sample augmentations ONCE from Rubin, then rebuild the same
        # logical operations for Roman's dimensions so that ROIs stay
        # spatially aligned across modalities
        rubin_auginput = T.AugInput(rubin_imgs)
        rubin_auginput_resize = T.AugInput(rubin_imgs)

        if self.augmentations is not None:
            rubin_augs = self.augmentations(rubin_imgs)
        else:
            rubin_augs = T.AugmentationList([])
        
        # Apply to Rubin (samples random choices & modifies auginput in place)
        # Add on the initial resize transform
        transform_rubin_resize = initial_resize(rubin_auginput_resize)         
        transform_rubin_resize+=rubin_augs(rubin_auginput_resize)
        
        #Separately apply the same spatial transforms to the original rubin image (no resizing) 
        transform_rubin=rubin_augs(rubin_auginput)

        #Speed optimization
        rubin_img = torch.as_tensor(np.ascontiguousarray(rubin_auginput.image.transpose(2, 0, 1)))
        rubin_img_resize = torch.as_tensor(np.ascontiguousarray(rubin_auginput_resize.image.transpose(2, 0, 1)))

        # Rebuild the same transforms for Roman's (h, w) and apply
        # by iterating over transforms in rubin's transform list
        rescaled_transforms = [
            rescale_transform(tfm, roman_imgs.shape)
            for tfm in transform_rubin_resize.transforms[1:]  #[1:] there to avoid the resize trasnform
        ]
        # new transform list resized for Roman that we can apply to roman imgs 
        transform_roman = T.TransformList(rescaled_transforms)         
        roman_img_transformed = transform_roman.apply_image(roman_imgs)
        roman_img = torch.as_tensor(np.ascontiguousarray(roman_img_transformed.transpose(2, 0, 1)))

        comb_img[:C_rubin] = rubin_img_resize
        comb_img[C_rubin:] = roman_img

        #transform annotations to match the LSST image augmentations
        annos = [
            utils.transform_instance_annotations(
                annotation, [transform_rubin], rubin_img.shape[1:], keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for annotation in dataset_dict.pop('annotations')
        ] 

        instances = utils.annotations_to_instances(annos, comb_img.shape[1:])
        instances = utils.filter_empty_instances(instances)
        result = {
            'image_combined': comb_img,
            'image_rubin': rubin_img,
            'height': rubin_img.shape[1],
            'width': rubin_img.shape[2],
            'image_id': dataset_dict['image_id'],
            'instances': instances,
            'annotations':annos
        }

        return result

class CLIPMapper(DataMapper):
    """
    Data mapper for CLIP training with Roman-LSST pairs.
    Handles both labeled and unlabeled data.
    """
    def __init__(self, *args, keypoint_hflip_indices=None, **kwargs):
        self.keypoint_hflip_indices = keypoint_hflip_indices
        super().__init__(*args, **kwargs)
    
    def map_data(self, dataset_dict):
        """
        Map COCO dict data to format for CLIP + detection training
        
        Parameters
        ----------
        dataset_dict: dict
            COCO formatted metadata with LSST paths
            
        Returns
        -------
        dict with 'rubin_image', 'roman_image', and optionally 'instances'
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        rubin_fns = self.km(dataset_dict)
        roman_fns = rubin_fns.replace('lsst_data', 'roman_data')
        # for grant m: roman_fns = rubin_fns.replace('lsst_data', 'truth-roman').replace('/truth/','/')
        # file paths for both modalities
        filenames = {
            'rubin_path': rubin_fns,
            'roman_path': roman_fns
        }
        # loading in both imgs using DualRomanRubinImageReader
        rubin_imgs, roman_imgs = self.IR(filenames)
        # --- Synchronized spatial augmentations ---
        # Sample augmentations ONCE from Rubin, then rebuild the same
        # logical operations for Roman's dimensions so that ROIs stay
        # spatially aligned across modalities
        rubin_auginput = T.AugInput(rubin_imgs)
        if self.augmentations is not None:
            rubin_augs = self.augmentations(rubin_imgs)
        else:
            rubin_augs = T.AugmentationList([])
        # Apply to Rubin (samples random choices & modifies auginput in place)
        transform_rubin = rubin_augs(rubin_auginput)
        rubin_img = torch.from_numpy(rubin_auginput.image.copy().transpose(2, 0, 1))

        # Rebuild the same transforms for Roman's (h, w) and apply
        # by iterating over transforms in rubin's transform list
        rescaled_transforms = [
            rescale_transform(tfm, roman_imgs.shape)
            for tfm in transform_rubin.transforms
        ]
        # new transform list resized for Roman that we can apply to roman imgs 
        transform_roman = T.TransformList(rescaled_transforms)         
        roman_img_transformed = transform_roman.apply_image(roman_imgs)
        roman_img = torch.from_numpy(roman_img_transformed.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(
                annotation, [transform_rubin], rubin_img.shape[1:], keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for annotation in dataset_dict.pop('annotations')
        ] 
        instances = utils.annotations_to_instances(annos, rubin_img.shape[1:])
        # add obj_id field so we can track which objs match to which proposals across modalities
        # label_and_sample_propsals will then copy it onto sampled proposals thru:
        #  props_per_img.set(trg_name, trg_vale[sampled_targets])
        # for any field starting with gt_ that isn't already set on proposals
        instances.gt_objid = torch.tensor([ann['obj_id'] for ann in annos], dtype=torch.int64)
        instances = utils.filter_empty_instances(instances)
        # and we don't modify roman_img annotations since we only use rubin for detection
        result = {
            'image_rubin': rubin_img,
            'image_roman': roman_img,
            'height': rubin_img.shape[1],
            'width': rubin_img.shape[2],
            'image_id': dataset_dict['image_id'],
            'instances': instances
        }
        return result

class CLIPEvalMapper(DataMapper): 
    """
    Data mapper for CLIP evaluation with just LSST data.
    
    This mapper returns "image_rubin" instead of "image" like DataMapper does so the model's
    training-mode forward pass can compute validation losses.
    """
    
    def __init__(self, *args, keypoint_hflip_indices=None, **kwargs):
        self.keypoint_hflip_indices = keypoint_hflip_indices
        super().__init__(*args, **kwargs)
    
    def map_data(self, dataset_dict):
        """
        Map COCO dict data to format for CLIP + detection evaluation
        
        Parameters
        ----------
        dataset_dict: dict
            COCO formatted metadata with LSST paths
            
        Returns
        -------
        dict with 'rubin_image', and optionally 'instances'
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        rubin_fns = self.km(dataset_dict)
        # loading in imgs using RomanRubinImageReader
        rubin_imgs = self.IR(rubin_fns)
        # apply augmentations to Rubin images
        rubin_auginput = T.AugInput(rubin_imgs)
        if self.augmentations is not None:
            rubin_augs = self.augmentations(rubin_imgs)
        else:
            rubin_augs = T.AugmentationList([])
        # same transformations to ensure spatial consistency
        transform = rubin_augs(rubin_auginput) # modifies rubin auginput in place
        rubin_img = torch.from_numpy(rubin_auginput.image.copy().transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(
                annotation, [transform], rubin_img.shape[1:], keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for annotation in dataset_dict.pop('annotations')
        ]
        instances = utils.annotations_to_instances(annos, rubin_img.shape[1:])
        # don't really need it for eval but just include for consistencey with training mapper
        instances.gt_objid = torch.tensor([ann['obj_id'] for ann in annos], dtype=torch.int64)
        instances = utils.filter_empty_instances(instances)
        # and we don't modify roman_img annotations since we only use rubin for detection
        result = {
            'image_rubin': rubin_img,
            'height': rubin_img.shape[1],
            'width': rubin_img.shape[2],
            'image_id': dataset_dict['image_id'],
            'instances': instances
        }
        return result

class CLIPTestMapper(CLIPEvalMapper):
    """
    Data mapper for CLIP testing with just LSST data.
    
    This mapper builds on top of CLIPEvalMapper and returns file names and WCS info as well.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def map_data(self, dataset_dict):
        # calling the map_data method from the parent class (CLIPEvalMapper)
        # This will do all the image loading, augmentations, and instance transformations
        d = super().map_data(dataset_dict)
        d['file_name'] = dataset_dict['file_name']
        d['wcs'] = dataset_dict['wcs']
        return d

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

# class CLIPRubinMapper(DataMapper):
#     """
#     Data mapper for CLIP training with LSST-LSST pairs.
#     Handles both labeled and unlabeled data.
#     """
    
#     def __init__(self, keypoint_hflip_indices=None, *args, **kwargs):
#         self.keypoint_hflip_indices = keypoint_hflip_indices
#         super().__init__(*args, **kwargs)
    
#     def map_data(self, dataset_dict):
#         """
#         Map COCO dict data to format for CLIP + detection training
        
#         Parameters
#         ----------
#         dataset_dict: dict
#             COCO formatted metadata with LSST paths
            
#         Returns
#         -------
#         dict with 'rubin_image', 'roman_image', and optionally 'instances'
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)
#         rubin_fns = self.km(dataset_dict)
#         # file paths for both modalities
#         filenames = {
#             'rubin_path': rubin_fns,
#             'roman_path': rubin_fns
#         }
#         # loading in both imgs using DualRubinRubinImageReader
#         rubin_imgs, rubin_imgs_2 = self.IR(filenames)
#         # apply augmentations to Rubin/Roman images (same augs for both but will have slightly different transforms)
#         rubin_auginput = T.AugInput(rubin_imgs)
#         rubin_auginput_2 = T.AugInput(rubin_imgs_2)
#         if self.augmentations is not None:
#             rubin_augs = self.augmentations(rubin_imgs)
#             rubin_augs_2 = self.augmentations(rubin_imgs_2)
#         else:
#             rubin_augs = T.AugmentationList([])
#             rubin_augs_2 = T.AugmentationList([])
#         # same transformations to ensure spatial consistency
#         transform_rubin = rubin_augs(rubin_auginput) # modifies rubin auginput in place
#         rubin_img = torch.from_numpy(rubin_auginput.image.copy().transpose(2, 0, 1))
        
#         transform_rubin_2 = rubin_augs_2(rubin_auginput_2) # modifies rubin auginput in place
#         rubin_img_2 = torch.from_numpy(rubin_auginput_2.image.copy().transpose(2, 0, 1))
#         annos = [
#             utils.transform_instance_annotations(
#                 annotation, [transform_rubin], rubin_img.shape[1:], keypoint_hflip_indices=self.keypoint_hflip_indices
#             )
#             for annotation in dataset_dict.pop('annotations')
#         ]
#         instances = utils.annotations_to_instances(annos, rubin_img.shape[1:])
#         instances = utils.filter_empty_instances(instances)
#         # and we don't modify rubin_img_2 annotations since we only use rubin for detection
#         result = {
#             'image_rubin': rubin_img,
#             'image_roman': rubin_img_2,
#             'height': rubin_img.shape[1],
#             'width': rubin_img.shape[2],
#             'image_id': dataset_dict['image_id'],
#             'instances': instances
#         }
#         return result