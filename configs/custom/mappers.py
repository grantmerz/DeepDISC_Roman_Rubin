import copy
import torch
import warnings
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

class CLIPMapper(DataMapper):
    """
    Data mapper for CLIP training with Roman-LSST pairs.
    Handles both labeled and unlabeled data.
    """
    def __init__(self, keypoint_hflip_indices=None, *args, **kwargs):
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
        roman_img = torch.from_numpy(roman_img_transformed.image.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(
                annotation, [transform_rubin], rubin_img.shape[1:], keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for annotation in dataset_dict.pop('annotations')
        ] 
        instances = utils.annotations_to_instances(annos, rubin_img.shape[1:])
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
    
    def __init__(self, keypoint_hflip_indices=None, *args, **kwargs):
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