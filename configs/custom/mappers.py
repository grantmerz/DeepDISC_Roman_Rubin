import copy
import torch
from deepdisc.model.loaders import DataMapper, DictMapper
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils



def rescale_transform(tfm, shape):
    """
    Reinitialize a transform applied to image A so it is correctly rescaled to the size of a new image B.
    Handles HFlip, VFlip, and Rotation. Passthrough for others.
    Assumes the new image B is spatially aligned with the old image A.

    Parameters
    ----------
    tfm: detectron2.data.transforms.Transform
        the transform to apply
    shape: Tuple (H,W,C)
        The shape of the image to apply the transform to
        
    Returns
    -------
    tfm: detectron2.data.transforms.Transform
        The new rescaled transform

    """


    h, w = shape[:2]
    cx = w//2
    cy = h//2

    if isinstance(tfm, T.HFlipTransform):
        return T.HFlipTransform(w)
    
    elif isinstance(tfm, T.VFlipTransform):
        return T.VFlipTransform(h)
    
    elif isinstance(tfm, T.RotationTransform):
        return T.RotationTransform(
            h=h,
            w=h,
            angle=tfm.angle,         
            expand=tfm.expand,
            center=(cx, cy),
            interp=tfm.interp,
        )

    else:
        return tfm



class SimpleMapper(DataMapper):
    """Maps file names in the dataset.
    Args:
        DictMapper (DictMapper): Base class for mapping COCO-formatted dictionary data.
    """
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
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
        image = torch.from_numpy(image.transpose(2, 0, 1))
        annos = [
            annotation for annotation in dataset_dict.pop("annotations")
        ]
        instances = utils.annotations_to_instances(annos, image.shape[1:])
        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": self.IR(key),
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"]
        }


class DictMapperWithKeypoints(DataMapper):
    """Class that will map COCO dictionary data to the format necessary for the model"""

    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
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
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:],keypoint_hflip_indices=[0])
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


class MoCoMapper(DataMapper):
    """
    Data mapper for MoCo training with Roman-LSST pairs.
    Handles both labeled and unlabeled data.
    """
    
    def __init__(self, keypoints=False, *args, **kwargs):
        self.keypoints = keypoints
        super().__init__(*args, **kwargs)
    
    def map_data(self, dataset_dict):
        """
        Map COCO dict data to format for MoCo + detection training
        
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
        # file paths for both modalities
        filenames = {
            'rubin_path': rubin_fns,
            'roman_path': roman_fns
        }
        # loading in both imgs using DualRomanRubinImageReader
        rubin_imgs, roman_imgs = self.IR(filenames)
        # apply augmentations to Rubin/Roman images (same augs for both but will have slightly different transforms)
        rubin_auginput = T.AugInput(rubin_imgs)
        #roman_auginput = T.AugInput(roman_imgs)
        if self.augmentations is not None:
            rubin_augs = self.augmentations(rubin_imgs)
            #roman_augs = self.augmentations(roman_imgs)
        else:
            rubin_augs = T.AugmentationList([])
            roman_augs = T.AugmentationList([])
        
        transform_rubin = rubin_augs(rubin_auginput) 
        rubin_img = torch.from_numpy(rubin_auginput.image.copy().transpose(2, 0, 1))
        
        #transform_roman = roman_augs(roman_auginput)
        rescaled_transforms = [
            rescale_transform(tfm, roman_imgs.shape)
            for tfm in transform_rubin.transforms
        ]
        #transform_roman = T.AugmentationList(rescaled_transforms)(roman_auginput)
        #roman_img = torch.from_numpy(roman_auginput.image.copy().transpose(2, 0, 1))
        transform_roman = T.TransformList(rescaled_transforms)
        roman_img_transformed = transform_roman.apply_image(roman_imgs)
        roman_img = torch.from_numpy(roman_img_transformed.copy().transpose(2, 0, 1))

        if self.keypoints:
            annos = [
            utils.transform_instance_annotations(
                annotation, [transform_rubin], rubin_img.shape[1:], keypoint_hflip_indices=[0]
            )
            for annotation in dataset_dict.pop('annotations')
        ]

        else:
            annos = [
                utils.transform_instance_annotations(
                    annotation, [transform_rubin], rubin_img.shape[1:]
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

class MoCoRubinMapper(DataMapper):
    """
    Data mapper for MoCo training with LSST-LSST pairs.
    Handles both labeled and unlabeled data.
    """
    
    def __init__(self, keypoints=False,*args, **kwargs):
        self.keypoints=keypoints
        super().__init__(*args, **kwargs)

    
    def map_data(self, dataset_dict):
        """
        Map COCO dict data to format for MoCo + detection training
        
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
        # file paths for both modalities
        filenames = {
            'rubin_path': rubin_fns,
            'roman_path': rubin_fns
        }
        # loading in both imgs using DualRomanRubinImageReader
        rubin_imgs, rubin_imgs_2 = self.IR(filenames)
        # apply augmentations to Rubin/Roman images (same augs for both but will have slightly different transforms)
        rubin_auginput = T.AugInput(rubin_imgs)
        rubin_auginput_2 = T.AugInput(rubin_imgs_2)
        if self.augmentations is not None:
            rubin_augs = self.augmentations(rubin_imgs)
            rubin_augs_2 = self.augmentations(rubin_imgs_2)
        else:
            rubin_augs = T.AugmentationList([])
            rubin_augs_2 = T.AugmentationList([])
        # same transformations to ensure spatial consistency
        transform_rubin = rubin_augs(rubin_auginput) # modifies rubin auginput in place
        rubin_img = torch.from_numpy(rubin_auginput.image.copy().transpose(2, 0, 1))
        
        transform_rubin_2 = rubin_augs_2(rubin_auginput_2) # modifies rubin auginput in place
        rubin_img_2 = torch.from_numpy(rubin_auginput_2.image.copy().transpose(2, 0, 1))

        if self.keypoints:
            annos = [
                        utils.transform_instance_annotations(
                            annotation, [transform_rubin], rubin_img.shape[1:], keypoint_hflip_indices=[0]
                        )
                        for annotation in dataset_dict.pop('annotations')
                    ]
        else:
            annos = [
                utils.transform_instance_annotations(
                    annotation, [transform_rubin], rubin_img.shape[1:]
                )
                for annotation in dataset_dict.pop('annotations')
            ]
        instances = utils.annotations_to_instances(annos, rubin_img.shape[1:])
        instances = utils.filter_empty_instances(instances)
        # and we don't modify rubin_img_2 annotations since we only use rubin for detection
        result = {
            'image_rubin': rubin_img,
            'image_roman': rubin_img_2,
            'height': rubin_img.shape[1],
            'width': rubin_img.shape[2],
            'image_id': dataset_dict['image_id'],
            'instances': instances
        }
        return result


class MoCoEvalMapper(DataMapper):
    """
    Data mapper for MoCo evaluation with just LSST data.
    
    This mapper returns "image_rubin" instead of "image" like DataMapper does so the model's
    training-mode forward pass can compute validation losses.
    """
    
    def __init__(self, keypoints=False, *args, **kwargs):
        self.keypoints = keypoints
        super().__init__(*args, **kwargs)
        
    
    def map_data(self, dataset_dict):
        """
        Map COCO dict data to format for MoCo + detection evaluation
        
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

        if self.keypoints:
            annos = [
                utils.transform_instance_annotations(
                    annotation, [transform], rubin_img.shape[1:], keypoint_hflip_indices=[0]
                )
                for annotation in dataset_dict.pop('annotations')
            ]

        else:
            annos = [
                utils.transform_instance_annotations(
                    annotation, [transform], rubin_img.shape[1:]
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

    
class MoCoTestMapper(MoCoEvalMapper):
    """
    Data mapper for MoCo testing with just LSST data.
    
    This mapper builds on top of MoCoEvalMapper and returns file names and WCS info as well.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def map_data(self, dataset_dict):
        # calling the map_data method from the parent class (MoCoEvalMapper)
        # This will do all the image loading, augmentations, and instance transformations
        d = super().map_data(dataset_dict)
        d['file_name'] = dataset_dict['file_name']
        d['wcs'] = dataset_dict['wcs']
        return d