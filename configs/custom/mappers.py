import copy
import torch
from deepdisc.model.loaders import DataMapper, DictMapper
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

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
    
    def __init__(self, *args, **kwargs):
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
        roman_fns = rubin_fns.replace('lsst_data', 'roman_data')
        # file paths for both modalities
        filenames = {
            'rubin_path': rubin_fns,
            'roman_path': roman_fns
        }
        # loading in both imgs using DualRomanRubinImageReader
        rubin_imgs, roman_imgs = self.IR(filenames)
        # apply augmentations to Rubin/Roman images (same augs for both but will have slightly different transforms)
        rubin_auginput = T.AugInput(rubin_imgs)
        roman_auginput = T.AugInput(roman_imgs)
        if self.augmentations is not None:
            rubin_augs = self.augmentations(rubin_imgs)
            roman_augs = self.augmentations(roman_imgs)
        else:
            rubin_augs = T.AugmentationList([])
            roman_augs = T.AugmentationList([])
        # same transformations to ensure spatial consistency
        transform_rubin = rubin_augs(rubin_auginput) # modifies rubin auginput in place
        rubin_img = torch.from_numpy(rubin_auginput.image.copy().transpose(2, 0, 1))
        
        transform_roman = roman_augs(roman_auginput) # modifies roman auginput in place
        roman_img = torch.from_numpy(roman_auginput.image.copy().transpose(2, 0, 1))
        
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


class MoCoEvalMapper(DataMapper):
    """
    Data mapper for MoCo evaluation with just LSST data.
    
    This mapper returns "image_rubin" instead of "image" like DataMapper does so the model's
    training-mode forward pass can compute validation losses.
    """
    
    def __init__(self, *args, **kwargs):
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