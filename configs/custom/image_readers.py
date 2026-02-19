from matplotlib import image
import numpy as np
from deepdisc.data_format.image_readers import ImageReader
from typing import List
# src/deepdisc/data_format/image_readers.py

class DualRomanRubinImageReader(ImageReader):
    """
    An ImageReader that loads BOTH Roman and LSST images for paired training.
    Returns a dict with both modalities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, filenames):
        """Read the filenames and apply scaling.
        Parameters
        ----------
        filenames : dictionary
            Dictionary containing both Rubin and Roman image paths to read         
            
            Ex: filenames = {
                'rubin_path': rubin_fns,
                'roman_path': roman_fns
            }
        Returns
        -------
        rubin_scaled : numpy array
            The scaled Rubin image.
        roman_scaled : numpy array
            The scaled Roman image.
        """
        if isinstance(filenames, dict):
            rubin_imgs, roman_imgs = self._read_image(filenames)
        else:
            raise ValueError("For DualRomanRubinImageReader, please provide a dict with 'rubin_path' and 'roman_path' \
                             containing the paths to the respective image files.")
        # apply scaling to each modality separately
        rubin_scaled = self.scaling(rubin_imgs, **self.scalekwargs)
        roman_scaled = self.scaling(roman_imgs, **self.scalekwargs)        
        return rubin_scaled, roman_scaled
    
    def _read_image(self, filenames):
        """
        Read paired Roman and LSST images.
        
        Parameters
        ----------
        filenames : dict
            Dictionary with keys 'rubin_path' and 'roman_path'
            
        Returns
        -------
        Rubin and Roman image arrs
        """
        if isinstance(filenames, dict):
            rubin_path, roman_path = filenames['rubin_path'], filenames['roman_path']
        else:            
            raise ValueError("For DualRomanRubinImageReader, please provide a dict with 'rubin_path' and 'roman_path' \
                             containing the paths to the respective image files.")
        # load Rubin image (6 bands)
        rubin_img = np.load(rubin_path)  # (6, height, width)
        rubin_img = np.transpose(rubin_img, axes=(1, 2, 0)).astype(np.float32)  # (H, W, 6)
        # load Roman image (3 bands)
        roman_img = np.load(roman_path)  # (3, 512, 512)
        roman_img = np.transpose(roman_img, axes=(1, 2, 0)).astype(np.float32)  # (512, 512, 3)
        return rubin_img, roman_img

class DualRubinRubinImageReader(ImageReader):
    """
    An ImageReader that loads BOTH Roman and LSST images for paired training.
    Returns a dict with both modalities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, filenames):
        """Read the filenames and apply scaling.
        Parameters
        ----------
        filenames : dictionary
            Dictionary containing both Rubin and Roman image paths to read         
            
            Ex: filenames = {
                'rubin_path': rubin_fns,
                'roman_path': roman_fns
            }
        Returns
        -------
        rubin_scaled : numpy array
            The scaled Rubin image
        rubin_scaled_2 : numpy array
            The scaled Rubin image 
        """
        if isinstance(filenames, dict):
            rubin_imgs, rubin_imgs_2 = self._read_image(filenames)
        else:
            raise ValueError("For DualRubinRubinImageReader, please provide a dict with 'rubin_path' and 'roman_path' \
                             containing the paths to the respective image files.")
        # apply scaling to each modality separately
        rubin_scaled = self.scaling(rubin_imgs, **self.scalekwargs)
        rubin_scaled_2 = self.scaling(rubin_imgs_2, **self.scalekwargs)        
        return rubin_scaled, rubin_scaled_2
    
    def _read_image(self, filenames):
        """
        Read paired Roman and LSST images.
        
        Parameters
        ----------
        filenames : dict
            Dictionary with keys 'rubin_path' and 'roman_path'
            
        Returns
        -------
        Rubin and Roman image arrs
        """
        if isinstance(filenames, dict):
            rubin_path, roman_path = filenames['rubin_path'], filenames['roman_path']
        else:            
            raise ValueError("For DualRomanRubinImageReader, please provide a dict with 'rubin_path' and 'roman_path' \
                             containing the paths to the respective image files.")
        # load Rubin image (6 bands)
        rubin_img = np.load(rubin_path)  # (6, height, width)
        rubin_img = np.transpose(rubin_img, axes=(1, 2, 0)).astype(np.float32)  # (H, W, 6)
        # load Roman image (3 bands)
        rubin_img_2 = np.load(roman_path)
        rubin_img_2 = np.transpose(rubin_img_2, axes=(1, 2, 0)).astype(np.float32)
        return rubin_img, rubin_img_2

class RomanRubinImageReader(ImageReader):
    """An ImageReader for Roman/ Rubin image files."""

    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def _read_image(self, filename):
        """Read the image.

        Parameters
        ----------
        filename : str
            The filename indicating the image to read.

        Returns
        -------
        im : numpy array
            The image.
        """
        image = np.load(filename) # (4, 512, 512)
        image = np.transpose(image, axes=(1, 2, 0)).astype(np.float32) # (512, 512, 4)
        return image