import numpy as np
from deepdisc.data_format.image_readers import ImageReader

# src/deepdisc/data_format/image_readers.py

class DualRomanRubinImageReader(ImageReader):
    """
    An ImageReader that loads BOTH Roman and LSST images for paired training.
    Returns a dict with both modalities.
    """
    
    def __init__(self, roman_bands=3, lsst_bands=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roman_bands = roman_bands
        self.lsst_bands = lsst_bands
    
    def _read_image(self, filename_dict):
        """
        Read paired Roman and LSST images.
        
        Parameters
        ----------
        filename_dict : dict
            Dictionary with keys 'lsst_path' and 'roman_path'
            
        Returns
        -------
        dict with keys 'lsst' and 'roman', each containing image arrs
        """
        lsst_path = filename_dict['lsst_path']
        roman_path = filename_dict['roman_path']
        # load LSST image (6 bands)
        lsst_image = np.load(lsst_path)  # (6, height, width)
        lsst_image = np.transpose(lsst_image, axes=(1, 2, 0)).astype(np.float32)  # (512, 512, 6)
        # load Roman image (3 bands)
        roman_image = np.load(roman_path)  # (3, 512, 512)
        roman_image = np.transpose(roman_image, axes=(1, 2, 0)).astype(np.float32)  # (512, 512, 3)
        
        return {
            'lsst': lsst_image,
            'roman': roman_image
        }

class RomanRubinImageReader(ImageReader):
    """An ImageReader for Roman image files."""

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