import abc
import os

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb


from deepdisc.data_format.image_readers import ImageReader

class SimpleImageReader(ImageReader):
    """An ImageReader for DC2 image files."""

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
        image = np.load(filename)
        image = np.transpose(image, axes=(1, 2, 0)).astype(np.float32)
        return image
    
    
    
class CombinedImageReader(ImageReader):
    """An ImageReader for DC2 image files."""

    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def _read_image(self, suffix):
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
        
        rubinim = np.load('/home/shared/hsc/roman_lsst/./lsst_data/truth-ups/dc2_'+suffix)
        romanim = np.load(os.path.join('/home/shared/hsc/roman_lsst/roman_data/truth/',suffix))
        
        image = np.vstack((rubinim,romanim))
        image = np.transpose(image, axes=(1, 2, 0)).astype(np.float32)
        return image