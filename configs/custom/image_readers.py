import abc
import os

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from deepdisc.data_format.image_readers import ImageReader

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