import cv2, hashlib, os
import numpy as np
import detectron2.data.transforms as T
import deepdisc.astrodet.detectron as detectron_addons

def build_remap(wcs_rubin, wcs_roman, dst_shape):
    """
    Precompute float32 remap arrays for cv2.remap.
    dst pixel -> sky -> src pixel
    """
    new_h, new_w = dst_shape
    # Grid of all Roman pixel coords
    yg, xg = np.indices((512, 512))
    ra, dec = wcs_roman.wcs_pix2world(xg.ravel(), yg.ravel(), 0)
    src_x, src_y = wcs_rubin.wcs_world2pix(ra, dec, 0)
    map_x = src_x.reshape(new_h, new_w).astype(np.float32)
    map_y = src_y.reshape(new_h, new_w).astype(np.float32)
    return map_x, map_y

class LanczosResizeTransform(T.Transform):
    """
    from /projects/bdsp/miniconda3/envs/ddbtknv/lib/python3.10/site-packages/fvcore/transforms/transform.py,
    we need to implement apply_image, apply_segmentation, and apply_coords to be compatible with Detectron2's transform_instance_annotations
    
    Resize an image from (h, w) to (new_h, new_w) using Lanczos interpolation
    and cv2 remapping to account for pixel distortions between Rubin and Roman WCS.  
    Use Lanczos for the image and nearest-neighbor for segmentation masks. Transform 
    coords (bboxes, polygons, keypoints) by converting rubin_pixel->world->roman_pixel with the WCS's
    
    Detectron2's transform_instance_annotations delegates bbox corner scaling
    and polygon point scaling to apply_coords
    
    Parameters
    ----------
    h, w : int
        Original image height and width.
    new_h, new_w : int
        Target image height and width.
    """
    def __init__(self, h: int, w: int, new_h: int, new_w: int, wcs_rubin, wcs_roman, rubin_fns, roman_fns, cache_dir):
        super().__init__()
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.scale_x = new_w / w
        self.scale_y = new_h / h
        self.wcs_rubin = wcs_rubin
        self.wcs_roman = wcs_roman
        # We cache the maps that cv2 uses to map rubin to roman pixels based on the WCS 
        cache_key = hashlib.md5((rubin_fns + roman_fns).encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
        self.map_x, self.map_y = np.load(cache_path)
    
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image using Lanczos (cv2.INTER_LANCZOS4)
        Works on float32 multi-channel arrays (e.g. 6-band Rubin HxWx6)

        Parameters
        ----------
        img : np.ndarray, shape (H, W, C) or (H, W)
            Input image array.

        Returns
        -------
        np.ndarray, shape (new_H, new_W, C) or (new_H, new_W)
        """
        h, w = img.shape[:2]
        assert self.h == h and self.w == w, (
            f"Input size mismatch: expected {self.h}x{self.w}, got {h}x{w}"
        )   
        C = img.shape[2]
        channels = [
            cv2.remap(img[..., c], self.map_x, self.map_y,
                      interpolation=cv2.INTER_LANCZOS4,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for c in range(C)
        ]
        return np.stack(channels, axis=-1)
        # return cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LANCZOS4)
    
    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Resize a dense integer segmentation label map using nearest-neighbor bc Lanczos would blur hard label boundaries 
        and introduce fractions b/w class indices, corrupting the mask
        Nearest-neighbor (the default for resize transforms in Detectron2) preserves exact label values

        Parameters
        ----------
        segmentation : np.ndarray, shape (H, W), dtype int
            Integer label map.

        Returns
        -------
        np.ndarray, shape (new_H, new_W), dtype int
        """
        return cv2.resize(segmentation.astype(np.float32), (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Scale (x, y) coordinate pairs by (scale_x, scale_y)
        Detectron2's transform_instance_annotations calls this method for
          - bounding box corners  (via apply_box)
          - segmentation polygon points (via apply_polygons)
          - keypoint xy positions
        Parameters
        ----------
        coords : np.ndarray, shape (N, 2)
            Array of (x, y) coordinate pairs.

        Returns
        -------
        np.ndarray, shape (N, 2)
        """
        coords = coords.astype(float)
        # coords[:, 0] *= self.scale_x
        # coords[:, 1] *= self.scale_y
        # return coords
        # Use low-level C API instead of SkyCoord overhead
        ra, dec = self.wcs_rubin.wcs_pix2world(coords[:, 0], coords[:, 1], 0)
        x_rom, y_rom = self.wcs_roman.wcs_world2pix(ra, dec, 0)
        return np.stack([x_rom, y_rom], axis=1)

    def inverse(self) -> "LanczosResizeTransform":
        """Return the inverse transform (resize back to original dimensions)"""
        return LanczosResizeTransform(self.new_h, self.new_w, self.h, self.w)

class LanczosResize(T.Augmentation):
    """
    Augmentation wrapper around LanczosResizeTransform

    Reads the input image shape at runtime and returns a
    LanczosResizeTransform sized to match. DO THIS FIRST in AugmentationList so that all subsequent
    geometric augs (flips, rotations) operate on the
    already-upsampled image with correctly scaled coords
    Parameters
    ----------
    new_h, new_w : int
        Target height and width.
    
    like this: 
    augmentations = T.AugmentationList([
        LanczosResize(512, 512),
        T.RandomFlip(horizontal=True)
    ])
    """

    def __init__(self, new_h: int, new_w: int, wcs_rubin, wcs_roman, rubin_fns, roman_fns, cache_dir):
        super().__init__()
        self.new_h = new_h
        self.new_w = new_w
        self.wcs_rubin = wcs_rubin
        self.wcs_roman = wcs_roman
        self.rubin_fns = rubin_fns
        self.roman_fns = roman_fns
        self.cache_dir = cache_dir

    def get_transform(self, image: np.ndarray) -> LanczosResizeTransform:
        """
        Parameters
        ----------
        image : np.ndarray, shape (H, W, C)
            The input image (used only to read current h, w)

        Returns
        -------
        LanczosResizeTransform
        """
        h, w = image.shape[:2]
        return LanczosResizeTransform(h, w, self.new_h, self.new_w, self.wcs_rubin, self.wcs_roman, self.rubin_fns, self.roman_fns, self.cache_dir)

def dc2_train_augs_512(new_h=512, new_w=512):
    """
    Same as dc2_train_augs, but with a fixed new_h and new_w resize to match LSST to Roman

    Parameters
    ----------
    new_h, new_w : int
        Target height and width for the LanczosResize transform. Default is 512x512 to match Roman image size

    Returns
    -------
    augs: detectron_addons.KRandomAugmentationList
        The list of augs for training.  Set to RandomRotation, RandomFlip, RandomCrop
    """
    # have to do resize first so that all subsequent augs operate on the correctly scaled image and coords
    resize = T.AugmentationList([LanczosResize(new_h, new_w)])
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([0, 180, -90, 90], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            #detectron_addons.CustomAug(multiband_gaussblur,prob=0.5),
        ],
        k=-1,
        cropaug=None,
        #cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    # we're making this class so that we can just 
    # return a single transform from this function, which is what the mapper expects    
    class CombinedAug:
        def __call__(self, aug_input):
            tfm1 = resize(aug_input)
            tfm2 = augs(aug_input)
            return tfm1 + tfm2

    return CombinedAug()