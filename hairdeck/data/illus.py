"""Model input of the image DiT at inference, the counterpart of the training crops (hairdeck.data.crop): the input
image matted onto white, cut with the crop box at the model resolution, white outside the image (hairdeck.infer.image).
"""
import numpy as np
from PIL import Image

from .crop import apply_crop, composite


def crop_white(img, box, out=512):
    """Crop an RGB array with white outside the image."""
    rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.asarray(composite(apply_crop(Image.fromarray(rgba, 'RGBA'), box, out)))
