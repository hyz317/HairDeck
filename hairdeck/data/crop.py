"""Image-condition crop rule, shared by training (renders) and inference (character images), so both see the same
operator.

box = bbox(hair mask U face mask), made square with a margin. Training adds random scale / shift jitter (covers the gap
between the GT masks used on renders and the SAM3 masks used on input images: SAM3 vs GT hair bbox IoU 0.93 to 0.95)
and, sometimes, cuts the bottom of long hair off (upper-body images crop long hair at the image border).

Training masks come from gen_data.py's label image (1 hair mesh, 2 face mesh); inference uses SAM3 "hair" / "face".
"""
import numpy as np
from PIL import Image

LAB_HAIR, LAB_FACE = 1, 2


def mask_box(mask):
    """(x0, y0, x1, y1), exclusive, of a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def crop_box(hair, face, rng=None, margin=0.08, scale_jitter=0.0, shift_jitter=0.0, trunc_p=0.0, trunc_keep=(0.45, 0.8)):
    """Square crop box (x0, y0, x1, y1) in pixels, possibly extending past the image.

    hair, face: boolean masks (face may be empty, e.g. back views). rng=None gives the deterministic inference box.
    trunc_p: probability (training only) of cutting the box bottom so only trunc_keep of the hair height stays,
    applied only when the hair is much taller than the face (long hair); the face always stays inside."""
    hb, fb = mask_box(hair), mask_box(face)
    b = union(hb, fb)
    if b is None:
        raise ValueError('empty hair and face masks')
    x0, y0, x1, y1 = b
    if rng is not None and hb is not None and rng.random() < trunc_p:
        face_h = (fb[3] - fb[1]) if fb is not None else 0.0
        if (y1 - y0) > 2.0 * max(face_h, 1.0):
            keep = rng.uniform(*trunc_keep)
            y1 = max(y0 + keep * (y1 - y0), fb[3] if fb is not None else y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    if rng is not None:
        side *= float(np.exp(rng.normal(0, scale_jitter))) if scale_jitter else 1.0
        cx += float(rng.normal(0, shift_jitter)) * side if shift_jitter else 0.0
        cy += float(rng.normal(0, shift_jitter)) * side if shift_jitter else 0.0
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def apply_crop(img, box, out=512, resample=Image.Resampling.BICUBIC):
    """Crop a PIL image to `box` (areas outside the image become transparent / zero) and resize to out x out."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mode = img.mode
    canvas = Image.new(mode, (x1 - x0, y1 - y0), (0,) * len(mode) if len(mode) > 1 else 0)
    canvas.paste(img, (-x0, -y0))
    return canvas.resize((out, out), resample)


def composite(rgba, bg=(255, 255, 255)):
    """RGBA render onto a flat background (inference: the input image's character is matted onto the same colour)."""
    base = Image.new('RGBA', rgba.size, (*bg, 255))
    base.alpha_composite(rgba.convert('RGBA'))
    return base.convert('RGB')


def train_crop(rgba_path, lab_path, rng, out=512, **kw):
    """One training sample: RGB crop from a gen_data.py view. kw: crop_box options (TRAIN_CROP by default)."""
    lab = np.asarray(Image.open(lab_path))
    box = crop_box(lab == LAB_HAIR, lab == LAB_FACE, rng=rng, **({**TRAIN_CROP, **kw}))
    return composite(apply_crop(Image.open(rgba_path), box, out)), box


TRAIN_CROP = dict(margin=0.08, scale_jitter=0.08, shift_jitter=0.04, trunc_p=0.15)
