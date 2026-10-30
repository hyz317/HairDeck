"""Condition images of the image DiT: crops of the rendered views (scripts/data.py) and their appearance
augmentation in training.

RenderViews has one item per (hairstyle, view): the render composited on white, cut with the crop rule of
hairdeck.data.crop (the deterministic inference box at test time; scale / shift jitter and truncation of long hair in
training) and resized to the model input. Training items are augmented as set by a style config (configs/style_aug.json,
load_style_cfg), with two random generators per item, seeded by (seed, epoch, item index): rng draws the crop box and
the base augmentation (augment_rgb), srng every other decision. Per training item:

  * with probability p_style the style branch (style_crop): the same crop box cut at the render's native resolution
    (crop side about 600 to 1000 px), so that pixel-sized parameters of the style operations mean the same on every
    character; a plan of at least min_ops operations (draw_plan): a colour stage from colour (keep, gray, mono_tone,
    sketch, hue_full, chan, flat, painterly; hairdeck.data.style_ops), ink (p_ink), rim light / shading (p_light,
    light_kinds), then back under the alpha matte on white, low resolution (p_lowres, lowres_px), the resize to the
    model input, augment_rgb, and at the model input Gaussian blur (p_blur) or sharpening (p_sharpen), noise inside
    the matte (p_noise) and JPEG (p_jpeg);
  * otherwise the plain path: crop, resize, augment_rgb.
With p_style 0 the training images are the plain path only (the base augmentation).
"""
import glob
import io
import json
import os

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from . import style_ops as so
from .crop import LAB_FACE, LAB_HAIR, TRAIN_CROP, apply_crop, composite, crop_box
from .views import dir_to_key

VIEWS = 8  # rendered views per character
STYLE_STREAM = 0x5717  # srng = default_rng([seed, epoch, j, STYLE_STREAM]); rng = default_rng([seed, epoch, j])
STYLE_KEYS = ('p_style', 'min_ops', 'colour', 'p_ink', 'p_light', 'light_kinds', 'p_lowres', 'lowres_px', 'p_blur',
              'blur_sigma', 'p_sharpen', 'sharpen_amount', 'sharpen_radius', 'p_noise', 'noise_sigma', 'p_jpeg', 'jpeg_q')


def load_style_cfg(path):
    """The style config (every key of STYLE_KEYS; keys starting with '_' are comments)."""
    c = {k: v for k, v in json.load(open(path)).items() if not k.startswith('_')}
    if set(c) != set(STYLE_KEYS):
        raise KeyError(f'{path}: missing keys {sorted(set(STYLE_KEYS) - set(c))}, unknown keys {sorted(set(c) - set(STYLE_KEYS))}')
    s = sum(c['colour'].values())
    if abs(s - 1) > 1e-6:
        raise ValueError(f'colour probabilities sum to {s}')
    return c


def augment_rgb(img, rng):
    """Base appearance augmentation: brightness / contrast / saturation jitter, hue shift, blur, JPEG, grayscale."""
    for E, lo, hi in ((ImageEnhance.Brightness, 0.8, 1.2), (ImageEnhance.Contrast, 0.8, 1.2), (ImageEnhance.Color, 0.6, 1.4)):
        if rng.random() < 0.8:
            img = E(img).enhance(float(rng.uniform(lo, hi)))
    if rng.random() < 0.5:  # hue shift
        hsv = np.asarray(img.convert('HSV')).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(rng.integers(-13, 14))) % 256
        img = Image.fromarray(hsv, 'HSV').convert('RGB')
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=int(rng.integers(40, 95)))
        img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
    if rng.random() < 0.1:
        img = img.convert('L').convert('RGB')
    return img


def draw_plan(srng, cfg):
    """The operations of one style-branch item (dict), drawn from srng; at least cfg['min_ops'] of them."""
    names, probs = zip(*cfg['colour'].items())
    for _ in range(100):
        p = dict(colour=str(srng.choice(names, p=np.array(probs) / sum(probs))))
        p['ink'] = p['colour'] not in so.LINE_OPS and srng.random() < cfg['p_ink']
        p['light'] = (str(srng.choice(list(cfg['light_kinds']), p=list(cfg['light_kinds'].values())))
                      if p['colour'] not in so.LINE_OPS and srng.random() < cfg['p_light'] else None)
        p['lowres'] = bool(srng.random() < cfg['p_lowres'])
        r = srng.random()
        p['blur'] = bool(r < cfg['p_blur'])
        p['sharpen'] = bool(cfg['p_blur'] <= r < cfg['p_blur'] + cfg['p_sharpen'])
        p['noise'] = bool(srng.random() < cfg['p_noise'])
        p['jpeg'] = bool(srng.random() < cfg['p_jpeg'])
        n = (p['colour'] != 'keep') + p['ink'] + (p['light'] is not None) + sum(p[k] for k in ('lowres', 'blur', 'sharpen', 'noise', 'jpeg'))
        if n >= cfg['min_ops']:
            return p
    return p


def style_crop(rgba, box, out, plan, rng, srng, cfg):
    """The style branch for one item. rgba: the whole render (PIL RGBA); box: crop_box output; rng: the item's crop /
    base-augmentation generator (only augment_rgb draws from it here); srng: the item's style generator. -> PIL RGB
    out x out."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    canvas = Image.new('RGBA', (x1 - x0, y1 - y0), (0, 0, 0, 0))
    canvas.paste(rgba.convert('RGBA'), (-x0, -y0))
    arr = np.asarray(canvas).astype(np.float32)
    alpha = arr[..., 3] / 255
    rgb = so.u8(arr[..., :3] * alpha[..., None] + 255 * (1 - alpha[..., None]))
    side = float(x1 - x0)
    ctx = dict(alpha=alpha, side=side, seed=int(srng.integers(2 ** 31)))
    x = rgb
    if plan['colour'] == 'flat' and (alpha > 0.5).sum() < 256:  # flat runs k-means on the matte's pixels; a few renders
        plan['colour'] = 'keep'  # are fully transparent in some views
    if plan['colour'] != 'keep':
        x = so.COLOUR_OPS[plan['colour']](x, ctx, srng)
    if plan['ink']:
        x = so.thick_ink(x, ctx, srng)
    if plan['light'] in ('rim', 'both'):
        x = so.rim_light(x, ctx, srng)
    if plan['light'] in ('shade', 'both'):
        x = so.shade(x, ctx, srng)
    if x is not rgb:
        x = so.recomposite(x, alpha)
    if plan['lowres']:
        x = so.lowres(x, side, srng, *cfg['lowres_px'])
    img = augment_rgb(Image.fromarray(x).resize((out, out), Image.Resampling.BICUBIC), rng)
    if plan['blur']:
        img = img.filter(ImageFilter.GaussianBlur(float(srng.uniform(*cfg['blur_sigma']))))
    elif plan['sharpen']:
        img = img.filter(ImageFilter.UnsharpMask(radius=float(srng.uniform(*cfg['sharpen_radius'])),
                                                 percent=int(100 * srng.uniform(*cfg['sharpen_amount'])), threshold=0))
    if plan['noise']:
        a = np.asarray(Image.fromarray(so.u8(alpha * 255)).resize((out, out), Image.Resampling.BILINEAR)).astype(np.float32) / 255
        y = np.asarray(img).astype(np.float32)
        sig = float(srng.uniform(*cfg['noise_sigma']))
        n = srng.standard_normal(y.shape[:2] + ((3,) if srng.random() < 0.5 else (1,))).astype(np.float32) * sig
        img = Image.fromarray(so.u8(y + n * a[..., None]))
    if plan['jpeg']:
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=int(srng.integers(cfg['jpeg_q'][0], cfg['jpeg_q'][1] + 1)))
        img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
    return img


class RenderViews(Dataset):
    """Items (hairstyle index, render dir, view) of the characters of meta_json with renders in render_root, in render
    dir order; item j -> dict(hid, view, img = (3, out, out) uint8). train=False: the inference crop, no augmentation;
    train=True: the training crop and the augmentation of the style config cfg (seed and the attribute epoch seed the
    items' generators)."""

    def __init__(self, render_root, meta_json, out=512, train=False, cfg=None, seed=0):
        key2idx = {e['key']: i for i, e in enumerate(json.load(open(meta_json)))}
        self.items = []
        for d in sorted(glob.glob(f'{render_root}/*/meta.json')):
            d = os.path.dirname(d)
            k = dir_to_key(d)
            if k in key2idx:
                self.items += [(key2idx[k], d, v) for v in range(VIEWS)]
        self.out, self.train, self.cfg, self.seed = out, train, cfg, seed
        self.epoch = 0

    def keep(self, hids):
        """Only the views of the hairstyles in hids."""
        self.items = [it for it in self.items if it[0] in hids]

    def __len__(self):
        return len(self.items)

    def _box(self, d, v0, rng):
        """(view, crop box) from the label image of view v0 in d; when it shows neither hair nor face, the next view
        that does (or, with no such view, the whole image of view v0)."""
        lab = None
        for v in [v0] + [u for u in range(VIEWS) if u != v0]:
            if not os.path.exists(f'{d}/v{v}_lab.png'):
                continue
            lab = np.asarray(Image.open(f'{d}/v{v}_lab.png'))
            try:
                return v, crop_box(lab == LAB_HAIR, lab == LAB_FACE, rng=rng, **(TRAIN_CROP if self.train else {}))
            except ValueError:
                continue
        return v0, (0.0, 0.0) + lab.shape[::-1]

    def __getitem__(self, j):
        hid, d, v0 = self.items[j]
        if not self.train:
            v, box = self._box(d, v0, None)
            img = composite(apply_crop(Image.open(f'{d}/v{v}.png'), box, self.out))
        else:
            cfg = self.cfg
            rng = np.random.default_rng([self.seed, self.epoch, j])
            srng = np.random.default_rng([self.seed, self.epoch, j, STYLE_STREAM])
            v, box = self._box(d, v0, rng)
            rgba = Image.open(f'{d}/v{v}.png')
            if srng.random() < cfg['p_style']:
                img = style_crop(rgba, box, self.out, draw_plan(srng, cfg), rng, srng, cfg)
            else:
                img = augment_rgb(composite(apply_crop(rgba, box, self.out)), rng)
        return dict(hid=hid, view=v, img=torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
