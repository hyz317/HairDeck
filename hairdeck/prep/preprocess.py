"""Preprocessing of one character image for the image model: person matte (soft alpha), hair mask, face mask and the
crop box of the model input.

Models: SAM 3 (text prompts, bf16; all prompts of an image go through the grounding head in one batch, the text
features are computed once, at the first image) and BiRefNet-matting for the whole-character alpha. The mask logic is
recipe.py with the preset presets.json "best" (7 prompts: person, anime character, hair, face, clothes, hand, fur;
BiRefNet-matting at 768; working resolution 1024): it runs at the working resolution, hair and face are rebuilt at full
resolution from the selected SAM 3 instances, the person alpha is upsampled bilinearly. The box is the inference rule of
hairdeck.data.crop.crop_box (hair | face, 8% margin, square). A transparent input (a character already cut out) is
composited onto black for the models and the outputs are limited to its alpha.

  pre = Preprocessor()                         # loads SAM 3 and BiRefNet-matting (about 6.6 GB of GPU memory)
  p = pre(rgb, alpha=None, image_path=None)    # rgb (H, W, 3) uint8, alpha (H, W) uint8 or None
  p['person'] (H, W) uint8, p['hair'] / p['face'] (H, W) bool, p['box'] (x0, y0, x1, y1) or None,
  p['status'] ok / no_person_mask / no_hair / no_hair_no_face, p['notes'], p['info']
image_path: the unmodified file, read as it is (otherwise the arrays are used, with the same result for a lossless file).
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from ..data.crop import crop_box, union
from . import recipe
from .matte import build as build_matte
from .sam3_ground import ground, load_model, text_feats

HERE = os.path.dirname(os.path.abspath(__file__))


def gpu_box(m):
    """crop.mask_box of a CUDA bool mask, without copying it to the CPU."""
    rows, cols = m.any(1), m.any(0)
    if not bool(rows.any()):
        return None
    ys, xs = torch.nonzero(rows)[:, 0], torch.nonzero(cols)[:, 0]
    return float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1)


def box_of(hb, fb, margin=0.08):
    """hairdeck.data.crop.crop_box(hair, face) (inference branch, rng=None) from the two mask boxes."""
    b = union(hb, fb)
    if b is None:
        raise ValueError('empty hair and face masks')
    x0, y0, x1, y1 = b
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def exif_rotated(f):
    """True for a JPEG / WebP file whose EXIF orientation is not 1 (an in-memory PIL image: False). Only these formats
    are checked: PIL's getexif() on a PNG decodes the whole image to look for a trailing eXIf chunk."""
    if isinstance(f, Image.Image):
        return False
    with Image.open(f) as im:
        return im.format in ('JPEG', 'MPO', 'WEBP') and im.getexif().get(0x0112, 1) != 1


def load_image(f, use_alpha=True, bg=(0, 0, 0)):
    """f: a file or a PIL image. Returns (the image as the image model reads it, Image.open(f).convert('RGB'); the image
    the models see; the input alpha or None). A transparent input (an RGBA / LA / palette image with at least 0.1% of
    its pixels below alpha 250, e.g. a character the user has already cut out) is composited onto a flat background
    `bg` for SAM3 and the matting model, and the caller limits the outputs to its alpha. convert('RGB') alone keeps the colour stored under
    the transparent pixels (black in most such files, but also grey or green) and lets the matte reach past the
    alpha. Black is the default: on the 76 renders as RGBA files SAM3 finds more of the hair against black than against
    white (box IoU 0.955 against 0.931). Any other image goes through as it is."""
    im0 = f if isinstance(f, Image.Image) else Image.open(f)
    im = im0.convert('RGB')
    if use_alpha and (im0.mode in ('RGBA', 'LA', 'PA', 'RGBa', 'La') or 'transparency' in im0.info):
        rgba = im0.convert('RGBA')
        al = np.asarray(rgba)[..., 3]
        if (al < 250).mean() > 0.001:
            if len(bg) == 4:  # (r, g, b, 'fill'): stored colours kept, bg only under the (almost) fully transparent pixels
                x = np.asarray(rgba)[..., :3].copy()
                x[al < 8] = bg[:3]
                return im, Image.fromarray(x), al
            base = Image.new('RGBA', rgba.size, (*bg, 255))
            base.alpha_composite(rgba)
            return im, base.convert('RGB'), al
    return im, im, None


BG = dict(black=(0, 0, 0), white=(255, 255, 255), grey=(128, 128, 128), fill_black=(0, 0, 0, 'fill'))


PRESETS = json.load(open(f'{HERE}/presets.json'))


class Prep:
    def __init__(self, cfg, matte=None, matte_size=0, wr=1024, crop_expand=1.3, compile=False, sam_blur=0.0,
                 matte_blur=0.0, sam_max=0):
        self.cfg = dict(recipe.DEFAULT, **cfg)
        c = self.cfg
        self.prompts = list(dict.fromkeys(list(c['person_prompts']) + list(c['hair_prompts']) + list(c['face_prompts'])
                                          + list(c['acc_prompts']) + list(c['hair_minus']) + list(c['extra_prompts'])
                                          + list(c['acc_exclude']) + list(c['acc_cut'])))
        self.crop_prompts = list(dict.fromkeys(list(c['hair_prompts']) + list(c['face_prompts']) + list(c['acc_prompts'])))
        ts = [c['t_person'], c['t_hair'], c['t_face']] + ([c['t_acc']] if c['acc_prompts'] else [])
        ts += ([c['t_minus']] if c['hair_minus'] else []) + ([c['t_extra']] if c['extra_prompts'] else [])
        ts += [c['t_excl']] if c['acc_exclude'] else []
        ts += [c['t_cut']] if c['acc_cut'] else []
        ts += [c['box_fallback']] if c['box_fallback'] > 0 else []
        ts += [c['hair_low']] if c['hair_low'] > 0 else []
        ts += [x for x in (c['face_fallback'], c['hair_fallback']) if x > 0]
        self.t_min = min(ts)
        self.model, self.proc = load_model(compile)
        self._tf = None  # the prompts' text features, made at the first use (property tf)
        self.matte = build_matte(matte, matte_size) if matte else None
        self.wr = wr
        self.crop_expand = crop_expand
        self.sam_blur, self.matte_blur = sam_blur, matte_blur  # Gaussian blur (px) of the SAM3 / matting input only
        self.sam_max = sam_max  # > 0: SAM3 input downscaled (box filter) to at most this long side

    @property
    def tf(self):
        """made at the first use, not in __init__: on a Hugging Face ZeroGPU Space the models load without a GPU, and the
        GPU worker cannot use inference tensors made at load ("Inference tensors do not track version counter")"""
        if self._tf is None:
            self._tf = text_feats(self.model, self.prompts)
        return self._tf

    def _sam(self, im, prompts, shape, paste=None):
        """One SAM3 pass. Returns {prompt: [(score, bool mask at working res)]} and the kept low-res logits.
        paste = (X0, Y0, cw, ch, shape): the image is a crop; masks are upsampled to (ch, cw) and pasted at (X0, Y0)
        into a working-res canvas of the full image."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            st = self.proc.set_image(im)
            pr, bx, mk = ground(self.model, st, self.tf, [self.prompts.index(p) for p in prompts])
        inst, logits = {}, {}
        for j, p in enumerate(prompts):
            keep = torch.nonzero(pr[j] > self.t_min)[:, 0]
            keep = keep[torch.argsort(pr[j][keep], descending=True)]
            if len(keep) == 0:
                inst[p] = []
                continue
            lg = mk[j][keep].float()[:, None]
            logits[p] = lg
            if paste is None:
                up = (F.interpolate(lg, shape, mode='bilinear', align_corners=False)[:, 0] > 0).cpu().numpy()
            else:
                X0, Y0, cw, ch = paste
                u = F.interpolate(lg, (ch, cw), mode='bilinear', align_corners=False)[:, 0] > 0
                full = torch.zeros((len(keep),) + tuple(shape), dtype=torch.bool, device='cuda')
                ys0, xs0, ys1, xs1 = max(0, Y0), max(0, X0), min(shape[0], Y0 + ch), min(shape[1], X0 + cw)
                if ys1 > ys0 and xs1 > xs0:
                    full[:, ys0:ys1, xs0:xs1] = u[:, ys0 - Y0:ys1 - Y0, xs0 - X0:xs1 - X0]
                up = full.cpu().numpy()
            inst[p] = list(zip(pr[j][keep].float().cpu().tolist(), list(up)))
        return inst, logits

    @torch.inference_mode()
    def __call__(self, im):
        T = {}
        c = self.cfg
        rgb = np.asarray(im)
        H, W = rgb.shape[:2]
        s = min(1.0, self.wr / max(H, W))
        shape = (max(1, round(H * s)), max(1, round(W * s)))
        torch.cuda.synchronize()
        t0 = time.time()
        im_s = im.filter(ImageFilter.GaussianBlur(self.sam_blur)) if self.sam_blur > 0 else im
        if self.sam_max > 0 and max(im_s.size) > self.sam_max:
            k = self.sam_max / max(im_s.size)
            im_s = im_s.resize((max(1, round(im_s.width * k)), max(1, round(im_s.height * k))), Image.Resampling.BOX)
        inst, logits = self._sam(im_s, self.prompts, shape)
        inst_c = cbox = logits_c = crop_px = None
        if c['crop_mode'] != 'none':
            # head crop: crop.py box of the first pass with the default hair / face rule, enlarged around its centre
            _, ph0, pf0, _ = recipe.run(inst, shape, dict(matte='sam', no_person='ones', fill='none', close_r=0), None)
            try:
                b = crop_box(ph0, pf0)
            except ValueError:
                b = None
            if b is not None:
                k = H / shape[0]
                cx, cy = (b[0] + b[2]) / 2 * k, (b[1] + b[3]) / 2 * k
                side = max(b[2] - b[0], b[3] - b[1]) * k * self.crop_expand
                x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
                x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
                cim = Image.new('RGB', (x1 - x0, y1 - y0), (255, 255, 255))
                cim.paste(im, (-x0, -y0))
                X0, Y0 = int(round(x0 * s)), int(round(y0 * s))
                cw, ch = max(1, int(round((x1 - x0) * s))), max(1, int(round((y1 - y0) * s)))
                inst_c, logits_c = self._sam(cim, self.crop_prompts, shape, paste=(X0, Y0, cw, ch))
                cbox, crop_px = (X0, Y0, X0 + cw, Y0 + ch), (x0, y0, x1, y1)
        torch.cuda.synchronize()
        t1 = time.time()
        T['sam3'] = t1 - t0
        alpha = None
        if self.matte is not None:
            a = self.matte(np.asarray(im.filter(ImageFilter.GaussianBlur(self.matte_blur))) if self.matte_blur > 0 else rgb)
            alpha = F.interpolate(a[None, None], shape, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            torch.cuda.synchronize()
        t2 = time.time()
        T['matte'] = t2 - t1
        pa, ph, pf, info = recipe.run(inst, shape, self.cfg, alpha, inst_crop=inst_c, crop_box=cbox)
        t3 = time.time()
        T['recipe'] = t3 - t2

        def full(keys):
            out = torch.zeros((H, W), dtype=torch.bool, device='cuda')
            for kk in keys:
                if kk[0] == 'cc':  # hysteresis instance restricted to its parts touching the hair
                    _, p, j = kk
                    reg = torch.from_numpy(info['hair_cc_region']).cuda()[None, None].float()
                    reg = F.interpolate(reg, (H, W), mode='nearest')[0, 0] > 0.5
                    out |= (F.interpolate(logits[p][j:j + 1], (H, W), mode='bilinear', align_corners=False)[0, 0] > 0) & reg
                    continue
                if kk[0] == 'crop':
                    _, p, j = kk
                    x0, y0, x1, y1 = crop_px
                    u = F.interpolate(logits_c[p][j:j + 1], (y1 - y0, x1 - x0), mode='bilinear', align_corners=False)[0, 0] > 0
                    ys0, xs0, ys1, xs1 = max(0, y0), max(0, x0), min(H, y1), min(W, x1)
                    if ys1 > ys0 and xs1 > xs0:
                        out[ys0:ys1, xs0:xs1] |= u[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
                    continue
                p, j = kk
                out |= F.interpolate(logits[p][j:j + 1], (H, W), mode='bilinear', align_corners=False)[0, 0] > 0
            return out

        hair, face = full(info.get('hair_keys', [])), full(info.get('face_keys', []))
        if info.get('hair_minus_keys'):
            hair &= ~full(info['hair_minus_keys'])
        if info.get('hair_region') is not None:  # pruned stray hair components
            reg = torch.from_numpy(info['hair_region']).cuda()[None, None].float()
            hair &= F.interpolate(reg, (H, W), mode='nearest')[0, 0] > 0.5
        person = F.interpolate(torch.from_numpy(pa).cuda()[None, None], (H, W), mode='bilinear',
                               align_corners=False)[0, 0].clamp(0, 1)
        person = (person * 255).round().byte()
        torch.cuda.synchronize()
        T['full_res'] = time.time() - t3
        return person, hair, face, info, T


def one(prep, f):
    """The steps for one image (a file or a PIL image): load_image (a transparent input is composited onto black for
    the models), Prep, the outputs limited to the input alpha, the warnings, the crop box (box_of of the hair and
    face boxes, rounded to 0.1 px), the status. Returns (record, person uint8 soft alpha, hair bool, face bool)."""
    t0 = time.time()
    im, im_model, in_alpha = load_image(f, True, BG['black'])
    person_g, hair_g, face_g, info, T = prep(im_model)
    r = dict(status=info.get('status', 'ok'), size=list(im.size))
    if in_alpha is not None:  # transparent input: outputs limited to the input alpha
        ag = torch.from_numpy(np.ascontiguousarray(in_alpha)).cuda()
        person_g = torch.minimum(person_g, ag)
        hair_g = hair_g & (ag > 127)
        face_g = face_g & (ag > 127)
        r['input_alpha'] = True
    warn = []
    if 'agree' in info:
        r['matte_agree'] = round(info['agree'], 3)
        if info['agree'] < 0.8:
            warn.append('matte_uncertain')
    if info.get('status') == 'no_person_mask':
        warn.append('no_person')
    if info.get('box_fallback'):
        warn.append('box_from_low_scores')
    if exif_rotated(f):
        warn.append('exif_rotated')
    if warn:
        r['warn'] = warn
    hb = gpu_box(hair_g)
    try:
        box = box_of(hb, gpu_box(face_g))
        r['box'] = [round(v, 1) for v in box]
        if hb is None:
            r['status'] = 'no_hair'
    except ValueError:
        r['status'] = 'no_hair_no_face'
    person, hair, face = person_g.cpu().numpy(), hair_g.cpu().numpy(), face_g.cpu().numpy()
    T['total'] = time.time() - t0
    r['time'] = {k: round(v, 4) for k, v in T.items()}
    return r, person, hair, face


class Preprocessor:
    """The released preprocessing: Prep with a preset of presets.json and the steps of `one`, models loaded once.
    Returns dict(rgb, person (H, W) uint8 0..255, hair, face (H, W) bool, box (x0, y0, x1, y1) or None, status, notes,
    info, backend); rgb is the RGB the model input is cut from (the stored colours, Image.open(f).convert('RGB'), the
    person matte being limited to the input alpha)."""

    def __init__(self, preset='best'):
        self.name = f'matting:{preset}'
        t0 = time.time()
        P = PRESETS[preset]
        self.prep = Prep(P['cfg'], P.get('alpha'), P.get('alpha_size', 0), P.get('wr', 1024), P.get('crop_expand', 1.3),
                         False)
        self.load_sec = round(time.time() - t0, 1)

    def __call__(self, rgb, alpha=None, image_path=None):
        if image_path is None:  # the decoded input, as the lossless file it came from
            arr = np.concatenate([rgb, np.asarray(alpha)[..., None]], -1) if alpha is not None else rgb
            image_path = Image.fromarray(arr)
        r, person, hair, face = one(self.prep, image_path)
        r['peak_gpu_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        box = tuple(r['box']) if 'box' in r else None
        return dict(rgb=rgb, person=np.asarray(person, np.uint8), hair=np.asarray(hair, bool), face=np.asarray(face, bool),
                    box=None if box is None else tuple(float(v) for v in box), status=r['status'],
                    notes=list(r.get('warn', [])) + (['input_alpha'] if r.get('input_alpha') else []),
                    info=dict(matting={k: v for k, v in r.items() if k not in ('box', 'status', 'warn')}), backend=self.name)
