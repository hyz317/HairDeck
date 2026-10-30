"""Mask recipe: from SAM3 instance masks of many text prompts (and optionally a whole-character soft alpha from a
matting model) to the three output masks (person soft alpha, hair, face), at a working resolution.

cfg keys and defaults: see DEFAULT; the released preset is presets.json "best".
"""
import numpy as np
from scipy import ndimage

DEFAULT = dict(
    person_prompts=('person', 'anime character'), t_person=0.5,
    person_sel='all',          # all: union of all instances | main: instances overlapping the anchor (face or top)
    main_overlap=0.3,          # main: min fraction of the anchor covered by an instance
    cluster_ov=0.5,            # main: then add instances with at least this fraction of their area on the kept ones
                               # (iterated until nothing changes when cluster_iter is True)
    cluster_iter=False,
    cluster_alpha=0.0,         # >0: an instance added by the cluster step must also have a matting-alpha mean of at
                               # least this (both models agree it is foreground; effects and backdrops are not)
    dens_fill=0.0,             # >0: also add matting-alpha pixels (alpha > dens_t) missing from the SAM matte where
    dens_w=0.03, dens_t=0.5,   # the SAM matte covers at least dens_fill of a (dens_w * long side) window around them
    hair_prompts=('hair',), t_hair=0.5,
    hair_sel='all',            # all | main: instances touching the main person / face | main_in: instances with at
                               # least hair_in of their area on the main person / face dilated by 3% (all of them if
                               # none qualifies)
    hair_in=0.5,
    hair_in_dil=0.03,          # main_in: dilation of the main person / face, fraction of the long side, or (if > 1)
                               # in face heights given as 1 + value, e.g. 1.5 = half a face height
    hair_keep_face=False,      # main_in: also always keep the hair instances that touch the (2% dilated) face
    hair_grow=False,           # main_in: then also add dropped instances that touch the kept hair (1% dilation), repeated
    acc_prompts=(), t_acc=0.5, acc_touch=0.02, acc_max=0.5,  # accessories added to hair if touching hair|face
    acc_dy=None,               # head rule: the accessory centroid must lie at most this many face heights below the
                               # face centre (None = off; with no face, no accessory is added)
    acc_area=None,             # head rule: accessory area at most this many face areas (None = off)
    acc_exclude=(), t_excl=0.5,  # drop accessory instances lying mostly (> 50%) on instances of these prompts
                               # (animal ears: on the renders they are not part of the VRoid hair mesh)
    face_prompts=('face',), t_face=0.5,
    face_sel='top',            # top: highest score | largest | score_area: highest score x sqrt(area) | main: highest
                               # score inside the top-scoring person | main_big: highest score inside the largest person
    face_fallback=0.0,         # >0: if no face above t_face, take the best above this
    hair_fallback=0.0,         # >0: if no hair above t_hair, take instances above this
    box_fallback=0.0,          # >0: if neither hair nor face is found (no box possible), take the best hair and the
                               # best face above this lower score
    hair_minus=(), t_minus=0.5,  # drop hair instances (not the top one) that lie mostly (> 50%) on instances of these
                               # prompts (e.g. fur, animal tail: fox tails called hair)
    extra_prompts=(), t_extra=0.5, extra_alpha=0.5, extra_max=0.5,  # extra person parts (clothes, hand, ...) added
                               # when they touch the matte and the matting alpha agrees (mean above extra_alpha)
    hair_prune=0.0,            # >0: drop hair components smaller than this fraction of the hair area that lie more
    prune_dist=1.0,            # than prune_dist face heights from the largest hair component (stray detections such
                               # as a tassel on the chest that move the box); applied at full resolution via a region
    hair_low=0.0,              # >0: hysteresis, also take hair instances above this lower score when at least
    hair_low_ov=0.5,           # this fraction of their area overlaps the hair kept at t_hair (extensions, not new hair)
    hair_low_cc=False,         # True: from such an instance only take its connected parts that touch the kept hair
                               # (the instance can carry far-away bits, e.g. ornaments on the clothes, that move the box)
    close_r=0.006,             # binary closing radius, fraction of the long side (0 = off)
    fill='all',                # none | all | small | alpha | small_or_alpha
    fill_small=0.002,          # small: holes below this fraction of the image are filled
    fill_alpha=0.5,            # alpha: a hole is filled if the matting alpha mean inside it is above this
    matte='sam',               # sam | rematte (max(sam, alpha near sam)) | rematte_bin | alpha (matting model only)
                               # | inter (sam & alpha > t) | union (sam | alpha > t)
    near_r=0.05,               # rematte: band around the SAM matte where the alpha may add, fraction of long side
    alpha_t=0.5,
    alpha_cut=0.0,             # >0: remove SAM matte components (not touching hair / face) whose alpha mean is below
    fill_amax=1.0,             # alpha hole rule only for holes below this fraction of the image
    add='none',                # none | enclosed: add matting-alpha components missing from the SAM matte when most of
                               # their outer ring lies on the SAM matte or the image border (gaps, not props or effects)
    add_t=0.5, add_ratio=0.6, add_max=1.0,  # alpha threshold, min supported fraction of the ring, max area vs matte
    add_min_sam=0.0,           # and at least this fraction of the ring on the SAM matte itself (not the border)
    add_border=0.0,            # >0: also add matting-alpha components (alpha above this) that touch the image border
    add_border_sam=0.1,        # and have at least this fraction of their ring on the SAM matte (arms, clothes or hair
                               # leaving the frame, which the ring rule rejects because of their free sides)
    acc_to='hair',             # hair: accessories join the hair mask (and so the box) | person: only the person matte
    acc_cut=(), t_cut=0.5,     # cut instances of these prompts (head rule: centroid not below the face centre, at most
                               # 2.5 face areas) out of the person matte where they lie outside the hair, so that
                               # ornaments and animal ears do not show in the model input
    crop_mode='none',          # second SAM3 pass on the head crop, for the hair / face / accessory prompts:
                               # none | replace (crop instances only) | union (both passes) | paste (crop pass inside
                               # the crop, first pass outside it)
    crop_prompts='head',       # prompts taken from the crop pass: head (hair, face, accessories) | person (person and
                               # extra-part prompts, for a crop around the whole character) | all
    soft='none',               # none | alpha (matting alpha inside a thin band at the boundary) | feather (Gaussian
                               # blur of the binary matte, sigma soft_sigma px) | agree (the matting alpha wherever its
                               # 0.5 decision agrees with the binary matte, else the binary value): feather and agree
                               # keep the 0.5 level set, so the binary masks and the scores do not change
    soft_sigma=0.8,
    soft_band=0.004,
    no_person='ones',          # ones | empty | alpha: the matte when no person is found (the whole image, nothing,
                               # or the matting alpha)
)


def _disk(r):
    r = max(1, int(round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _dil(m, r):
    """Binary dilation by a (2r+1) square, outside = background: a separable max filter, same result as
    ndimage.binary_dilation(m, structure=_square(r)) but much faster for large r."""
    r = max(1, int(round(r)))
    return ndimage.maximum_filter(m.view(np.uint8), size=2 * r + 1, mode='constant', cval=0).view(bool)


def _ero(m, r):
    r = max(1, int(round(r)))
    return ndimage.minimum_filter(m.view(np.uint8), size=2 * r + 1, mode='constant', cval=0).view(bool)


def _square(r):
    r = max(1, int(round(r)))
    return np.ones((2 * r + 1, 2 * r + 1), bool)


def _union(ms, shape):
    out = np.zeros(shape, bool)
    for m in ms:
        out |= m
    return out


def _cands(inst, prompts, t):
    """(score, mask, key) of the instances of these prompts above t; key = (prompt, index in inst[prompt]), or the key
    stored with the instance (crop pass)."""
    return [(x[0], x[1], x[2] if len(x) > 2 else (p, j)) for p in prompts for j, x in enumerate(inst.get(p, []))
            if x[0] > t]


def merge_crop(inst, inst_crop, crop_box, prompts, mode, shape):
    """Instance lists of these prompts after merging the crop pass (see crop_mode, crop_prompts)."""
    out = dict(inst)
    if inst_crop is None or mode == 'none':
        return out
    inside = np.zeros(shape, bool)
    x0, y0, x1, y1 = crop_box
    inside[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    for p in prompts:
        full, cr = inst.get(p, []), [(s, m, ('crop', p, j)) for j, (s, m) in enumerate(inst_crop.get(p, []))]
        if mode == 'replace':
            out[p] = cr
        elif mode == 'union':
            out[p] = sorted([(s, m, (p, j)) for j, (s, m) in enumerate(full)] + cr, key=lambda x: -x[0])
        elif mode == 'paste':
            out[p] = sorted([(s, m & ~inside, (p, j)) for j, (s, m) in enumerate(full)] + cr, key=lambda x: -x[0])
    return out


def run(inst, shape, cfg, alpha=None, inst_crop=None, crop_box=None):
    """inst: {prompt: [(score, bool mask at working res), ...]} sorted by score; shape: (H, W) working res;
    alpha: optional (H, W) float32 whole-character alpha from a matting model. Returns person alpha (float32 in [0, 1]),
    hair (bool), face (bool), info dict."""
    c = dict(DEFAULT, **cfg)
    H, W = shape
    L = max(H, W)
    info = {}
    if c['crop_mode'] != 'none' and inst_crop is not None:
        head = list(c['hair_prompts']) + list(c['face_prompts']) + list(c['acc_prompts'])
        body = list(c['person_prompts']) + list(c['extra_prompts'])
        mp = {'head': head, 'person': body, 'all': list(dict.fromkeys(head + body))}[c['crop_prompts']]
        inst = merge_crop(inst, inst_crop, crop_box, mp, c['crop_mode'], shape)
    # face
    fc = _cands(inst, c['face_prompts'], c['t_face'])
    if not fc and c['face_fallback'] > 0:
        fc = _cands(inst, c['face_prompts'], c['face_fallback'])[:1]
    if not fc and c['box_fallback'] > 0 and not _cands(inst, c['hair_prompts'], c['t_hair']):
        fc = sorted(_cands(inst, c['face_prompts'], c['box_fallback']), key=lambda x: -x[0])[:1]
        info['box_fallback'] = True
    # person candidates
    pc = _cands(inst, c['person_prompts'], c['t_person'])
    face = np.zeros(shape, bool)
    info['face_keys'] = []
    if fc:
        if c['face_sel'] == 'largest':
            f = max(fc, key=lambda x: x[1].sum())
        elif c['face_sel'] == 'score_area':
            f = max(fc, key=lambda x: x[0] * np.sqrt(x[1].sum()))
        elif c['face_sel'] in ('main', 'main_big') and pc:
            top_p = max(pc, key=(lambda x: x[0]) if c['face_sel'] == 'main' else (lambda x: x[1].sum()))[1]
            inside = [x for x in fc if (x[1] & top_p).sum() > 0.5 * max(1, x[1].sum())]
            f = max(inside or fc, key=lambda x: x[0])
        else:
            f = max(fc, key=lambda x: x[0])
        face = f[1]
        info['face_keys'] = [f[2]]
    # person instances
    if c['person_sel'] == 'main' and pc:
        anchor = face if face.any() else max(pc, key=lambda x: x[0])[1]
        na = max(1, anchor.sum())
        keep = [x[1] for x in pc if (x[1] & anchor).sum() >= c['main_overlap'] * na]
        if not keep:
            keep = [max(pc, key=lambda x: x[0])[1]]
        # also keep instances that overlap the kept ones strongly (parts of the same character)
        base = _union(keep, shape)
        used = set()
        while True:
            changed = False
            for i, x in enumerate(pc):
                if i in used:
                    continue
                if (x[1] & base).sum() > c['cluster_ov'] * max(1, x[1].sum()):
                    if c['cluster_alpha'] > 0 and alpha is not None and alpha[x[1]].mean() < c['cluster_alpha']:
                        continue
                    base |= x[1]
                    used.add(i)
                    changed = True
            if not (c['cluster_iter'] and changed):
                break
        person0 = base
    else:
        person0 = _union([x[1] for x in pc], shape)
    info['n_person'] = len(pc)
    # hair
    hc = _cands(inst, c['hair_prompts'], c['t_hair'])
    if not hc and c['hair_fallback'] > 0:
        hc = _cands(inst, c['hair_prompts'], c['hair_fallback'])
    if not hc and info.get('box_fallback'):
        hc = sorted(_cands(inst, c['hair_prompts'], c['box_fallback']), key=lambda x: -x[0])[:1]
    if c['hair_sel'] == 'main' and (person0.any() or face.any()):
        ref = _dil(person0 | face, 0.01 * L)
        hc = [x for x in hc if (x[1] & ref).any()]
    elif c['hair_sel'] == 'main_in' and (person0.any() or face.any()) and len(hc) > 1:
        d = c['hair_in_dil']
        if d > 1 and face.any():
            ys_ = np.nonzero(face)[0]
            r = (d - 1) * (ys_.max() + 1 - ys_.min())
        else:
            r = (d if d <= 1 else 0.03) * L
        ref = _dil(person0 | face, r)
        fz = _dil(face, 0.02 * L) if (c['hair_keep_face'] and face.any()) else None
        keep = [x for x in hc if (x[1] & ref).sum() >= c['hair_in'] * max(1, x[1].sum())
                or (fz is not None and (x[1] & fz).any())]
        if keep and c['hair_grow']:
            changed = True
            while changed:
                changed = False
                kh = _dil(_union([x[1] for x in keep], shape), 0.01 * L)
                for x in hc:
                    if all(x[2] != y[2] for y in keep) and (x[1] & kh).any():
                        keep.append(x)
                        changed = True
        hc = keep or hc
    hair = _union([x[1] for x in hc], shape)
    info['hair_keys'] = [x[2] for x in hc]
    if c['hair_minus'] and len(hc) > 1:
        # instance level: drop hair instances (except the top one) lying mostly on fur / tail instances
        mc = _cands(inst, c['hair_minus'], c['t_minus'])
        if mc:
            mu = _union([x[1] for x in mc], shape)
            top = max(hc, key=lambda x: x[0])
            keep = [x for x in hc if x is top or (x[1] & mu).sum() < 0.5 * max(1, x[1].sum())]
            if len(keep) < len(hc):
                hc = keep
                hair = _union([x[1] for x in hc], shape)
                info['hair_keys'] = [x[2] for x in hc]
                info['hair_minus_n'] = len(mc)
    if c['hair_low'] > 0 and hair.any():
        have = {x[2] for x in hc}
        base_hair = hair.copy()
        region = np.zeros(shape, bool)
        for x in _cands(inst, c['hair_prompts'], c['hair_low']):
            if x[2] not in have and (x[1] & base_hair).sum() >= c['hair_low_ov'] * max(1, x[1].sum()):
                if c['hair_low_cc']:
                    lab, n = ndimage.label(x[1])
                    ok = np.zeros(n + 1, bool)
                    ok[np.unique(lab[x[1] & base_hair])] = True
                    ok[0] = False
                    part = ok[lab]
                    hair = hair | part
                    region |= part
                    info['hair_keys'].append(('cc',) + tuple(x[2]))
                else:
                    hair = hair | x[1]
                    info['hair_keys'].append(x[2])
        if region.any():
            info['hair_cc_region'] = _dil(region, 2)
    if c['hair_prune'] > 0 and hair.any():
        lab, n = ndimage.label(hair)
        if n > 1:
            idx = np.arange(1, n + 1)
            area = ndimage.sum(hair, lab, idx)
            big = int(np.argmax(area)) + 1
            if face.any():
                ys_ = np.nonzero(face)[0]
                ref = ys_.max() + 1 - ys_.min()
            else:
                ref = np.sqrt(area.max())
            dist = ndimage.distance_transform_edt(lab != big)
            mind = ndimage.minimum(dist, lab, idx)
            drop = (area < c['hair_prune'] * hair.sum()) & (mind > c['prune_dist'] * ref)
            if drop.any():
                keep = np.zeros(n + 1, bool)
                keep[1:] = ~drop
                hair = keep[lab]
                info['hair_region'] = _dil(hair, 2)
    # accessories
    if c['acc_prompts']:
        ac = _cands(inst, c['acc_prompts'], c['t_acc'])
        if ac and (hair.any() or face.any()):
            ref = _dil(hair | face, c['acc_touch'] * L)
            hf = max(1, (hair | face).sum())
            acc = np.zeros(shape, bool)
            fy = fh = fa = None
            if face.any():
                ys_, xs_ = np.nonzero(face)
                fy, fh, fa = (ys_.min() + ys_.max() + 1) / 2, ys_.max() + 1 - ys_.min(), face.sum()
            excl = _union([x[1] for x in _cands(inst, c['acc_exclude'], c['t_excl'])], shape) if c['acc_exclude'] else None
            for s, m, k in ac:
                if excl is not None and (m & excl).sum() > 0.5 * max(1, m.sum()):
                    continue
                if c['acc_dy'] is not None or c['acc_area'] is not None:
                    if fy is None:
                        continue
                    if c['acc_dy'] is not None and (np.nonzero(m)[0].mean() - fy) / fh > c['acc_dy']:
                        continue
                    if c['acc_area'] is not None and m.sum() > c['acc_area'] * fa:
                        continue
                if (m & ref).any() and m.sum() <= c['acc_max'] * hf:
                    acc |= m
                    if c['acc_to'] == 'hair':
                        info['hair_keys'].append(k)
            if c['acc_to'] == 'hair':
                hair = hair | acc
            info['acc'] = acc
    # person matte
    if not pc and not person0.any():
        if c['no_person'] == 'alpha' and alpha is not None and (alpha > 0.5).any():
            m = alpha > 0.5
        else:
            info['status'] = 'no_person_mask'
            fill = 0.0 if c['no_person'] == 'empty' else 1.0
            return np.full(shape, fill, np.float32), hair, face, info
    else:
        m = person0 | hair | face
        if 'acc' in info:
            m = m | info['acc']
    if c['extra_prompts'] and alpha is not None and m.any():
        near = _dil(m, 0.01 * L)
        for x in _cands(inst, c['extra_prompts'], c['t_extra']):
            e = x[1]
            if (e & near).any() and e.sum() <= c['extra_max'] * m.sum() and alpha[e].mean() >= c['extra_alpha']:
                m = m | e
    if c['close_r'] > 0:
        r = max(3, round(c['close_r'] * L))
        m = ndimage.binary_closing(m, structure=_square(r), border_value=0)
    if c['fill'] != 'none':
        filled = ndimage.binary_fill_holes(m)
        holes = filled & ~m
        if c['fill'] == 'all' or not holes.any():
            m = filled
        else:
            lab, n = ndimage.label(holes)
            idx = np.arange(1, n + 1)
            area = ndimage.sum(holes, lab, idx)
            ok = np.zeros(n + 1, bool)
            if 'small' in c['fill']:
                ok[1:] |= area < c['fill_small'] * H * W
            if 'alpha' in c['fill'] and alpha is not None:
                am = ndimage.mean(alpha, lab, idx)
                ok[1:] |= (am > c['fill_alpha']) & (area < c['fill_amax'] * H * W)
            m = m | ok[lab]
    if c['dens_fill'] > 0 and alpha is not None and m.any():
        w = max(3, int(round(c['dens_w'] * L)) | 1)
        dens = ndimage.uniform_filter(m.astype(np.float32), size=w, mode='constant', cval=0.0)
        m = m | ((alpha > c['dens_t']) & (dens >= c['dens_fill']))
    if c['add'] == 'enclosed' and alpha is not None and m.any():
        cand = (alpha > c['add_t']) & ~m
        lab, n = ndimage.label(cand)
        if n:
            idx = np.arange(1, n + 1)
            labp = np.pad(lab, 1)
            mp = np.pad(m, 1, constant_values=True)  # the image border counts as support
            labd = ndimage.grey_dilation(labp, size=(3, 3))
            ring = (labd > 0) & (labp == 0)
            tot = ndimage.sum(ring, labd, idx)
            sup = ndimage.sum(ring & mp, labd, idx)
            area = ndimage.sum(cand, lab, idx)
            ok = np.zeros(n + 1, bool)
            ok[1:] = (sup >= c['add_ratio'] * np.maximum(tot, 1)) & (area <= c['add_max'] * m.sum())
            if c['add_min_sam'] > 0:
                on_sam = ndimage.sum(ring & np.pad(m, 1), labd, idx)
                ok[1:] &= on_sam >= c['add_min_sam'] * np.maximum(tot, 1)
            m = m | ok[lab]
    if c['add_border'] > 0 and alpha is not None and m.any():
        cand = (alpha > c['add_border']) & ~m
        lab, n = ndimage.label(cand)
        if n:
            idx = np.arange(1, n + 1)
            edge = np.zeros(shape, bool)
            edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
            touch = ndimage.sum(edge, lab, idx) > 0
            labd = ndimage.grey_dilation(lab, size=(3, 3))
            ring = (labd > 0) & (lab == 0)
            tot = ndimage.sum(ring, labd, idx)
            on_sam = ndimage.sum(ring & m, labd, idx)
            area = ndimage.sum(cand, lab, idx)
            ok = np.zeros(n + 1, bool)
            ok[1:] = touch & (on_sam >= c['add_border_sam'] * np.maximum(tot, 1)) & (area <= c['add_max'] * m.sum())
            m = m | ok[lab]
    if c['alpha_cut'] > 0 and alpha is not None:
        lab, n = ndimage.label(m)
        if n > 1:
            idx = np.arange(1, n + 1)
            am = ndimage.mean(alpha, lab, idx)
            hf_lab = np.unique(lab[hair | face])
            drop = np.zeros(n + 1, bool)
            drop[1:] = am < c['alpha_cut']
            drop[hf_lab] = False
            drop[0] = False
            m = m & ~drop[lab]
    mt = c['matte']
    a = m.astype(np.float32)
    if alpha is not None and mt != 'sam':
        if mt in ('rematte', 'rematte_bin'):
            near = ndimage.distance_transform_edt(~m) <= max(3, round(c['near_r'] * L)) if m.any() else np.ones(shape, bool)
            if mt == 'rematte':
                a = np.maximum(a, alpha * near)
            else:
                a = np.maximum(a, ((alpha > c['alpha_t']) & near).astype(np.float32))
        elif mt == 'alpha':
            a = alpha.copy()
        elif mt == 'inter':
            keep = (alpha > c['alpha_t']) | hair | face
            a = (m & keep).astype(np.float32)
        elif mt == 'union':
            a = np.maximum(a, (alpha > c['alpha_t']).astype(np.float32))
    if c['soft'] == 'feather':
        b = a > 0.5
        a = ndimage.gaussian_filter(b.astype(np.float32), c['soft_sigma'])
        a = np.where(b, np.maximum(a, 0.5 + 1e-3), np.minimum(a, 0.5 - 1e-3))
    elif c['soft'] == 'agree' and alpha is not None:
        b = a > 0.5
        al = np.clip(alpha, 0, 1)
        a = np.where((al > 0.5) == b, al, b.astype(np.float32))
    if c['soft'] == 'alpha' and alpha is not None:
        b = a > 0.5
        band = _dil(b, c['soft_band'] * L) & ~_ero(b, c['soft_band'] * L)
        a = np.where(band, np.clip(alpha, 0, 1), a)
    if alpha is not None:  # agreement of the final matte with the matting model: a confidence signal for the demo
        b, q = a > 0.5, alpha > 0.5
        info['agree'] = float((b & q).sum() / max(1, (b | q).sum()))
    if c['acc_cut'] and face.any():
        ys_, xs_ = np.nonzero(face)
        fy, fh, fa = (ys_.min() + ys_.max() + 1) / 2, ys_.max() + 1 - ys_.min(), face.sum()
        cut = np.zeros(shape, bool)
        for s, mk, k in _cands(inst, c['acc_cut'], c['t_cut']):
            if (np.nonzero(mk)[0].mean() - fy) / fh <= 0.0 and mk.sum() <= 2.5 * fa:
                cut |= mk
        cut &= ~_dil(hair | face, 2)
        if cut.any():
            a = np.where(cut, 0.0, a)
            info['cut_px'] = int(cut.sum())
    return a.astype(np.float32), hair, face, info
