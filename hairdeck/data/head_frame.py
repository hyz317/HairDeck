"""Head coordinate frame of the image DiTs and the head-frame chain DiT.

The dataset (CHARM) frame normalises the whole character: hair top near y = 0.5, so the same hairstyle is 2.6x larger
on a chibi than on a tall character, and a hair + face crop cannot tell which. The head frame expresses hair relative
to the head instead: unit = face-mesh width w, origin = anchor() of the rest-pose face-mesh bbox, axes = dataset axes
(the rest pose is upright and faces +z). Every head is mapped onto the canonical (median train) head, so coordinate and
card-size distributions stay close to the dataset's and the card VAE still fits:
  p_head = k * p + b,  k = W / w,  b = O - k * anchor      (W, O = median face width and anchor over the train fits)

fit_one: per character, the Blender rest-pose hair bbox (render meta.json, Z up) is matched to the dataset hair.glb
bbox (Y up) to get the Blender -> dataset similarity (scale, shift), then the face bbox and the head joint are mapped
through it. The head-frame data (k, b, token stats, latents of the cards scaled by k): root_stats, encode_head. Drivers:
scripts/data.py (step head: fits, k, b, token stats) and scripts/train/export_latents.py (latents).
"""
import json
import os

import numpy as np

# axis signs of the glTF export (dataset = s * (Bx, Bz, -By) + t), verified on the vertices of 5 test characters;
# a bbox alone cannot tell the signs apart
SX, SZ = 1, -1


def glb_bbox(path):
    import trimesh
    sc = trimesh.load(path)
    V = np.concatenate([g.vertices for g in (sc.dump() if isinstance(sc, trimesh.Scene) else [sc])])
    return V.min(0), V.max(0)


def axis_map(p, sx=SX, sz=SZ):
    """Blender (x, y, z), Z up, -Y front -> dataset axes (x, y, z), Y up, before scale / shift."""
    p = np.asarray(p, np.float64)
    return np.stack([sx * p[..., 0], p[..., 2], sz * p[..., 1]], -1)


def items_of(meta, render_root):
    """(key, dataset dir, render meta.json) of the characters of meta that have a render and a hair.glb."""
    out = []
    for e in meta:
        d = os.path.join(render_root, e['key'].replace('/', '_'))
        if os.path.exists(f'{d}/meta.json') and os.path.exists(f"{e['dir']}/hair.glb"):
            out.append((e['key'], e['dir'], f'{d}/meta.json'))
    return out


def fit_one(item):
    """item = (key, dataset dir with hair.glb, render meta.json) -> (key, fit dict); status 'ok', 'no_face' or 'error'.
    Fit: o (face bbox centre), w / h / d (face bbox size), head_joint, hair bbox, s / t (Blender -> dataset), the
    per-axis scales s_axes and the bbox residual (both diagnostics of the similarity)."""
    key, ddir, mpath = item
    try:
        m = json.load(open(mpath))
        hb, fb = np.array(m['hair_rest_bbox']), m.get('face_rest_bbox')
        if fb is None:
            return key, dict(status='no_face')
        fb = np.array(fb)
        dlo, dhi = glb_bbox(f'{ddir}/hair.glb')
        c = axis_map(hb)
        lo, hi = c.min(0), c.max(0)
        s_axes = (dhi - dlo) / np.maximum(hi - lo, 1e-9)
        s = float(np.median(s_axes))
        t = (dlo + dhi) / 2 - s * (lo + hi) / 2
        resid = float((np.abs(s * lo + t - dlo) + np.abs(s * hi + t - dhi)).sum() / (dhi - dlo).max())
        sim = dict(sx=SX, sz=SZ, s=s, t=t.tolist(), s_axes=s_axes.tolist(), resid=resid)
        fc = axis_map((fb[0] + fb[1]) / 2) * s + t
        fe = np.abs(axis_map(fb[1] - fb[0])) * s
        hj = axis_map(np.array(m['head_rest_matrix'])[:3, 3]) * s + t  # head bone rest matrix: translation = head joint
        return key, dict(status='ok', o=fc.tolist(), w=float(fe[0]), h=float(fe[1]), d=float(fe[2]), head_joint=hj.tolist(),
                         hair_lo=dlo.tolist(), hair_hi=dhi.tolist(), **sim)
    except Exception as e:  # noqa: BLE001
        return key, dict(status='error', err=repr(e)[:200])


def anchor(f):
    """Head-frame origin: face-bbox centre in x and z, face-bbox top in y (cuts the between-character spread of the
    scalp centroid height from 0.024 to 0.010, robust std, compared with the bbox centre)."""
    o = np.array(f['o'], np.float64)
    return np.array([o[0], o[1] + f['h'] / 2, o[2]])


def canonical_head(fits):
    """W, O = median face width and median anchor over the 'ok' fits (dict key -> fit)."""
    ok = [f for f in fits.values() if f['status'] == 'ok']
    return float(np.median([f['w'] for f in ok])), np.median(np.array([anchor(f) for f in ok]), 0)


def params(meta, fits, W, O):
    """Per hairstyle of meta (list of {'key': ...}): valid (bool), k (float32), b (3,) float32 of p_head = k * p + b."""
    n = len(meta)
    valid, k, b = np.zeros(n, bool), np.ones(n, np.float32), np.zeros((n, 3), np.float32)
    for i, e in enumerate(meta):
        f = fits.get(e['key'])
        if f is None or f['status'] != 'ok':
            continue
        valid[i] = True
        k[i] = W / f['w']
        b[i] = O - k[i] * anchor(f)
    return valid, k, b


def card_hair(hair_ptr):
    """The hairstyle of every card."""
    return np.repeat(np.arange(len(hair_ptr) - 1), np.diff(hair_ptr))


def root_stats(st, roots_h, vc):
    """Token stats st with root_mu / root_sd replaced by those of the head-frame card roots roots_h[vc]."""
    return dict(st, root_mu=roots_h[vc].mean(0).tolist(), root_sd=roots_h[vc].std(0).tolist())


def encode_head(vae, X, lens, lat, kc, idx):
    """lat with the rows idx replaced by the latents of the cards X[idx] scaled by kc[idx]."""
    out = lat.copy()
    out[idx] = vae.encode_padded(X[idx] * kc[idx, None, None], lens[idx])
    return out


def recon_err(vae, X, lens, z):
    """Mean xyz error (relative to card length) and width error of decode(z) vs X, both resampled to 32 points."""
    import torch
    from ..cards.codec import arclen, resample
    seqs = vae.decode_cards(torch.from_numpy(z).to(vae.device), np.zeros((len(z), 3)))
    ex, ew = [], []
    for s, x, n in zip(seqs, X, lens):
        a, g = resample(np.asarray(s, np.float64), 32), resample(x[:n].astype(np.float64), 32)
        L = max(arclen(g[:, :3])[-1], 1e-6)
        ex.append(np.linalg.norm(a[:, :3] - g[:, :3], axis=1).mean() / L)
        ew.append(np.abs(a[:, 3] - g[:, 3]).mean() / max(g[:, 3].mean(), 1e-6))
    return float(np.mean(ex)), float(np.mean(ew))
