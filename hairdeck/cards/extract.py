"""Card templates from the original hair meshes (hair.glb), without CHARM code.

Per card:
  - control points = centroids of the cross-section rings walked from one end (topology.card_sections),
  - root width direction from the root cross-section ring, width / thickness = ring extent along the Bishop frame
    (N_i, N_i x T_i) transported from that root direction (the frame of the mesher, mesh.bishop_mesh).
Cards are oriented root->tip by the dataset's rule (root = endpoint closer to (0, p95 of the per-card max y, 0)), or,
with ref_roots, by the nearer reference root. Driver: scripts/data.py (step cards).
"""
import os

import numpy as np
import trimesh

from .frames import bishop, perp, tangents, true_dir_wt
from .topology import card_sections

SHAPES = {'square': 0, 'triangle': 1}


def ring_extent(r, c, n, b):
    if len(r) < 2:
        return 0.0, 0.0
    q = r - c
    pn, pb = q @ n, q @ b
    return float(pn.max() - pn.min()), float(pb.max() - pb.min())


def load_cards(d):
    """hair.glb -> list of cleaned single-card meshes (vertices merged at 1e-8, connected parts, unique non-degenerate faces)."""
    scene = trimesh.load(os.path.join(d, 'hair.glb'))
    meshes = scene.dump() if isinstance(scene, trimesh.Scene) else [scene]
    cards = []
    for mesh in meshes:
        unique, group = trimesh.grouping.group_distance(mesh.vertices, 1e-8)
        idx = np.zeros(len(mesh.vertices), dtype=int)
        for k, g in enumerate(group):
            idx[g] = k
        cards += trimesh.Trimesh(vertices=unique, faces=idx[mesh.faces]).split()
    for card in cards:
        card.update_faces(card.unique_faces())
        card.update_faces(card.nondegenerate_faces())
    return cards


def extract(d, ref_roots=None):
    """List of (shape_id, seq (n,5) float64, root width dir (3,), status) for one hairstyle directory.
    ref_roots: optional (m,3) root points; each card's root is then its endpoint nearer to a reference root."""
    raw = []
    for card in load_cards(d):
        shape, p, rings = card_sections(card)
        raw.append((shape, p, rings))
    max_y = np.percentile([p[:, 1].max() for _, p, _ in raw], 95)
    anchor = np.array([0.0, max_y, 0.0])
    ref = None if ref_roots is None else np.asarray(ref_roots, np.float64)
    out = []
    for shape, p, rings in raw:
        if ref is not None:
            flip = np.linalg.norm(ref - p[-1], axis=1).min() < np.linalg.norm(ref - p[0], axis=1).min()
        else:
            flip = np.linalg.norm(p[0] - anchor) > np.linalg.norm(p[-1] - anchor)
        if flip:
            p, rings = p[::-1].copy(), rings[::-1]
        if len(p) < 3:
            out.append((SHAPES[shape], np.concatenate([p, np.zeros((len(p), 2))], 1), np.array([1.0, 0, 0]), 'short'))
            continue
        T = tangents(p)
        k0 = next((i for i, r in enumerate(rings) if len(r) >= 3), None)
        if k0 is None:
            n0, status = perp(np.cross(T[0], [0, 1, 0]) + 1e-6, T[0]), 'noring'
        else:
            n0, status = true_dir_wt(rings[k0])[0], 'ok'
        N = bishop(n0, T)
        wt = np.array([ring_extent(r, p[i], N[i], np.cross(N[i], T[i])) for i, r in enumerate(rings)])
        out.append((SHAPES[shape], np.concatenate([p, wt], 1), N[0], status))
    return out


def work(item):
    """Pool worker: (key, dir) -> (key, dir, cards or None, error message or None)."""
    key, d = item
    try:
        return key, d, extract(d), None
    except Exception as e:  # noqa
        return key, d, None, repr(e)[:200]


def pack(keys, res):
    """Ordered keys + {key: (dir, cards)} -> (npz arrays dict, meta list, card status counts)."""
    pts, card_len, card_shape, card_nrm, hair_nc, mt, status = [], [], [], [], [], [], {}
    for k in keys:
        d, cards = res[k]
        hair_nc.append(len(cards))
        mt.append({'key': k, 'dir': d, 'n_cards': len(cards)})
        for sh, seq, n0, st in cards:
            pts.append(seq.astype(np.float32))
            card_len.append(len(seq))
            card_shape.append(sh)
            card_nrm.append(n0)
            status[st] = status.get(st, 0) + 1
    arrays = dict(points=np.concatenate(pts), card_ptr=np.concatenate([[0], np.cumsum(card_len)]).astype(np.int64),
                  card_shape=np.asarray(card_shape, np.uint8), card_nrm=np.asarray(card_nrm, np.float32),
                  hair_ptr=np.concatenate([[0], np.cumsum(hair_nc)]).astype(np.int64))
    return arrays, mt, status
