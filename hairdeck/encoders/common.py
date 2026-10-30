"""Point-cloud condition inputs: PLY reading, the condition point clouds of a split, and the point-cloud-only sharp-edge
proxy. Only numpy / torch; nothing from upstream encoder repositories."""
import json
import os

import numpy as np
import torch

from .. import paths

NPTS = 10000


def condition_plys(split):
    """The condition point cloud of every row of a split: train, <dir>/hair_reform.ply of every data/train_meta.json
    entry; test, CHARM's released test point clouds data/charm_test_pc/{idx:03d}.ply (pc/ of the dataset repository,
    byte-equal to the hair_reform.ply of the test rows), so the test condition needs no dataset."""
    if split == 'test':
        d = paths.data('charm_test_pc')
        n = len([f for f in os.listdir(d) if f.endswith('.ply')])
        fns = [os.path.join(d, f'{i:03d}.ply') for i in range(n)]
        missing = [f for f in fns if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f'{d}: expected {n} files 000.ply ..., missing {missing[:3]}')
        return fns
    return [os.path.join(d, 'hair_reform.ply') for d in meta_dirs(split)]


def read_ply_xyzn(fn):
    """hair_reform.ply (binary little endian, double x y z nx ny nz + uchar rgb) -> float64 [N, 6]."""
    with open(fn, 'rb') as f:
        raw = f.read()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    hdr = raw[:end].decode('ascii')
    assert 'format binary_little_endian' in hdr
    props = [l.split()[-1] for l in hdr.split('\n') if l.startswith('property')]
    assert props[:6] == ['x', 'y', 'z', 'nx', 'ny', 'nz'] and props[6:] == ['red', 'green', 'blue'], props
    n = int([l for l in hdr.split('\n') if l.startswith('element vertex')][0].split()[-1])
    dt = np.dtype([('p', '<f8', 6), ('c', 'u1', 3)])
    return np.frombuffer(raw[end:end + n * dt.itemsize], dtype=dt)['p'].copy()


def load_points(fn):
    """A condition point cloud (hair_reform.ply format) -> float32 [N, 6]: positions * 2 (dataset frame -> about
    [-1, 1]), normals unchanged, no per-sample normalisation."""
    x = read_ply_xyzn(fn)
    x[:, :3] *= 2.0
    return x.astype(np.float32)


def condition_points(fn):
    """load_points of one condition point cloud with exactly NPTS points (the released clouds all have 10,000; another
    count is resampled with np.random.default_rng(0), with replacement if there are fewer)."""
    x = load_points(fn)
    if len(x) != NPTS:
        x = x[np.random.default_rng(0).choice(len(x), NPTS, replace=len(x) < NPTS)]
    return x


def meta_dirs(split):
    return [m['dir'] for m in json.load(open(paths.data(f'{split}_meta.json')))]


@torch.no_grad()
def sharp_proxy(pts, m=2048, k=16):
    """Point-cloud-only stand-in for the mesh sharp-edge samples that Dora expects in its second branch (not used by
    the released condition cache, which feeds the same points to both branches: --sharp copy).
    Score of a point = max over its k nearest neighbours of (1 - |n_i . n_j|). Returns the m highest-scoring points
    [B, m, 6] (stable order: score desc, then index asc)."""
    out = []
    for b in range(pts.shape[0]):
        p = pts[b].float()
        d = torch.cdist(p[:, :3], p[:, :3])
        nn_idx = d.topk(k + 1, largest=False).indices[:, 1:]
        n = p[:, 3:]
        s = (1 - (n[:, None, :] * n[nn_idx]).sum(-1).abs()).amax(-1)
        order = torch.sort(-s, stable=True).indices[:m]
        out.append(p[order])
    return torch.stack(out)
