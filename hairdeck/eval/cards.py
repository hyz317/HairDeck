"""Card-level evaluation of a generated hairstyle against its reference cards.

  card level (CHARM Table 4 style): per-card uniform samples (256 points); each predicted card is matched to the
      reference card with minimum Chamfer distance and vice versa; CD / EMD / Hausdorff / voxel IoU averaged over the
      matched pairs; root-direction angle to the matched reference card (median, share > 30 deg)
  roots: recall / precision of card roots within 2 cm, mean root distance, root voxel coverage
  duplicates: share of predicted cards whose nearest other predicted card is within CD tau
  interpenetration: share of card pairs whose meshes intersect, share of cards hitting any other
  folds: median of the per-card maximum turning angle, share of cards with a turn > 90 deg
  counts: n_pred, n_gt
Every function that meshes cards takes `mesher` (default hairdeck.cards.mesh.card_mesh_vf, 8-column cards only); CHARM's
5-column reference cards need CHARM's own mesher.
run_eval evaluates the hairstyles of a run (RUN/JsonResults/{idx:03d}.json) in worker processes, optionally with the
point metrics of its CHARM-protocol point clouds (RUN/PointClouds/{idx:03d}.ply, hairdeck.sampling.scan) against
reference point clouds of the original meshes (TDIR/{idx:03d}.ply); summarize gives the means.
"""
import json
import os

import numpy as np

from .. import paths
from ..cards.codec import root_normal_of
from ..cards.mesh import card_mesh_vf
from .metrics import earth_movers_distance, point_metrics, read_ply_points, uniform_sample, voxel_iou
from .pool import worker_pool

NPC = 256


def load_cards(path):
    """CHARM-format json -> list of (seq float64, shape id) with >= 3 finite points."""
    out = []
    for c in json.load(open(path)):
        s = np.asarray(c['seq'], dtype=np.float64)
        if s.ndim == 2 and len(s) >= 3 and np.isfinite(s).all():
            out.append((s, 1 if c.get('shape') == 'triangle' else 0))
    return out


def card_points(cards, seed=0, mesher=None):
    """(C, NPC, 3) uniform samples per card (card k with seed + k) and the per-card (vertices, faces)."""
    import trimesh
    mesher = mesher or card_mesh_vf
    pts, vf = [], []
    for k, (s, sh) in enumerate(cards):
        v, f = mesher(s, sh)
        m = trimesh.Trimesh(v, f, process=False)
        m.update_faces(m.nondegenerate_faces())
        if len(m.faces) == 0:
            pts.append(np.repeat(s[:1, :3], NPC, 0))
        else:
            pts.append(uniform_sample(m, NPC, seed=seed + k))
        vf.append((np.asarray(m.vertices), np.asarray(m.faces)))
    return np.stack(pts) if pts else np.zeros((0, NPC, 3)), vf


def pair_cd(A, B):
    """A (P,n,3), B (G,n,3) -> (P,G) chamfer (sum of two directed means), plus directed maxima for Hausdorff."""
    import torch
    a = torch.from_numpy(A).float()
    b = torch.from_numpy(B).float().reshape(-1, 3)
    G = B.shape[0]
    cd = np.zeros((A.shape[0], G))
    hd = np.zeros((A.shape[0], G))
    for p in range(A.shape[0]):
        d = torch.cdist(a[p], b).reshape(a.shape[1], G, -1)  # (n, G, n)
        d_ab = d.min(2).values  # (n,G)
        d_ba = d.min(0).values  # (G,n)
        cd[p] = (d_ab.mean(0) + d_ba.mean(1)).numpy()
        hd[p] = torch.maximum(d_ab.max(0).values, d_ba.max(1).values).numpy()
    return cd, hd


def seg_tri_any(P0, P1, V0, V1, V2, eps=1e-14):
    D = P1 - P0
    E1, E2 = V1 - V0, V2 - V0
    for s in range(0, len(P0), 256):
        d, p0 = D[s:s + 256, None], P0[s:s + 256, None]
        pvec = np.cross(d, E2[None])
        det = (E1[None] * pvec).sum(-1)
        ok = np.abs(det) > eps
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = p0 - V0[None]
        u = (tvec * pvec).sum(-1) * inv
        qvec = np.cross(tvec, E1[None])
        v = (d * qvec).sum(-1) * inv
        t = (E2[None] * qvec).sum(-1) * inv
        if np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)):
            return True
    return False


def mesh_edges(f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def interpenetration(vf):
    """Share of card pairs whose meshes intersect (edge-triangle test both ways), and share of cards hitting any other.
    Per pair, only edges/triangles inside the other card's AABB are tested (exact, just pruned)."""
    n = len(vf)
    lo = np.array([v.min(0) if len(v) else np.full(3, np.inf) for v, _ in vf])
    hi = np.array([v.max(0) if len(v) else np.full(3, -np.inf) for v, _ in vf])
    edges = [mesh_edges(f) if len(f) else np.zeros((0, 2), int) for _, f in vf]
    seg = [(v[e[:, 0]], v[e[:, 1]]) for (v, _), e in zip(vf, edges)]
    tri = [(v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]) if len(f) else None for v, f in vf]
    seg_lo = [np.minimum(a, b) for a, b in seg]
    seg_hi = [np.maximum(a, b) for a, b in seg]
    tri_lo = [np.minimum(np.minimum(t[0], t[1]), t[2]) if t else None for t in tri]
    tri_hi = [np.maximum(np.maximum(t[0], t[1]), t[2]) if t else None for t in tri]

    def test(i, j):
        bl, bh = np.maximum(lo[i], lo[j]), np.minimum(hi[i], hi[j])  # overlap box
        es = np.all(seg_hi[i] >= bl, 1) & np.all(seg_lo[i] <= bh, 1)
        ts = np.all(tri_hi[j] >= bl, 1) & np.all(tri_lo[j] <= bh, 1)
        if not es.any() or not ts.any():
            return False
        return seg_tri_any(seg[i][0][es], seg[i][1][es], tri[j][0][ts], tri[j][1][ts], tri[j][2][ts])

    hit = np.zeros(n, bool)
    npairs = 0
    for i in range(n):
        if tri[i] is None:
            continue
        ov = np.nonzero(np.all(lo[i] <= hi, 1) & np.all(hi[i] >= lo, 1))[0]
        for j in ov[ov > i]:
            if tri[j] is None:
                continue
            if test(i, j) or test(j, i):
                npairs += 1
                hit[i] = hit[j] = True
    tot = n * (n - 1) / 2
    return dict(pair_ratio=npairs / tot if tot else 0.0, card_ratio=float(hit.mean()) if n else 0.0)


def root_dir(seq):
    """Root width direction of a card: 8-column cards store it in columns 5:8; CHARM's 5-column cards encode it in the
    positions of the first control points (smallest PCA axis of the first 3 points)."""
    if seq.shape[1] >= 8:
        n = seq[0, 5:8]
        return n / (np.linalg.norm(n) + 1e-12)
    return root_normal_of(seq)


def card_level(pred, gt, Pp, Pg, fast=False):
    cd, hd = pair_cd(Pp, Pg)
    m_pg, m_gp = cd.argmin(1), cd.argmin(0)
    pairs = [(p, m_pg[p]) for p in range(len(pred))] + [(m_gp[g], g) for g in range(len(gt))]
    out = {'cd': [], 'hd': [], 'emd': [], 'iou': [], 'ang': []}
    no_emd = os.environ.get('HAIRDECK_NO_EMD') or os.environ.get('CHARM_EVAL_NO_EMD')
    for p, g in pairs:
        out['cd'].append(cd[p, g])
        out['hd'].append(hd[p, g])
        if not fast and not no_emd:
            out['emd'].append(float(earth_movers_distance(Pp[p], Pg[g])))
        out['iou'].append(voxel_iou(Pp[p], Pg[g]))
    for p in range(len(pred)):
        a, b = root_dir(pred[p][0]), root_dir(gt[m_pg[p]][0])
        out['ang'].append(float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))))
    res = {k: float(np.mean(v)) for k, v in out.items() if k != 'ang' and v}
    res.update({'ang_med': float(np.median(out['ang'])), 'ang_gt30': float(np.mean(np.array(out['ang']) > 30))})
    return res


def turn_max(seq):
    """Max turning angle (deg) between consecutive segments of a card's control polyline (zigzag / fold-back check)."""
    d = np.diff(seq[:, :3], axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-9]
    if len(d) < 2:
        return 0.0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))).max())


def roots_metrics(pred, gt, r=0.02):
    from scipy.spatial import cKDTree
    rp = np.array([s[0, :3] for s, _ in pred])
    rg = np.array([s[0, :3] for s, _ in gt])
    dpg = cKDTree(rg).query(rp)[0]
    dgp = cKDTree(rp).query(rg)[0]
    vg = set(map(tuple, np.floor(rg * 32).astype(int)))
    vp = set(map(tuple, np.floor(rp * 32).astype(int)))
    return dict(root_recall=float(np.mean(dgp < r)), root_precision=float(np.mean(dpg < r)), root_dist=float(dpg.mean()),
                root_vox_cov=len(vg & vp) / len(vg))


def dup_ratio(Pc, taus=(0.002, 0.004)):
    if len(Pc) < 2:
        return {f'dup@{t}': 0.0 for t in taus}
    cd, _ = pair_cd(Pc, Pc)
    np.fill_diagonal(cd, np.inf)
    nn = cd.min(1)
    return {f'dup@{t}': float(np.mean(nn < t)) for t in taus}


class GTCards:
    """Reference cards of the test set: the cards extracted from the original hair meshes of every test case (the
    combined-template cases: the cards of their two template meshes), one CHARM-format json per test row,
    data/test_cards_orig/{idx:03d}.json (8 columns)."""

    def __init__(self, root=None):
        self.root = root or paths.data('test_cards_orig')

    def cards(self, i):
        return load_cards(os.path.join(self.root, f'{i:03d}.json'))


def evaluate_sample(pred, gt, fast=False, with_gt=False, mesher=None):
    """All card-level metrics of one generated hairstyle (pred, gt: lists of (seq, shape id)); cards with < 3 points
    in gt are dropped first, as in the stored evaluations."""
    gt = [(s, sh) for s, sh in gt if len(s) >= 3]
    out = dict(n_pred=len(pred), n_gt=len(gt))
    if len(pred) == 0:
        return out
    Pp, vfp = card_points(pred, seed=0, mesher=mesher)
    Pg, vfg = card_points(gt, seed=100000, mesher=mesher)
    out.update({f'card_{k}': v for k, v in card_level(pred, gt, Pp, Pg, fast).items()})
    out.update(roots_metrics(pred, gt))
    out.update(dup_ratio(Pp))
    if not fast:
        out.update({f'inter_{k}': v for k, v in interpenetration(vfp).items()})
    turns = [turn_max(s) for s, _ in pred]
    out.update(turn_max_p50=float(np.median(turns)), fold90=float(np.mean(np.array(turns) > 90)))
    if with_gt:
        out.update({f'gt_{k}': v for k, v in dup_ratio(Pg).items()})
        out.update({f'gt_inter_{k}': v for k, v in interpenetration(vfg).items()})
    return out


G = {}


def init(run, gt, fast, with_gt, mesher):
    G.update(run=run, gt=gt, fast=fast, with_gt=with_gt, mesher=mesher)


def work(i):
    f = f"{G['run']}/JsonResults/{i:03d}.json"
    pred = load_cards(f) if os.path.isfile(f) else []
    return i, evaluate_sample(pred, G['gt'].cards(i), G['fast'], G['with_gt'], G['mesher'])


def run_eval(run, idxs, gt=None, tdir=None, workers=12, fast=False, with_gt=False, mesher=None, pc=False):
    """Per-sample metrics {idx: {...}} (in idxs order) of a run directory against the reference cards gt (default
    GTCards()); pc: add the point metrics where RUN/PointClouds exist, against tdir (default data/test_pc_orig, reference
    point clouds of the original meshes; not part of the test data of scripts/test.py). fast: skip interpenetration and
    card-level EMD. HAIRDECK_NO_EMD=1 skips every EMD (NaN)."""
    gt = GTCards() if gt is None else gt
    tdir = tdir or paths.data('test_pc_orig')
    with worker_pool(workers, init, (run, gt, fast, with_gt, mesher)) as pool:
        res = dict(pool.imap_unordered(work, idxs))
    per = {i: res[i] for i in idxs}
    if pc:
        for i in idxs:
            p = f'{run}/PointClouds/{i:03d}.ply'
            if os.path.isfile(p):
                per[i].update(point_metrics(read_ply_points(p), read_ply_points(f'{tdir}/{i:03d}.ply'), seed=i))
    return per


def summarize(per):
    keys = sorted({k for o in per.values() for k in o})
    return dict(mean={k: float(np.mean([per[i][k] for i in per if k in per[i]])) for k in keys},
                n_valid=sum(1 for i in per if per[i]['n_pred'] > 0), per_sample=per)
