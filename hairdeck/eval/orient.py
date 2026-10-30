"""Orientation error along the cards (Orient. of Table 1) of a generated hairstyle against the original meshes.

Every generated card is matched to the ground-truth card with the smallest card CD, as hairdeck.eval.cards.card_level
matches them (256 surface samples per card, seeds 0 / 100000; ground truth: the cards of data/test_cards_orig with at
least 3 points). For each generated card, the angle between the width direction of its mesh and the true width
direction of the matched ground-truth card, averaged over K = 32 uniform positions of normalised arc length (root and
tip included), is its section mean. Width directions along a card, known at its control points:
  generated card: the Bishop frame from its stored root normal (columns 5:8), the direction hairdeck.cards.mesh meshes it
                  with (8-column cards only);
  ground truth:   the width direction of each cross-section ring of the original card mesh (data/test_cards_orig_dirs.npz,
                  one per control point of the cards of data/test_cards_orig; NaN where a ring has none: single-vertex
                  tips).
Both fields are made sign-consistent along the card (the width direction is unsigned), interpolated componentwise and
linearly in the normalised arc length (u of a control point = its arc length along the control polyline / the total
length), renormalised and held constant beyond the first / last known point. Angle: degrees(arccos |<a, b>|), 0 to 90.
Per sample: the mean of the section means of its cards (ground-truth cards without a known direction are left out);
a run: the mean over its samples. evaluate_run: every sample of a run in worker processes (RUN/JsonResults).
"""
import os

import numpy as np

from .. import paths
from ..cards.codec import arclen
from ..cards.frames import bishop, tangents
from .cards import GTCards, card_points, load_cards, pair_cd
from .pool import worker_pool

K = 32


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def norm_u(xyz):
    """normalised arc length of the control points"""
    s = arclen(np.asarray(xyz, np.float64)[:, :3])
    return s / s[-1]


def mesh_dirs(seq):
    """width direction of the card mesh at each control point (8-column cards: Bishop frame from the root normal)"""
    seq = np.asarray(seq, np.float64)
    if seq.shape[1] < 8:
        raise ValueError('the orientation metric needs cards with their root normals (8 columns)')
    return bishop(seq[0, 5:8], tangents(seq[:, :3]))


def field_at(u_pts, dirs, u_q):
    """an unsigned direction field known at u_pts (rows with NaN are unknown) at u_q; None if nothing is known"""
    dirs = np.asarray(dirs, np.float64)
    ok = np.isfinite(dirs).all(1)
    if not ok.any():
        return None
    u, d = np.asarray(u_pts)[ok], unit(dirs[ok]).copy()
    for i in range(1, len(d)):
        if np.dot(d[i], d[i - 1]) < 0:
            d[i] = -d[i]
    return unit(np.stack([np.interp(u_q, u, d[:, c]) for c in range(3)], 1))


def section_angles(pred_seq, gt_xyz, gt_dirs, k=K):
    """(k,) angles between the generated card's mesh width direction and the ground-truth width direction at the same
    normalised arc lengths, or None if the ground-truth card has no known direction"""
    uq = np.linspace(0.0, 1.0, k)
    b = field_at(norm_u(gt_xyz), gt_dirs, uq)
    if b is None:
        return None
    a = field_at(norm_u(pred_seq), mesh_dirs(pred_seq), uq)
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


class GTDirs:
    """The ground-truth cards of GTCards (data/test_cards_orig, cards with >= 3 points) with the true width direction at
    each control point (data/test_cards_orig_dirs.npz: dirs (P, 3) float64; row_ptr: the cards of test row i are
    card_ptr indices row_ptr[i]:row_ptr[i + 1]; card_ptr: the points of card j are dirs[card_ptr[j]:card_ptr[j + 1]])."""

    def __init__(self, path=None, cards=None):
        z = np.load(path or paths.data('test_cards_orig_dirs.npz'))
        self.dirs, self.card_ptr, self.row_ptr = z['dirs'], z['card_ptr'], z['row_ptr']
        self.gt = cards or GTCards()

    def cards(self, i):
        """[(seq, shape id, dirs (n, 3))] of test row i"""
        cards = self.gt.cards(i)
        a, b = self.row_ptr[i], self.row_ptr[i + 1]
        if len(cards) != b - a:
            raise ValueError(f'test row {i}: {len(cards)} cards, {b - a} in the direction file')
        out = []
        for (s, sh), j in zip(cards, range(a, b)):
            d = self.dirs[self.card_ptr[j]:self.card_ptr[j + 1]]
            if len(d) != len(s):
                raise ValueError(f'test row {i}: card {j - a} has {len(s)} points, {len(d)} directions')
            out.append((s, sh, d))
        return out


def evaluate_sample(pred, gt, k=K):
    """pred: [(seq, shape id)], gt: [(seq, shape id, dirs)] -> dict(n_pred, n_gt, n_sec, sec_mean)"""
    out = dict(n_pred=len(pred), n_gt=len(gt))
    if not pred:
        return out
    Pp, _ = card_points(pred, seed=0)
    Pg, _ = card_points([(s, sh) for s, sh, _ in gt], seed=100000)
    m = pair_cd(Pp, Pg)[0].argmin(1)
    sec = []
    for p in range(len(pred)):
        s, _, d = gt[m[p]]
        a = section_angles(pred[p][0], s[:, :3], d, k)
        sec.append(np.nan if a is None else float(a.mean()))
    sec = np.array(sec)
    sec = sec[np.isfinite(sec)]
    out.update(n_sec=len(sec), sec_mean=float(sec.mean()) if len(sec) else float('nan'))
    return out


G = {}


def init(run):
    G['run'], G['gt'] = run, GTDirs()


def work(i):
    f = f"{G['run']}/JsonResults/{i:03d}.json"
    return i, evaluate_sample(load_cards(f) if os.path.isfile(f) else [], G['gt'].cards(i))


def evaluate_run(run, idxs, workers=12):
    """evaluate_sample for the test rows idxs: dict(mean (sec_mean over the samples with a value, n of them; an empty
    generation has none), per_sample (in idxs order))."""
    with worker_pool(workers, init, (run,)) as p:
        res = dict(p.imap_unordered(work, idxs))
    per = {i: res[i] for i in idxs}
    v = [r['sec_mean'] for r in per.values() if np.isfinite(r.get('sec_mean', np.nan))]
    return dict(mean=dict(sec_mean=float(np.mean(v)) if v else float('nan'), n=len(v)), per_sample=per)
