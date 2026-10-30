"""Generated spring-bone chains vs the characters' own chains (RUN/chains.json of the chain test, scripts/test.py chains).

Per test hairstyle, in face-width units (head_params_test.json 'w'): chain count error; Chamfer distance between the
sets of chain path points (M samples per chain); chain recall / precision at TAU: a GT chain is recovered when its
points lie within TAU of some generated chain's path (mean point-to-path distance), and a generated chain is precise
when its points lie within TAU of some GT chain's path. evaluate_run: every hairstyle of a
run and the means.
"""
import json

import numpy as np
from scipy.spatial import cKDTree

from .. import paths

TAU = 0.10  # face widths (~2 cm)


def test_keys():
    """Hairstyle key of every test row (data/test_keys.txt, the test rows in order; head_params_test.json and
    spring/chains_test.jsonl are keyed by it)."""
    return open(paths.data('test_keys.txt')).read().splitlines()


def path_dist(a, b):
    """Mean distance from the points of path a to the polyline points of path b (dense resampling of b)."""
    b = np.asarray(b)
    t = np.linspace(0, 1, 50)
    seg = np.concatenate([b[k] + t[:, None] * (b[k + 1] - b[k]) for k in range(len(b) - 1)])
    return cKDTree(seg).query(np.asarray(a))[0].mean()


def metrics(pred, gt, W):
    P = [np.asarray(c['path']) / W for c in pred]
    G = [np.asarray(c['path']) / W for c in gt]
    out = dict(n_pred=len(P), n_gt=len(G))
    if not P or not G:
        out.update(cd=np.nan, recall=0.0 if G else np.nan, precision=0.0 if P else np.nan)
        return out
    pp, gg = np.concatenate(P), np.concatenate(G)
    out['cd'] = float(cKDTree(gg).query(pp)[0].mean() + cKDTree(pp).query(gg)[0].mean())
    D = np.array([[path_dist(g, p) for p in P] for g in G])  # GT points to generated paths
    out['recall'] = float((D.min(1) < TAU).mean())
    D = np.array([[path_dist(p, g) for g in G] for p in P])  # generated points to GT paths
    out['precision'] = float((D.min(1) < TAU).mean())
    return out


def aggregate(M):
    agg = {k: float(np.nanmean([m[k] for m in M])) for k in ('cd', 'recall', 'precision')}
    agg['count_abs_err'] = float(np.mean([abs(m['n_pred'] - m['n_gt']) for m in M]))
    agg['n_pred'], agg['n_gt'] = float(np.mean([m['n_pred'] for m in M])), float(np.mean([m['n_gt'] for m in M]))
    return agg


def evaluate_run(run):
    """metrics of every test hairstyle of RUN/chains.json: dict(mean (aggregate), per_sample)."""
    res = json.load(open(f'{run}/chains.json'))
    keys = test_keys()
    fits = json.load(open(paths.data('head_params_test.json')))
    per = {h: metrics(r['pred'], r['gt'], fits[keys[int(h)]]['w']) for h, r in res.items()}
    return dict(mean=aggregate(list(per.values())), per_sample=per)
