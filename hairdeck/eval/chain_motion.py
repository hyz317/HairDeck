"""Does a generated chain set make the hair move like the character's own spring bones?

Per test hairstyle with hair chains: the GT cards are skinned to each rig with one geometric procedure (spring/sim.py,
no learned weights), the rig is simulated under scripted head motions, and the motion of points on the cards is
compared with the motion under the character's own VRM chains.
  rigs: gt (the VRM chains with their stiffness / drag / gravity; the reference), gt_tok (GT chains after the chain
        token round trip: 8-point path, uniform joints), pred (chain DiT sample, drag 0.4, gravity 0), none (no spring
        bones: hair rigid with the head)
  metric: per card point (16 per card), displacement relative to the rigidly moving head, minus the same rig's
        displacement with the head at rest; error = mean over points and frames of |displacement - displacement under
        gt| in face widths ('amp' = mean |displacement| under gt = error of none); amp_<rig> = the rig's own mean
        |displacement|; corr_<rig> = correlation over card points of the per-point mean |displacement| with gt's
evaluate_run: every hairstyle of a run (RUN/chains.json) in worker processes, and the means (summarize).
"""
import json

import numpy as np

from .. import paths
from ..spring import sim
from .cards import GTCards
from .chains import test_keys
from .pool import worker_pool

RIGS = ('gt_tok', 'pred', 'none')  # compared with gt
G = {}


def rigs_for(h):
    """Chains (meters) of every rig for test hairstyle h, plus the cards and the per-character constants."""
    if 'gt' not in G:
        G['gt'] = GTCards()
    key = G['keys'][h]
    fit = G['fits'][key]
    rec = G['chains'][key]
    s = rec['fit']['s']  # dataset units per meter
    W = fit['w'] / s
    piv = np.asarray(fit['head_joint']) / s
    cards = [(np.asarray(q, np.float64), sh) for q, sh in G['gt'].cards(h) if len(q) >= 3]
    gdown = np.array([0.0, -1.0, 0.0])
    rigs = dict(gt=[dict(J=sim.clean(np.asarray(c['pos']) / s), stiff=c['stiffness'], drag=c['drag'], grav=gdown * c['gravity'])
                    for c in rec['chains']])
    r = G['res'][str(h)]
    rigs['gt_tok'] = [dict(J=sim.clean(np.asarray(c['joints']) / s), stiff=c['stiffness'], drag=0.4, grav=np.zeros(3)) for c in r['gt']]
    rigs['pred'] = [dict(J=sim.clean(np.asarray(c['joints']) / s), stiff=c['stiffness'], drag=0.4, grav=np.zeros(3)) for c in r['pred']]
    rigs['none'] = []
    for k in rigs:
        rigs[k] = [c for c in rigs[k] if len(c['J']) >= 2]
    return rigs, cards, s, W, piv


def work(h):
    rigs, cards, s, W, piv = rigs_for(h)
    pts = [sim.centreline(q[:, :3] / s) for q, _ in cards]
    V = np.concatenate(pts)
    binds = {k: sim.bind(pts, v, W) for k, v in rigs.items()}
    R0, T0 = np.repeat(np.eye(3)[None], sim.NF, 0), np.zeros((sim.NF, 3))
    static = {}  # each rig with the head at rest (gravity sag etc.), subtracted so only motion-induced displacement counts
    for rname, chains in rigs.items():
        static[rname] = sim.deform(V, binds[rname], chains, sim.simulate(chains, R0, T0, piv), R0, T0, piv) - V
    out = {}
    for mname in sim.MOTIONS:
        R, T = sim.motion(mname)
        U = {}
        for rname, chains in rigs.items():
            X = sim.deform(V, binds[rname], chains, sim.simulate(chains, R, T, piv), R, T, piv)
            U[rname] = sim.local_disp(X, V, R, T, piv) - static[rname]
        mag = {r: np.linalg.norm(U[r], axis=-1) for r in U}  # (NF, points)
        e_gt = mag['gt'].mean(0)
        o = dict(amp=float(mag['gt'].mean() / W))
        for r in RIGS:
            o[r] = float(np.linalg.norm(U[r] - U['gt'], axis=-1).mean() / W)
            o['amp_' + r] = float(mag[r].mean() / W)
            e = mag[r].mean(0)
            o['corr_' + r] = float(np.corrcoef(e, e_gt)[0, 1]) if e.std() > 1e-12 and e_gt.std() > 1e-12 else 0.0
        out[mname] = o
    out['n_chains'] = {k: len(v) for k, v in rigs.items()}
    out['frac_moving'] = {k: float((binds[k][0] >= 0).mean()) for k in rigs}
    return h, out


def init(run):
    G['keys'] = test_keys()
    G['fits'] = json.load(open(paths.data('head_params_test.json')))
    G['chains'] = {r['key']: r for r in map(json.loads, open(paths.data('spring', 'chains_test.jsonl'))) if r['status'] == 'ok'}
    G['res'] = json.load(open(f'{run}/chains.json'))


def summarize(per):
    keys = ['amp'] + list(RIGS) + ['amp_' + r for r in RIGS] + ['corr_' + r for r in RIGS]
    mean = {m: {k: float(np.mean([per[h][m][k] for h in per])) for k in keys} for m in sim.MOTIONS}
    mean['all'] = {k: float(np.mean([mean[m][k] for m in sim.MOTIONS])) for k in keys}
    mean['frac_moving'] = {k: float(np.mean([per[h]['frac_moving'][k] for h in per])) for k in ('gt',) + RIGS}
    mean['n_chains'] = {k: float(np.mean([per[h]['n_chains'][k] for h in per])) for k in ('gt',) + RIGS}
    return mean


def evaluate_run(run, workers=12, limit=0):
    """work for every hairstyle of RUN/chains.json (the first `limit` hairstyles if limit): dict(mean (summarize),
    per_sample)."""
    with open(f'{run}/chains.json') as f:
        hs = sorted(int(h) for h in json.load(f))
    hs = hs[:limit] if limit else hs
    with worker_pool(workers, init, (run,)) as p:
        res = dict(p.imap_unordered(work, hs))
    per = {h: res[h] for h in hs}
    return dict(mean=summarize(per), per_sample={str(h): v for h, v in per.items()})
