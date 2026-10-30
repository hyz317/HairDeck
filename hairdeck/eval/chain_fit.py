"""Do the chains fit the cards they were generated for? Reference-free check for chains on generated hair. Everything
in face widths W of the test character (head_params_test.json).

  on_hair    fraction of chain path points (8 per chain) within TAU of the cards (card mesh vertices)
  far        mean distance of chain path points to the cards
  moving     fraction of card points that follow a chain under the skinning rule of spring/sim.py
  amp        mean motion of card points over the four head motions, with the static pose subtracted
evaluate_run: every hairstyle of a run (RUN/chains.json) in worker processes, and the means.
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from .. import paths
from ..cards.mesh import card_mesh_vf
from ..spring import sim
from .cards import GTCards, load_cards as load_json_cards
from .chains import test_keys
from .pool import worker_pool

TAU = 0.10
G = {}


def load_cards(h):
    if G['cards'] == 'gt':
        if 'gt_cards' not in G:
            G['gt_cards'] = GTCards()
        return [(np.asarray(q, np.float64), sh) for q, sh in G['gt_cards'].cards(h) if len(q) >= 3]
    return load_json_cards(f"{G['cards']}/JsonResults/{h:03d}.json")


def work(h):
    key = G['keys'][h]
    fit = G['fits'][key]
    s = G['chains_rec'][key]['fit']['s']
    W = fit['w'] / s
    piv = np.asarray(fit['head_joint']) / s
    cards = load_cards(h)
    chains = [dict(J=sim.clean(np.asarray(c['joints']) / s), stiff=c['stiffness'], drag=0.4, grav=np.zeros(3))
              for c in G['res'][str(h)][G['which']]]
    chains = [c for c in chains if len(c['J']) >= 2]
    out = dict(n_chains=len(chains), n_cards=len(cards))
    if not cards or not chains:
        out.update(on_hair=np.nan, far=np.nan, moving=0.0, amp=0.0)
        return h, out
    V = np.concatenate([card_mesh_vf(q, sh)[0] for q, sh in cards]) / s
    P = np.concatenate([np.asarray(c['path']) for c in G['res'][str(h)][G['which']]]) / s
    d = cKDTree(V).query(P)[0] / W
    out['on_hair'] = float((d < TAU).mean())
    out['far'] = float(d.mean())
    pts = [sim.centreline(q[:, :3] / s) for q, _ in cards]
    Vc = np.concatenate(pts)
    bind = sim.bind(pts, chains, W)
    out['moving'] = float((bind[0] >= 0).mean())
    R0, T0 = np.repeat(np.eye(3)[None], sim.NF, 0), np.zeros((sim.NF, 3))
    static = sim.deform(Vc, bind, chains, sim.simulate(chains, R0, T0, piv), R0, T0, piv) - Vc
    amps = []
    for m in sim.MOTIONS:
        R, T = sim.motion(m)
        X = sim.deform(Vc, bind, chains, sim.simulate(chains, R, T, piv), R, T, piv)
        amps.append(np.linalg.norm(sim.local_disp(X, Vc, R, T, piv) - static, axis=-1).mean() / W)
    out['amp'] = float(np.mean(amps))
    return h, out


def init(run, cards, which):
    """cards: 'gt' (the reference cards) or a run directory whose JsonResults hold the cards (dataset frame)."""
    G['keys'] = test_keys()
    G['fits'] = json.load(open(paths.data('head_params_test.json')))
    G['chains_rec'] = {r['key']: r for r in map(json.loads, open(paths.data('spring', 'chains_test.jsonl'))) if r['status'] == 'ok'}
    G['res'] = json.load(open(f'{run}/chains.json'))
    G['cards'], G['which'] = cards, which


def evaluate_run(run, cards='gt', which='pred', workers=12):
    """work for every hairstyle of RUN/chains.json, the chains `which` ('pred' or 'gt') on `cards` ('gt' or a run
    directory whose JsonResults hold the cards, dataset frame): (name fit_<which>_on_<gt | run name>,
    dict(mean, per_sample))."""
    with open(f'{run}/chains.json') as f:
        hs = sorted(int(h) for h in json.load(f))
    with worker_pool(workers, init, (run, cards, which)) as p:
        res = dict(p.imap_unordered(work, hs))
    per = {str(h): res[h] for h in hs}
    keys = ('on_hair', 'far', 'moving', 'amp', 'n_chains', 'n_cards')
    mean = {k: float(np.nanmean([v[k] for v in per.values()])) for k in keys}
    name = f"fit_{which}_on_{'gt' if cards == 'gt' else os.path.basename(cards.rstrip('/'))}"
    return name, dict(mean=mean, per_sample=per)
