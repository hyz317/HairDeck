"""Inputs of the Blender step of the VRM export (no Blender): the generated cards and chains of a saved result, from the
head frame into the body's dataset frame (p = (p_head - b) / k, width and thickness / k: the inverse of the head-frame
transform that built the head-frame training data), then into the Blender world of the imported body (to_blender, with
the body's fit), as one hair mesh with per-card UVs and the geometric skinning of hairdeck.spring.sim: a card follows
the chain nearest to the lower 60% of its centreline if that mean distance is below ASSIGN face widths, else it stays
with the head bone (sim.skin on the 16-point centreline, the assignment of the chain-motion evaluation); its mesh
vertices are then bound with sim.bind against that chain: chain weight w ramping from 0 at the chain root to 1 over
RAMP face widths of arc length (1 - w to the head bone), split between the nearest segment (alpha) and its neighbour
segment (1 - alpha) when BLEND. A card whose vertices fail the lower-60% test of its chain stays rigid. Chains that bind
no card are dropped. Without any chain (none predicted, all degenerate, or none binding a card) every card stays with
the head bone, with the same mesh, UVs and mapping.

prepare(out_dir, seed, body, colour, run, prep_dir, log) writes
  RUN/JsonResults/<cards>, RUN/chains.json       the cards and chains in the dataset frame (CHARM-format json)
  PREP_DIR/<prep>.npz                            V, F, UV, vcard, vchain, vseg, vw, vseg2, valpha, chain_ptr, chain_J,
                                                  chain_J_ds, stiffness, W, fit_s, fit_t, fit_sx, fit_sz, head_joint,
                                                  ASSIGN, RAMP, BLEND, colour (read by blender/vrm_export.py,
                                                  blender/vrm_worker.py and hairdeck.vrm.check)
(<cards>, <prep>: hairdeck.vrm.body.file_name of the body)
  PREP_DIR/<prep>.json                           summary (cards on chains, chains dropped, ...)
"""
import json
import os

import numpy as np

from ..cards.mesh import card_mesh_vf, card_uv
from ..spring.sim import ASSIGN, BLEND, RAMP, bind, centreline, clean, skin
from .body import file_name


def to_blender(p, fit):
    """Dataset frame -> Blender world of the body (inverse of the fit's Blender -> dataset similarity)."""
    a = (np.asarray(p, np.float64) - np.array(fit['t'])) / fit['s']
    return np.stack([fit['sx'] * a[..., 0], fit['sz'] * a[..., 2], a[..., 1]], -1)


def head_to_body(cards, chains, k, b):
    """Cards ([{'shape', 'seq'}], head frame) and chains ([{'joints', 'stiffness', 'n_joints'}]) -> the body's dataset
    frame: p = (p_head - b) / k, width and thickness / k. Cards that are not an (n >= 3, 8) finite array are left out."""
    out = []
    for c in cards:
        q = np.asarray(c['seq'], np.float64)
        if q.ndim != 2 or len(q) < 3 or not np.isfinite(q).all():
            continue
        q[:, :3] = (q[:, :3] - b) / k
        q[:, 3:5] /= k
        out.append({'shape': c['shape'], 'seq': q})
    pred = [dict(joints=(np.asarray(c['joints'], np.float64) - b) / k, stiffness=c['stiffness'], n_joints=c['n_joints'])
            for c in chains]
    return out, pred


def card_list(cards):
    """[{'shape', 'seq'}] -> [(seq float64, shape id)] with the rule of hairdeck.eval.cards.load_cards (>= 3 finite
    points)."""
    out = []
    for c in cards:
        s = np.asarray(c['seq'], dtype=np.float64)
        if s.ndim == 2 and len(s) >= 3 and np.isfinite(s).all():
            out.append((s, 1 if c.get('shape') == 'triangle' else 0))
    return out


def prep(h, key, cards, pred, fit):
    """Mesh, UVs and skinning with chains. cards: [(seq, shape id)] and pred: [{'joints', 'stiffness'}], both in the
    dataset frame. Returns (arrays, summary); raises ValueError when no chain binds a card."""
    W = fit['w'] / fit['s']  # face width in meters (Blender units)
    chains_ds = [clean(c['joints']) for c in pred]
    stiff = [float(c['stiffness']) for c in pred]
    keep = [k for k, J in enumerate(chains_ds) if len(J) >= 2]
    chains_ds, stiff = [chains_ds[k] for k in keep], [stiff[k] for k in keep]
    chains = [dict(J=to_blender(J, fit)) for J in chains_ds]
    Vs, Fs, vcard, vch, vsg, vw, vsg2, val, UVs = [], [], [], [], [], [], [], [], []
    off, mismatch = 0, 0
    for ci, (seq, sh) in enumerate(cards):
        v, f = card_mesh_vf(seq, sh)
        vb = to_blender(v, fit)
        cl = to_blender(centreline(seq[:, :3]), fit)  # 16 arc-length samples, root first
        c = int(skin([cl], chains, W)[0][0])  # card -> chain (-1 = rigid)
        n = len(vb)
        k, w, k2, al = np.zeros(n, int), np.zeros(n), np.zeros(n, int), np.ones(n)
        if c >= 0:
            b = bind([vb], [chains[c]], W)  # mesh vertices (root-first rows) against the assigned chain only
            if (b[0] < 0).any():  # the vertex-based lower-60% test disagrees with the centreline: keep the card rigid
                mismatch += 1
                c = -1
            else:
                k, w = b[1], b[2]
                if len(b) == 5:
                    k2, al = b[3], b[4]
                else:
                    k2 = k.copy()
        Vs.append(vb)
        UVs.append(card_uv(seq))  # per-vertex UVs in card_mesh_vf's vertex order
        Fs.append(f + off)
        off += n
        vcard.append(np.full(n, ci))
        vch.append(np.full(n, c))
        vsg.append(k)
        vw.append(w)
        vsg2.append(k2)
        val.append(al)
    V, F = np.concatenate(Vs), np.concatenate(Fs)
    vch, vsg, vw, vsg2, val = map(np.concatenate, (vch, vsg, vw, vsg2, val))
    vcard = np.concatenate(vcard)
    card_ch = np.array([vch[vcard == ci][0] for ci in range(len(cards))])
    used = sorted(set(card_ch[card_ch >= 0].tolist()))  # drop chains that bind no card
    remap = np.full(len(chains) + 1, -1)
    remap[used] = np.arange(len(used))
    vch = remap[vch]  # -1 stays -1 (index -1 -> last entry, which is -1)
    n_pred = len(pred)
    chains, chains_ds, stiff = [chains[c] for c in used], [chains_ds[c] for c in used], [stiff[c] for c in used]
    seg_len = np.concatenate([np.linalg.norm(np.diff(c['J'], axis=0), axis=1) for c in chains])
    out = dict(V=V, F=F, UV=np.concatenate(UVs), vcard=vcard, vchain=vch, vseg=vsg, vw=vw, vseg2=vsg2, valpha=val,
               chain_ptr=np.cumsum([0] + [len(c['J']) for c in chains]), chain_J=np.concatenate([c['J'] for c in chains]),
               chain_J_ds=np.concatenate(chains_ds), stiffness=np.array(stiff), W=W, fit_s=fit['s'],
               fit_t=np.array(fit['t']), fit_sx=fit['sx'], fit_sz=fit['sz'],
               head_joint=to_blender(np.array(fit['head_joint']), fit), ASSIGN=ASSIGN, RAMP=RAMP, BLEND=BLEND)
    blended = (vch >= 0) & (vsg2 != vsg) & (val < 1)
    info = dict(h=h, key=key, n_cards=len(cards), n_cards_on_chain=int((card_ch >= 0).sum() - mismatch),
                n_cards_bind_mismatch=mismatch, n_verts=len(V), n_faces=len(F), frac_verts_on_chain=float((vch >= 0).mean()),
                frac_verts_w_gt_half=float((vw > 0.5).mean()), frac_verts_two_segments=float(blended.mean()),
                n_chains_pred=n_pred, n_chains_degenerate=n_pred - len(keep), n_chains_dropped_no_cards=len(keep) - len(used),
                n_chains=len(chains), n_bones=int(sum(len(c['J']) - 1 for c in chains)),
                min_seg_mm=float(seg_len.min() * 1000), W_m=W, ASSIGN=ASSIGN, RAMP=RAMP, BLEND=BLEND)
    return out, info


def prep_no_chains(h, key, cards, fit):
    """Every card on the head bone: the mesh, UVs and mapping of prep() with empty chain arrays."""
    W = fit['w'] / fit['s']
    Vs, Fs, UVs, vcard, off = [], [], [], [], 0
    for ci, (seq, sh) in enumerate(cards):
        v, f = card_mesh_vf(seq, sh)
        Vs.append(to_blender(v, fit))
        Fs.append(f + off)
        UVs.append(card_uv(seq))
        vcard.append(np.full(len(v), ci))
        off += len(v)
    V = np.concatenate(Vs)
    n = len(V)
    z3 = np.zeros((0, 3))
    out = dict(V=V, F=np.concatenate(Fs), UV=np.concatenate(UVs), vcard=np.concatenate(vcard), vchain=np.full(n, -1),
               vseg=np.zeros(n, int), vw=np.zeros(n), vseg2=np.zeros(n, int), valpha=np.ones(n), chain_ptr=np.array([0]),
               chain_J=z3, chain_J_ds=z3, stiffness=np.zeros(0), W=W, fit_s=fit['s'], fit_t=np.array(fit['t']),
               fit_sx=fit['sx'], fit_sz=fit['sz'], head_joint=to_blender(np.array(fit['head_joint']), fit),
               ASSIGN=ASSIGN, RAMP=RAMP, BLEND=BLEND)
    info = dict(h=h, key=key, n_cards=len(cards), n_verts=n, n_faces=int(len(out['F'])), n_chains=0,
                note='no chains: every card on the head bone')
    return out, info


def prepare(out_dir, seed, body, colour, run, prep_dir, log=None):
    """One saved result (out_dir/cards_seed{s}.json, chains_seed{s}.json; head frame; without the chains file, e.g. from
    image_to_hair.py --no-chains, every card goes on the head bone) on a body (hairdeck.vrm.body):
    writes the files listed in the module docstring; colour: linear RGB of the hair material, stored with 6 decimals.
    log: a text file that gets the summary line ('PREP' + json, after 'PREP_FALLBACK' + the reason when chains were
    given but none binds a card). Returns (number of cards written, number of chains given)."""
    h, k, b = body['h'], body['k'], np.asarray(body['b'], np.float64)
    ch = f'{out_dir}/chains_seed{seed}.json'
    cards, pred = head_to_body(json.load(open(f'{out_dir}/cards_seed{seed}.json')),
                               json.load(open(ch)) if os.path.exists(ch) else [], k, b)
    os.makedirs(f'{run}/JsonResults', exist_ok=True)
    json.dump([{'shape': c['shape'], 'seq': c['seq'].tolist()} for c in cards],
              open(f"{run}/JsonResults/{file_name('cards', body)}", 'w'))
    json.dump({str(h): dict(pred=[dict(c, joints=c['joints'].tolist()) for c in pred], gt=[], n_cards=len(cards))},
              open(f'{run}/chains.json', 'w'))
    lines, out = [], None
    if [c for c in pred if len(clean(c['joints'])) >= 2]:
        try:
            out, info = prep(h, body['key'], card_list(cards), pred, body['fit'])
        except ValueError as e:  # every chain dropped (none binds a card): concatenation of nothing
            lines.append(f'PREP_FALLBACK {e}')
    if out is None:
        out, info = prep_no_chains(h, body['key'], card_list(cards), body['fit'])
    out['colour'] = np.array([float('%.6f' % x) for x in colour])
    os.makedirs(prep_dir, exist_ok=True)
    np.savez(f"{prep_dir}/{file_name('prep', body)}.npz", **out)
    json.dump(info, open(f"{prep_dir}/{file_name('prep', body)}.json", 'w'), indent=1)
    lines.append('PREP ' + json.dumps(info))
    if log:
        open(log, 'w').write(''.join(x + '\n' for x in lines))
    return len(cards), len(pred)
