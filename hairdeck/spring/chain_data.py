"""Chain-set training data for the chain DiT.

Per hairstyle (train_cards_meta / test_meta order, so it lines up with the card data): the hair chains extracted from
the VRM (data/spring/chains_{split}.jsonl, hairdeck.spring.vrm_spring; dataset frame), mapped into the head
frame (p_head = k * p + b from data/head_frame_{split}.npz) and resampled by arc length to M points; token = [M x 3
positions, joint count, stiffness] (raw units; the chain DiT normalises with the stats). Hairstyles with no hair chain,
with more than NMAX chains or without a head fit are left out (valid = False).
Driver: scripts/data.py (step chains).
"""
import json
import os

import numpy as np

from .. import paths

M, NMAX = 8, 48


def resample(p, m=M):
    p = np.asarray(p, np.float64)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(p[:1], m, 0)
    t = np.linspace(0, d[-1], m)
    return np.stack([np.interp(t, d, p[:, k]) for k in range(3)], 1)


def build_arrays(split):
    """-> tok (H, NMAX, 3M + 2) float32, count (H,), valid (H,), number of hairstyles dropped for > NMAX chains."""
    meta = json.load(open(paths.data(f"{'train_cards_meta' if split == 'train' else 'test_meta'}.json")))
    hz = np.load(paths.data(f'head_frame_{split}.npz'))
    key2i = {e['key']: i for i, e in enumerate(meta)}
    H = len(meta)
    tok = np.zeros((H, NMAX, M * 3 + 2), np.float32)
    cnt = np.zeros(H, np.int32)
    valid = np.zeros(H, bool)
    n_dropped_many = 0
    for line in open(paths.data('spring', f'chains_{split}.jsonl')):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a line still being written by the extractor
            continue
        if r['status'] != 'ok' or r['key'] not in key2i or not r['chains']:
            continue
        i = key2i[r['key']]
        if not (i < len(hz['valid']) and hz['valid'][i]):
            continue
        chains = r['chains']
        if len(chains) > NMAX:
            n_dropped_many += 1
            continue
        for k, c in enumerate(chains):
            pos = float(hz['k'][i]) * np.asarray(c['pos'], np.float64) + hz['b'][i]
            tok[i, k, :M * 3] = resample(pos).reshape(-1)
            tok[i, k, M * 3] = len(c['pos'])
            tok[i, k, M * 3 + 1] = c['stiffness']
        cnt[i] = len(chains)
        valid[i] = True
    return tok, cnt, valid, n_dropped_many


def build(split):
    """build_arrays + write data/spring/chains_{split}_head.npz; returns (tok, count, valid)."""
    tok, cnt, valid, n_dropped_many = build_arrays(split)
    H = len(cnt)
    fn = paths.data('spring', f'chains_{split}_head.npz')
    with open(fn + '.partial', 'wb') as f:  # a stopped run leaves no partial file
        np.savez(f, tok=tok, count=cnt, valid=valid)
    os.replace(fn + '.partial', fn)
    print(split, 'head frame', 'hairstyles', H, 'with chains', int(valid.sum()), 'dropped (> %d chains)' % NMAX, n_dropped_many,
          'chains per valid hairstyle p50 %.0f p90 %.0f' % tuple(np.percentile(cnt[valid], [50, 90])))
    return tok, cnt, valid


def stats_of(tok, cnt, valid):
    X = np.concatenate([tok[i, :cnt[i]] for i in np.nonzero(valid)[0]])
    return dict(mu=X.mean(0).tolist(), sd=np.maximum(X.std(0), 1e-4).tolist(), M=M, NMAX=NMAX, n_chains=int(len(X))), X
