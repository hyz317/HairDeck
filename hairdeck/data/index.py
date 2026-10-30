"""The index of the CHARM dataset: its train / test ids resolved to hairstyle directories, with their hair.json templates
(CHARM's cards, 5 columns) parsed for a statistics report. A hairstyle enters the index (data/{split}_meta.json, in the
order of the id lists) only if its hair.json parses; the parsed cards are used for the report only. Read-only on the
dataset. Driver: scripts/data.py (step cards).

Dataset layout (DATASET_DIR): vroid-hair, vroid-hair-new, vroid-hair-tempconb;
  train ids: vroid-hair-new/valid_path_hairs_train_suc.json (directories looked up in vroid-hair-new first, then vroid-hair)
  test ids : vroid-hair/final_test_ls.json (entries 'vroid-hair/...', 'vroid-hair-new/...', 'vroid-hair-tempconb/...')
"""
import json
import os
from multiprocessing import Pool

import numpy as np

SHAPES = {'square': 0, 'triangle': 1}
ROOTS = {}


def init(roots):
    ROOTS.update(roots)


def dataset_roots(dataset_dir):
    return dict(new=os.path.join(dataset_dir, 'vroid-hair-new'), old=os.path.join(dataset_dir, 'vroid-hair'),
                tc=os.path.join(dataset_dir, 'vroid-hair-tempconb'))


def resolve_train(rel):
    for root, tag in ((ROOTS['new'], 'new'), (ROOTS['old'], 'old')):
        d = os.path.join(root, rel)
        if os.path.isfile(os.path.join(d, 'hair.json')):
            return d, tag
    return None, None


def resolve_test(entry):
    prefix, rel = entry.split('/', 1)
    root = {'vroid-hair': ROOTS['old'], 'vroid-hair-new': ROOTS['new'], 'vroid-hair-tempconb': ROOTS['tc']}[prefix]
    d = os.path.join(root, rel)
    return (d, prefix) if os.path.isfile(os.path.join(d, 'hair.json')) else (None, prefix)


def orient_card(seq, anchor):
    """Dataset convention: root is the endpoint closer to anchor = (0, p95(max_y), 0)."""
    if np.linalg.norm(seq[0, :3] - anchor) > np.linalg.norm(seq[-1, :3] - anchor):
        seq = seq[::-1].copy()
    return seq


def parse_one(d):
    out = {'dir': d, 'status': 'ok'}
    try:
        with open(os.path.join(d, 'hair.json')) as f:
            model = json.load(f)
    except Exception as e:  # noqa
        out['status'] = f'json_error:{e}'
        return out, None
    if len(model) == 0:
        out['status'] = 'empty'
        return out, None
    seqs, shapes = [], []
    for block in model:
        s = np.asarray(block['seq'], dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 5 or len(s) == 0:
            out['status'] = 'bad_card'
            return out, None
        seqs.append(s)
        shapes.append(SHAPES.get(block['shape'], 255))
    allp = np.concatenate(seqs)
    if not np.isfinite(allp).all():
        out['status'] = 'nan'
        return out, None
    max_y = np.percentile([s[:, 1].max() for s in seqs], 95)
    anchor = np.array([0.0, max_y, 0.0])
    seqs = [orient_card(s, anchor) for s in seqs]
    out['n_cards'] = len(seqs)
    out['n_points'] = int(sum(len(s) for s in seqs))
    out['xyz_min'] = allp[:, :3].min(0).tolist()
    out['xyz_max'] = allp[:, :3].max(0).tolist()
    out['w_max'] = float(allp[:, 3].max())
    out['t_max'] = float(allp[:, 4].max())
    out['w_min'] = float(allp[:, 3].min())
    out['t_min'] = float(allp[:, 4].min())
    out['has_reform_ply'] = os.path.isfile(os.path.join(d, 'hair_reform.ply'))
    return out, (seqs, shapes)


def pack(results):
    pts, card_len, card_shape, hair_ncard = [], [], [], []
    for _, data in results:
        seqs, shapes = data
        hair_ncard.append(len(seqs))
        for s, sh in zip(seqs, shapes):
            pts.append(s.astype(np.float32))
            card_len.append(len(s))
            card_shape.append(sh)
    card_ptr = np.concatenate([[0], np.cumsum(card_len)]).astype(np.int64)
    hair_ptr = np.concatenate([[0], np.cumsum(hair_ncard)]).astype(np.int64)
    return dict(points=np.concatenate(pts), card_ptr=card_ptr, card_shape=np.asarray(card_shape, np.uint8), hair_ptr=hair_ptr)


def pct(a, qs):
    return [float(np.percentile(a, q)) for q in qs]


def stats(meta, packed, name):
    nc = np.array([m['n_cards'] for m in meta])
    npt = np.array([m['n_points'] for m in meta])
    cl = np.diff(packed['card_ptr'])
    lines = [f'== {name}: {len(meta)} hairstyles, {len(cl)} cards, {len(packed["points"])} points']
    lines.append(f'cards/hairstyle p0 p10 p50 p90 p95 p99 max {pct(nc, [0, 10, 50, 90, 95, 99, 100])}')
    lines.append(f'points/card p0 p10 p50 p90 p95 p99 max {pct(cl, [0, 10, 50, 90, 95, 99, 100])}')
    lines.append(f'points/hairstyle p50 p90 p99 max {pct(npt, [50, 90, 99, 100])}; share>8192 {np.mean(npt > 8192):.4f}')
    lines.append(f'share hairstyles with >192 cards {np.mean(nc > 192):.4f}; cards with <3 pts {np.mean(cl < 3):.5f}; <5 pts {np.mean(cl < 5):.5f}; >96 pts {np.mean(cl > 96):.5f}')
    sh = packed['card_shape']
    lines.append(f'shape: square {np.mean(sh == 0):.4f} triangle {np.mean(sh == 1):.4f} unknown {np.mean(sh == 255):.4f}')
    p = packed['points']
    lines.append(f'xyz min {p[:, :3].min(0).round(4).tolist()} max {p[:, :3].max(0).round(4).tolist()}; share |xyz|>0.501 {np.mean(np.abs(p[:, :3]) > 0.501):.6f}')
    lines.append(f'width p1 p50 p99 max {np.round(pct(p[:, 3], [1, 50, 99, 100]), 5).tolist()}; thickness p1 p50 p99 max {np.round(pct(p[:, 4], [1, 50, 99, 100]), 5).tolist()}; share w>0.1 {np.mean(p[:, 3] > 0.1):.6f} t>0.1 {np.mean(p[:, 4] > 0.1):.6f}')
    seg = np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1)
    same_card = np.ones(len(p) - 1, bool)
    same_card[packed['card_ptr'][1:-1] - 1] = False
    seg = seg[same_card]
    lines.append(f'segment length p1 p50 p99 {np.round(pct(seg, [1, 50, 99]), 5).tolist()}')
    lines.append(f'has hair_reform.ply {np.mean([m["has_reform_ply"] for m in meta]):.4f}')
    return '\n'.join(lines)


def build(dataset_dir, splits=('train', 'test'), workers=24):
    """{split: meta (the data/{split}_meta.json entries)} and the report lines."""
    roots = dataset_roots(dataset_dir)
    init(roots)
    report = []
    raw = json.load(open(f"{roots['new']}/valid_path_hairs_train_suc.json"))
    rels = list(dict.fromkeys(r.replace('../vroid-hair/', '') for r in raw))
    test_raw = json.load(open(f"{roots['old']}/final_test_ls.json"))
    items = {}
    if 'train' in splits:
        report.append(f'train_suc raw {len(raw)} dedup {len(rels)}')
        with Pool(workers, initializer=init, initargs=(roots,)) as pool:
            res = pool.map(resolve_train, rels, chunksize=64)
        both = sum(os.path.isfile(os.path.join(roots['new'], r, 'hair.json')) and os.path.isfile(os.path.join(roots['old'], r, 'hair.json')) for r in rels)
        tags = [t for _, t in res]
        report.append(f'resolved: new {tags.count("new")} old {tags.count("old")} missing {tags.count(None)}; present in both dirs {both} (rule: prefer vroid-hair-new)')
        items['train'] = [(r, d) for r, (d, _) in zip(rels, res) if d is not None]
    if 'test' in splits:
        tres = [resolve_test(e) for e in test_raw]
        report.append(f'test {len(test_raw)}: found {sum(d is not None for d, _ in tres)}; by prefix ' + str({p: sum(1 for _, q in tres if q == p) for p in dict.fromkeys(q for _, q in tres)}))
        items['test'] = list(zip(test_raw, [d for d, _ in tres]))
    leak = set(e.split('/', 1)[1] for e in test_raw) & set(rels)
    report.append(f'test ids overlapping train_suc ids: {len(leak)} {sorted(leak)[:5]}')
    metas = {}
    for split in ('train', 'test'):
        if split not in items:
            continue
        with Pool(workers) as pool:
            results = pool.map(parse_one, [d for _, d in items[split]], chunksize=16)
        status = {}
        for m, _ in results:
            status[m['status']] = status.get(m['status'], 0) + 1
        report.append(f'{split} parse status {status}')
        keep = [(k, r) for k, r in zip(items[split], results) if r[1] is not None]
        metas[split] = [dict(r[0], key=k[0]) for k, r in keep]
        report.append(stats(metas[split], pack([r for _, r in keep]), split))
    return metas, report
