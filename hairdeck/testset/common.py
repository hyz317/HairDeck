"""The files of a test: the generated cards of a seed's run, its RUN/eval.json, and OUT/eval.json with the table of
every seed and their mean, printed at the end."""
import json
import os

import numpy as np

from ..cards.mesh import cards_to_json, cards_to_mesh

SEEDS = (0, 1, 2)  # the paper's rows are the mean over these sampling seeds


def seed_dir(out, seed):
    return f'{out}/seed{seed}'


def write_cards(out, idxs, hairs):
    """OUT/JsonResults/{idx:03d}.json (CHARM format) and .glb of the hairstyles generated for the test rows idxs; an
    empty generation (no card with 3 or more points) gets a 0-byte .glb. The point cloud of an earlier sample of the same
    row (OUT/PointClouds/{idx:03d}.ply) is removed, so the evaluation scans the new one. Returns the names of the empty
    ones."""
    os.makedirs(f'{out}/JsonResults', exist_ok=True)
    empty = []
    for i, cards in zip(idxs, hairs):
        json.dump(cards_to_json(cards), open(f'{out}/JsonResults/{i:03d}.json', 'w'))
        m = cards_to_mesh(cards)
        if m.geometry:
            m.export(f'{out}/JsonResults/{i:03d}.glb')
        else:
            open(f'{out}/JsonResults/{i:03d}.glb', 'wb').close()
            empty.append(f'{i:03d}')
        ply = f'{out}/PointClouds/{i:03d}.ply'
        if os.path.exists(ply):
            os.remove(ply)
    return empty


def save_eval(run, model, settings, table, parts):
    """RUN/eval.json: the model, its settings, the table values and every evaluation part (means and per-sample values,
    as hairdeck.eval returns them)."""
    json.dump(dict(model=model, settings=settings, table=table, **parts), open(f'{run}/eval.json', 'w'), indent=1)


def print_table(title, columns, rows):
    """columns: (name, decimals) pairs; rows: (label, {name: value, or None for a value that is not measured})."""
    cells = [[label] + ['--' if t.get(c) is None else f'{t[c]:.{d}f}' for c, d in columns] for label, t in rows]
    names = [''] + [c for c, _ in columns]
    widths = [max(len(r[j]) for r in [names] + cells) for j in range(len(names))]
    print(title)
    for r in [names] + cells:
        print('  ' + r[0].ljust(widths[0]) + '  ' + '  '.join(v.rjust(w) for v, w in zip(r[1:], widths[1:])))
    print(flush=True)


def summarize(out, model, seeds, tables):
    """OUT/eval.json from the seeds' OUT/seed<k>/eval.json: dict(model, seeds, per_seed (each seed's table and
    settings), mean (the mean of the seeds' values, None where a seed has none)); then the printed tables (tables:
    (title, columns) pairs) with one row per seed and the mean. Returns the mean."""
    ev = {s: json.load(open(f'{seed_dir(out, s)}/eval.json')) for s in seeds}
    names = list(ev[seeds[0]]['table'])
    mean = {c: None if any(ev[s]['table'].get(c) is None for s in seeds) else
            float(np.mean([ev[s]['table'][c] for s in seeds])) for c in names}
    json.dump(dict(model=model, seeds=list(seeds), mean=mean,
                   per_seed={f'seed{s}': dict(table=ev[s]['table'], settings=ev[s]['settings']) for s in seeds}),
              open(f'{out}/eval.json', 'w'), indent=1)
    rows = [(f'seed {s}', ev[s]['table']) for s in seeds] + ([(f'mean of {len(seeds)} seeds', mean)] if len(seeds) > 1 else [])
    print(f'{out}:')
    for title, columns in tables:
        print_table(title, columns, rows)
    return mean
