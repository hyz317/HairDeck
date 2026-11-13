"""Real pipeline cache benchmark; writes comparable outputs and timings on the target GPU."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from hairdeck import paths
from hairdeck.infer.image import Pipeline
p = Pipeline()
image = paths.data('demo_examples', 'ex01.png')
root = Path(paths.results('cache-benchmark'))
rows = []
for label, enabled, seed in [('warmup', False, 0), ('original', False, 1), ('fill', True, 0), ('hit', True, 1)]:
    p.cache_enabled = enabled
    r = p.run(image, seed=seed, chains=True, out_dir=str(root / label))
    row = dict(label=label, ok=r['ok'], timings=r['timings'], cache=r['cache'], notes=r['notes'])
    rows.append(row)
    print(json.dumps(row), flush=True)
a, b = root / 'original', root / 'hit'
for name in ['tokens_seed1.npy', 'cards_seed1.json', 'chains_seed1.json']:
    same = np.array_equal(np.load(a/name), np.load(b/name)) if name.endswith('.npy') else (a/name).read_bytes() == (b/name).read_bytes()
    print(name, 'identical', same, flush=True)
    if not same:
        raise RuntimeError('Cache changed output: ' + name)
(root/'timings.json').write_text(json.dumps(rows, indent=2))
