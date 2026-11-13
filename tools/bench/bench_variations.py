"""Real-request sequential/batched comparison, with saved same-seed outputs."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.infer.image import Pipeline

p = Pipeline()
root = Path(paths.results('variation-benchmark'))
root.mkdir(parents=True, exist_ok=True)
image = paths.data('demo_examples', 'ex02.png')
rows = []
for count in (2, 4):
    for label, batched in [('sequential_first', False), ('batched_first', True),
                            ('sequential', False), ('batched', True)]:
        torch.cuda.reset_peak_memory_stats()
        out = root/f'{count}-{label}'
        r = p.run(image, seed=1, n_samples=count, batch_variations=batched, out_dir=str(out))
        row = dict(count=count, label=label, ok=r['ok'], timings=r['timings'], notes=r['notes'],
                   samples=[{k:s[k] for k in ('seed','n_cards','n_chains','colors')} for s in r['samples']],
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (root/'timings.json').write_text(json.dumps(rows, indent=2))
        if not r['ok'] or any(not s['colors'] or not s['n_chains'] for s in r['samples']):
            raise RuntimeError('Incomplete variation output')
    changes=[]
    for seed in range(1,count+1):
        a=np.load(root/f'{count}-sequential'/f'tokens_seed{seed}.npy')
        b=np.load(root/f'{count}-batched'/f'tokens_seed{seed}.npy')
        changes.append(dict(seed=seed,identical=bool(np.array_equal(a,b)),
                            relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                            max_absolute=float(np.abs(a-b).max())))
    (root/f'{count}-numerics.json').write_text(json.dumps(changes,indent=2))
    print('NUMERICS',count,json.dumps(changes),flush=True)
