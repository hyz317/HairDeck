"""The image pipeline with the release sampler and with hairdeck.sampling.fast: one demo example, seeds 1 and 2, 50 Euler
steps, CFG 2, colours and chains, no file writes; a warm-up request, then --n timed requests per sampler.

python tools/fast_sampler/bench_image.py [--n 3] [--example ex02.png] [--json out.json]
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hairdeck import paths  # noqa: E402
from hairdeck.infer.image import Pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=3)
ap.add_argument('--example', default='ex02.png')
ap.add_argument('--json')
a = ap.parse_args()
image = paths.data('demo_examples', a.example)
pipe = Pipeline()
report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda, hip=torch.version.hip,
              platform=platform.platform(), example=a.example, seeds=[1, 2], steps=50, cfg=2, modes={})
for mode, env in (('release', {'HAIRDECK_FAST': '0'}), ('fast', {'HAIRDECK_FAST': '1', 'HAIRDECK_COMPILE': '1'})):
    os.environ.update(env)
    rows = []
    for i in range(a.n + 1):  # the first request warms up (and compiles, for 'fast')
        t0 = time.perf_counter()
        r = pipe.run(image, seed=1, n_samples=2, steps=50, cfg=2, write=False)
        assert r['ok'], r['message']
        rows.append(dict({k: round(v, 3) for k, v in r['timings'].items()}, wall=round(time.perf_counter() - t0, 3)))
    warm = rows[1:]
    med = {k: float(np.median([r[k] for r in warm])) for k in ('wall', 'dino+dit+decode')}
    report['modes'][mode] = dict(first=rows[0], warm=warm, median=med)
    print(f"{mode:8s} first {rows[0]['wall']:6.2f} s   warm median: request {med['wall']:6.2f} s, "
          f"dino+dit+decode {med['dino+dit+decode']:6.2f} s", flush=True)
if a.json:
    with open(a.json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
