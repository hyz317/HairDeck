"""Real-request compatibility check against an explicit checkout; requires installed assets.

Run this same harness on the baseline and candidate. No synthetic models or unit tests.
The first request is process-cold with disk caches retained; warm samples include all
generation stages, colors, chains and GLB writes. Export timings are separate.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--revision', help='Explicit source commit for an archive/container without accessible Git metadata')
parser.add_argument('--no-exports', action='store_true', help='Inference-only coverage; explicitly records skipped exports')
args = parser.parse_args()
repo = args.repo.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
os.chdir(repo)
sys.path.insert(0, str(repo))
os.environ.update(HAIRDECK_FAST='0', HAIRDECK_BATCH_VARIATIONS='0', HAIRDECK_PREP_CACHE='0')
os.environ['HAIRDECK_VRM_WORK'] = str(out/'vrm-worker')
import numpy as np
import torch
import torchvision
from hairdeck import paths
from hairdeck.infer.image import Pipeline

report = dict(revision=args.revision or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
              torchvision=torchvision.__version__, cuda=torch.version.cuda, hip=torch.version.hip,
              gpu=torch.cuda.get_device_name(), disk_caches='retained', steps=50, cfg=2,
              seeds=[1, 2], exports='skipped explicitly' if args.no_exports else 'requested', rows=[], success=False)

def save():
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

try:
    started = time.perf_counter()
    pipe = Pipeline()
    report['load_seconds'] = time.perf_counter()-started
    image = paths.data('demo_examples', 'ex02.png')
    report['input_sha256'] = digest(image)
    for index in range(4):
        name = 'first' if index == 0 else f'warm-{index}'
        target = out/name
        torch.cuda.reset_peak_memory_stats()
        result = pipe.run(image, seed=1, n_samples=2, steps=50, cfg=2,
                          colors=True, chains=True, out_dir=str(target))
        row = dict(label=name, ok=result['ok'], timings=result['timings'], notes=result['notes'],
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), samples=[])
        for sample in result['samples']:
            seed = sample['seed']
            row['samples'].append({k: sample[k] for k in ('seed', 'n_cards', 'n_chains', 'colors')})
            row['samples'][-1]['files'] = {filename: digest(target/filename) for filename in
                (f'tokens_seed{seed}.npy', f'cards_seed{seed}.json', f'chains_seed{seed}.json', f'hair_seed{seed}.glb')}
        report['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
        if not result['ok'] or len(result['samples']) != 2 or any(
                not s['colors'] or not s['n_cards'] or not s['n_chains'] for s in result['samples']):
            raise RuntimeError('Generation did not produce both colored, rigged hairstyles')
    report['warm_median_seconds'] = float(np.median([row['timings']['total'] for row in report['rows'][1:]]))
    if not args.no_exports:
        from hairdeck.vrm.export import export_vrm, turn_preview
        colour = result['params']['hair_color'] if 'hair_color' in result.get('params', {}) else [.32, .22, .16]
        info = json.loads((target/'info.json').read_text(encoding='utf-8'))
        colour = info.get('hair_color', colour)
        report['vrm'] = export_vrm(str(target), 1, colour, render=True)
        save()
        if not report['vrm']['ok'] or report['vrm'].get('render_error'):
            raise RuntimeError('VRM export, validation or rest preview failed')
        report['preview'] = turn_preview(str(target), 1, colour)
        if report['preview'].get('frames') != 90 or not Path(report['preview']['out']).is_file():
            raise RuntimeError('Motion preview did not produce the expected 90-frame video')
    report['success'] = True
except Exception as error:
    report['error'] = f'{type(error).__name__}: {error}'
    save()
    raise
save()
print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'vrm', 'preview')}, indent=2), flush=True)
