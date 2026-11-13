"""Point-cloud model latency at batch 1, as the paper times it (Table 1: input point cloud -> cards, before meshing):
the Dora condition, 50 Euler steps with CFG 2 under bf16 autocast, and the token -> card decode.

python tools/fast_sampler/bench_pc.py [--n 5] [--profile] [--stage all|cond|sample|decode]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hairdeck import paths  # noqa: E402
from hairdeck.cards.tokens import LatentDecoder, tokens_to_cards  # noqa: E402
from hairdeck.encoders import dora  # noqa: E402
from hairdeck.models.pc_dit import load_pc_dit  # noqa: E402
from hairdeck.sampling.flow import fm_sample  # noqa: E402
from hairdeck.testset.pointcloud import CKPT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=5)
ap.add_argument('--profile', action='store_true')
a = ap.parse_args()
dev = 'cuda'
model, tok, ra = load_pc_dit(paths.resolve(CKPT), dev)
vae = LatentDecoder(paths.ckpt(ra.vae), dev)
cond_all = dora.condition_tokens('test', device=dev)  # [100, 512, 768] float16, warms the Dora encoder


def sync():
    torch.cuda.synchronize()


def one(i):
    t = {}
    sync(); t0 = time.perf_counter()
    c = torch.from_numpy(cond_all[i:i + 1]).to(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ctx = model.encode_cond(c)
        x = fm_sample(model, ctx, 192, tok.dim, 50, 2.0, 1.0, g, 'euler')
    sync(); t['sample'] = time.perf_counter() - t0
    t1 = time.perf_counter()
    hairs = tokens_to_cards(tok, x, vae)
    sync(); t['decode'] = time.perf_counter() - t1
    t['total'] = time.perf_counter() - t0
    return t, len(hairs[0])


one(0)  # warm-up (kernel selection, MIOpen / hipBLASLt caches)
if a.profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        one(1)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25))
rows = [one(i) for i in range(1, a.n + 1)]
for k in ('sample', 'decode', 'total'):
    v = np.array([r[0][k] for r in rows])
    print(f'{k:7s} mean {v.mean():.3f} s  min {v.min():.3f} s')
print('cards', [r[1] for r in rows])

# The Dora condition of one point cloud (the paper's time starts at the point cloud).
from hairdeck.encoders.common import condition_plys  # noqa: E402
ply = condition_plys('test')[:1]
for _ in range(2):
    sync(); t0 = time.perf_counter()
    dora.condition_tokens('test', device=dev, plys=ply)
    sync(); dt = time.perf_counter() - t0
print(f'dora cond (1 cloud, incl. load) {dt:.3f} s')
