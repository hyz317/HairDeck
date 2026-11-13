"""hairdeck.sampling.fast against the original sampler: the point-cloud DiT (192 tokens) and the chain DiT (48 tokens) at
batch 1 and batch 4, Euler 50, CFG 2, bf16 autocast; prints the max abs difference and the timings of both.

python tools/fast_sampler/check_fast.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch  # noqa: E402

from hairdeck import paths  # noqa: E402
from hairdeck.models.pc_dit import load_pc_dit  # noqa: E402
from hairdeck.sampling import fast  # noqa: E402
from hairdeck.sampling.flow import fm_sample  # noqa: E402
from hairdeck.testset.pointcloud import CKPT  # noqa: E402


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def run(model, cond, n_tok, dim, mode):
    os.environ['HAIRDECK_FAST'] = '0' if mode == 'orig' else '1'
    os.environ['HAIRDECK_GRAPHS'] = '1' if mode in ('graph', 'compile') else '0'
    os.environ['HAIRDECK_COMPILE'] = '1' if mode in ('compile', 'compile_nograph') else '0'
    g = torch.Generator(device='cuda').manual_seed(0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return fm_sample(model, model.encode_cond(cond), n_tok, dim, 50, 2.0, 1.0, g, 'euler')


import argparse  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument('--modes', nargs='+', default=['orig', 'nograph', 'graph'])
MODES = ap.parse_args().modes
model, tok, ra = load_pc_dit(paths.resolve(CKPT), 'cuda')
torch.manual_seed(0)
for B in (1, 4):
    cond = torch.randn(B, 512, 768, device='cuda').half()
    res = {}
    for mode in MODES:
        run(model, cond, 192, tok.dim, mode)  # warm-up
        res[mode] = timed(lambda: run(model, cond, 192, tok.dim, mode))
    ref = res[MODES[0]][0]
    for mode, (x, dt) in res.items():
        print(f'pc_dit B={B} {mode:8s} {dt * 1e3:7.1f} ms  max|dx| {(x - ref).abs().max().item():.3g}  '
              f'bitwise {torch.equal(x, ref)}')

# One velocity evaluation at a few t: the restructured forward against the module's, relative to the module's noise
# floor (the module's own fp32-weights forward against its bf16 autocast forward).
from hairdeck.sampling.fast import _Denoiser  # noqa: E402
cond = torch.randn(1, 512, 768, device='cuda').half()
x = torch.randn(2, 192, tok.dim, device='cuda')
with torch.no_grad():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        ctx = model.encode_cond(cond)
    for t in (1.0, 0.5, 0.02):
        T = torch.full((1, 1), t, device='cuda')
        tt = torch.full((2,), t, device='cuda')
        drop = torch.tensor([False, True], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ref = model(x, tt, torch.cat([ctx, ctx]), drop).float()
            d = _Denoiser(model, ctx, 2.0, T)
            fast = d.forward(x, [a[0] for a in d.ada], d.f_ada[0]).float()
        full = model(x, tt, torch.cat([ctx, ctx]).float(), drop).float()  # fp32 reference
        n = ref.norm()
        print(f't={t}: |fast-autocast|/|v| {(fast - ref).norm() / n:.2e}   |autocast-fp32|/|v| {(ref - full).norm() / n:.2e}'
              f'   |fast-fp32|/|v| {(fast - full).norm() / n:.2e}')
