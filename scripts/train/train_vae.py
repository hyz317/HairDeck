"""Train the card VAE (hairdeck.cards.vae.CardVAE) on the padded train cards.

Released checkpoint card_vae (one GPU):
  python train_vae.py --name train_card_vae --zdim 32 --nf 6 --steps 80000 --dec ada --geo_loss --uniform_target --accum 2
Inputs: data/train_templates.npz (scripts/data.py); caches written next to it on first use:
data/cards_train_pad{L}.npz, *_u.npy, *_stats.json, *_uni.npy. Outputs: ckpt/<name>/{model.pt, stats.json},
log logs/<name>.log.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hairdeck import paths  # noqa: E402
from hairdeck.ckpt import training_dir  # noqa: E402
from hairdeck.cards.vae import BUCKETS, CardVAE, Norm, arc_u, build_pad_cache, uniform_targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--zdim', type=int, default=64)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--L', type=int, default=128)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--kl', type=float, default=1e-6)
    ap.add_argument('--w_n', type=float, default=0.1)
    ap.add_argument('--nf', type=int, default=6)
    ap.add_argument('--fp32', action='store_true', help='no bf16 autocast, no TF32')
    ap.add_argument('--dec', default='tf', choices=['tf', 'mlp', 'ada'])
    ap.add_argument('--accum', type=int, default=1, help='split each batch into chunks (same effective batch, less memory)')
    ap.add_argument('--integ', action='store_true', help='decoder outputs dp/du, positions integrated from the root')
    ap.add_argument('--uniform_target', action='store_true', help='decoder queried at uniform u (as at inference), targets interpolated')
    ap.add_argument('--geo_loss', action='store_true', help='weight channels by their world-unit scale: w,t errors count as much as equal xyz errors in world units')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = not args.fp32
    torch.backends.cudnn.allow_tf32 = not args.fp32
    dev = 'cuda'
    odir = training_dir(args.name)
    os.makedirs(odir, exist_ok=True)
    os.makedirs(paths.logs(), exist_ok=True)
    log = open(paths.logs(f'{args.name}.log'), 'a')

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    P(json.dumps(vars(args)))
    z = np.load(build_pad_cache(args.L))
    X, lens = z['X'], z['lens']
    ucache, scache = paths.data(f'cards_train_pad{args.L}_u.npy'), paths.data(f'cards_train_pad{args.L}_stats.json')
    if os.path.isfile(ucache) and os.path.isfile(scache):
        U, stats = np.load(ucache), json.load(open(scache))
    else:
        U = arc_u(X, lens)
        m = np.arange(args.L)[None] < lens[:, None]
        Xm = X[m]
        stats = {'mu': Xm.mean(0).tolist(), 'sd': Xm.std(0).tolist()}
        stats['mu'][:3] = [0.0, 0.0, 0.0]
        stats['sd'][:3] = [float(Xm[:, :3].std())] * 3
        del Xm
        np.save(ucache + '.tmp.npy', U)
        os.replace(ucache + '.tmp.npy', ucache)
        json.dump(stats, open(scache, 'w'))
    json.dump(stats, open(f'{odir}/stats.json', 'w'))
    norm = Norm(stats, dev)
    if args.uniform_target:
        ucache2 = paths.data(f'cards_train_pad{args.L}_uni.npy')
        if os.path.isfile(ucache2):
            Xu = np.load(ucache2)
        else:
            Xu = uniform_targets(X, lens, U)
            np.save(ucache2 + '.tmp.npy', Xu)
            os.replace(ucache2 + '.tmp.npy', ucache2)
        Xut = torch.from_numpy(Xu)
    sd = np.array(stats['sd'])
    cw = torch.tensor((sd / sd[0]) ** 2 if args.geo_loss else np.ones(5), dtype=torch.float32, device=dev)
    cw = cw / cw.mean()  # keep the loss scale comparable
    P('channel loss weights', np.round(cw.cpu().numpy(), 4).tolist())
    Xt, Ut, Lt = torch.from_numpy(X), torch.from_numpy(U), torch.from_numpy(lens)
    bid = np.searchsorted(BUCKETS, lens)
    bucket_idx = [np.nonzero(bid == k)[0] for k in range(len(BUCKETS))]
    bprob = np.array([len(b) for b in bucket_idx], np.float64)
    bprob /= bprob.sum()
    model = CardVAE(args.L, args.dim, args.depth, 8, args.zdim, args.nf, args.dec, args.integ).to(dev)
    P('params', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * (0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)) * 0.9 + 0.1))
    ar = torch.arange(args.L, device=dev)
    rng = np.random.default_rng(0)
    t0 = time.time()
    hist = []
    for step in range(args.steps + 1):
        k = rng.choice(len(BUCKETS), p=bprob)
        idx = torch.from_numpy(rng.choice(bucket_idx[k], args.bs, replace=len(bucket_idx[k]) < args.bs))
        Lb = BUCKETS[k]
        x = Xt[idx, :Lb].to(dev, non_blocking=True)
        u = Ut[idx, :Lb].to(dev, non_blocking=True)
        n = Lt[idx].to(dev)
        mask = ar[None, :Lb] < n[:, None]
        xn = norm.fwd(x) * mask[..., None]
        if args.uniform_target:  # decoder queries/targets as at inference: u_j = j/(n-1)
            uq = (ar[None, :Lb].float() / (n[:, None].float() - 1).clamp(min=1)).clamp(max=1.0) * mask
            xt = norm.fwd(Xut[idx, :Lb].to(dev, non_blocking=True)) * mask[..., None]
        else:
            uq, xt = u, xn
        opt.zero_grad(set_to_none=True)
        nvalid = mask.sum()
        acc = []
        for xs, us, ms, ns, uqs, xts in zip(xn.chunk(args.accum), u.chunk(args.accum), mask.chunk(args.accum), n.chunk(args.accum), uq.chunk(args.accum), xt.chunk(args.accum)):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.fp32):
                mu, logvar = model.encode(xs, us, ms)
                zz = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if args.kl > 0 else mu
                feat = model.decode(zz, uqs, ms)
                nlog = model.n_head(zz)
            rec = ((((feat.float() - xts) ** 2) * cw).mean(-1) * ms).sum() / nvalid  # chunk share of the full-batch mean
            kl = 0.5 * (mu.float() ** 2 + logvar.float().exp() - 1 - logvar.float()).sum(-1).mean() / args.accum
            ce = F.cross_entropy(nlog.float(), (ns - 1).clamp(0, args.L - 1)) / args.accum
            (rec + args.w_n * ce + (args.kl * kl if args.kl > 0 else 0.0)).backward()
            acc.append((rec.item(), ce.item(), kl.item(), (nlog.argmax(-1) + 1 == ns).float().sum().item()))
        a = np.array(acc)
        rec, ce, kl, nacc = a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum(), a[:, 3].sum() / len(n)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append((rec, ce, kl, nacc))
        if step % 500 == 0:
            h = np.mean(hist, 0)
            hist = []
            P(f'step {step} rec {h[0]:.6f} n_ce {h[1]:.4f} kl {h[2]:.1f} n_acc {h[3]:.4f} {time.time() - t0:.0f}s')
    torch.save({'model': model.state_dict(), 'args': vars(args)}, f'{odir}/model.pt')
    P('saved')


if __name__ == '__main__':
    main()
