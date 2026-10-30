"""Train the point-cloud conditioned card DiT (hairdeck.models.pc_dit.GenericCondDiT) on card-VAE latent tokens.

Released checkpoint pointcloud_model (one GPU, 80K steps, EMA checkpoint ema_080000.pt):
  python train_pc_dit.py --cond dora --name train_pointcloud_model --vae train_card_vae --dim 768 --depth 12 --heads 12 --steps 80000
Fixed settings: the train cards (data/cards_train.npz), card VAE --vae (default card_vae), up to 192
cards, batch --bs 64, AdamW lr 1e-4 (betas 0.9 / 0.99, no weight decay), warmup 2000, EMA 0.9995, CFG dropout 0.1,
logit-normal t, loss weights root 4 / existence 4 / normal 1, bf16 autocast, gradient clip 1.0, seed 0.
Condition: data/cond_<cond>_train.npy (scripts/data.py). Outputs: ckpt/<name>/{last.pt, ema_XXXXXX.pt}, logs/<name>.log.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hairdeck import paths  # noqa: E402
from hairdeck.ckpt import training_dir  # noqa: E402
from hairdeck.data.hairset import GPUBatcher, HairSet, to_tokens  # noqa: E402
from hairdeck.models.pc_dit import build_model, cond_path  # noqa: E402
from hairdeck.models.tokenizer import build_tokenizer  # noqa: E402
from hairdeck.sampling.flow import fm_loss  # noqa: E402


def train(a):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = 'cuda'
    tr = HairSet('train', 192, np.load(paths.ckpt(a.vae, 'latents_train.npy')), train=True, cond_path=cond_path(a.cond, 'train'))
    tok = build_tokenizer(a.vae, train_ds=tr, device=dev)
    wvec = tok.loss_weights({'root': 4.0, 'exist': 4.0, 'nrm': 1.0})
    model = build_model(a, tok.dim).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 2000))
    odir = training_dir(a.name)
    os.makedirs(odir, exist_ok=True)
    os.makedirs(paths.logs(), exist_ok=True)
    log = open(paths.logs(f'{a.name}.log'), 'a')

    def P(*x):
        print(*x, file=log, flush=True)
    P(json.dumps(vars(a)), 'params %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))
    step = 0
    if a.resume and os.path.isfile(f'{odir}/last.pt'):  # full state (model, EMA, optimiser) saved with every checkpoint
        ck = torch.load(f'{odir}/last.pt', map_location='cpu')
        model.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        step = ck['step']
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_, s0=step: min(1.0, (s_ + s0 + 1) / 2000))
        P('resumed', step)
    P('loaded cond %s into RAM in %.0fs' % (tr.cond_path, tr.load_cond_ram()))
    dl = GPUBatcher(tr, a.bs, dev, seed=step)
    t0, losses = time.time(), []
    for b in dl:
        x1 = to_tokens(tok, b, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ctx = model.encode_cond(b['cond'].to(dev, non_blocking=True))
            loss = fm_loss(model, x1, ctx, wvec, 0.1, 'logitnorm')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.9995)
        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            P(f'step {step} loss {np.mean(losses):.5f} {(time.time() - t0) / 100:.3f}s/it')
            losses, t0 = [], time.time()
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'step': step,
                        'args': vars(a)}, f'{odir}/last.pt')
            torch.save({'ema': ema.state_dict(), 'step': step, 'args': vars(a)}, f'{odir}/ema_{step:06d}.pt')
        if step >= a.steps:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='dora')
    ap.add_argument('--name', default='pointcloud_model')
    ap.add_argument('--vae', default='card_vae')
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--ckpt_every', type=int, default=10000)
    ap.add_argument('--cfg', type=float, default=2.0, help='stored in the run args; scripts/test.py samples with CFG 2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true', help='continue from ckpt/NAME/last.pt')
    a = ap.parse_args()
    a.mode = 'train'
    a.ckpt, a.out, a.cond_file, a.idxs = '', '', '', ''  # the args record of the released checkpoint
    train(a)


if __name__ == '__main__':
    main()
