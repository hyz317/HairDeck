"""Train the spring-bone chain DiT (hairdeck.spring.chaindit.ChainDiT), conditioned on the hairstyle's card tokens, in
the head frame of the image model.

Released checkpoint chain_model (one GPU, head frame, selected checkpoint ema_080000.pt):
  python train_chain_dit.py --name train_chain_model --vae train_card_vae --steps 80000
Settings (defaults): dim 384, depth 8, heads 6, batch 128, AdamW lr 2e-4 (betas 0.9 / 0.99), warmup 1000, EMA 0.999,
CFG dropout 0.1, existence loss weight 4, bf16 autocast, gradient clip 1.0, generator seed 0, checkpoint every 5000.
Inputs: data/cards_train.npz, data/token_stats_head.json, data/head_frame_train.npz, data/spring/chains_train_head.npz
and chain_stats_head.json (scripts/data.py), ckpt/<vae>/latents_train_head.npy and latent_stats_head.json
(scripts/train/export_latents.py). Outputs: ckpt/<name>/ema_XXXXXX.pt, logs/<name>.log.
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
from hairdeck.sampling.flow import fm_loss  # noqa: E402
from hairdeck.spring.chaindit import ChainDiT, Data, chain_stats, make_tok  # noqa: E402


def train(a):
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    stats = chain_stats()
    tok = make_tok(a.vae, dev)
    D = Data('train', tok, a.vae, stats, dev)
    model = ChainDiT(D.dim, a.dim, a.depth, a.heads).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    wvec = torch.ones(D.dim, device=dev)
    wvec[-1] = 4.0  # existence, as w_exist in the card DiT
    odir = training_dir(a.name)
    os.makedirs(odir, exist_ok=True)
    os.makedirs(paths.logs(), exist_ok=True)
    log = open(paths.logs(f'{a.name}.log'), 'a')
    print(json.dumps(vars(a)), 'hairstyles', len(D.items), 'params %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), file=log, flush=True)
    g = torch.Generator(device=dev).manual_seed(0)
    items = torch.from_numpy(D.items).to(dev)
    t0, losses = time.time(), []
    for step in range(1, a.steps + 1):
        hid = items[torch.randint(0, len(items), (a.bs,), device=dev, generator=g)]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ctx = model.encode_cond(D.cards(hid, g))
            loss = fm_loss(model, D.chains(hid, g), ctx, wvec, a.p_drop)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        losses.append(loss.item())
        if step % 100 == 0:
            print(f'step {step} loss {np.mean(losses):.5f} {(time.time() - t0) / 100:.3f}s/it', file=log, flush=True)
            losses, t0 = [], time.time()
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save({'ema': ema.state_dict(), 'step': step, 'args': vars(a), 'dim': D.dim}, f'{odir}/ema_{step:06d}.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='chaindit_S')
    ap.add_argument('--vae', default='card_vae')
    ap.add_argument('--dim', type=int, default=384)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--p_drop', type=float, default=0.1)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    ap.add_argument('--cfg', type=float, default=2.0, help='stored in the run args')
    ap.add_argument('--seed', type=int, default=0, help='stored in the run args')
    a = ap.parse_args()
    a.head, a.mode, a.ckpt, a.out = True, 'train', '', ''  # the args record of the released checkpoint (head frame)
    train(a)


if __name__ == '__main__':
    main()
