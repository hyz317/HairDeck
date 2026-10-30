"""Train the image-conditioned card DiT (hairdeck.models.imgmodel.ImgSetDiT) on card-VAE latent tokens, conditioned on
one rendered view of the character (hairdeck.data.imgdata.RenderViews) through a frozen DINOv3 ViT-L/16, with the cards
in the head frame.

Released checkpoint image_model (one GPU, 182.3M parameters, 100K steps, EMA checkpoint ema_100000.pt):
  python train_img_dit.py --style_cfg configs/style_aug.json --name train_image_model --vae train_card_vae --res 448 --bs 48
         --workers 32 --ckpt_every 10000 --steps 100000 --init ckpt/train_pointcloud_model/ema_080000.pt
The released run also mixed in restyled renders (with probability 0.33); this release does not include them yet. The
ablations change one thing: no --init and --steps 180000 (no point-cloud pretraining); a style config with p_style 0
(the base augmentation only). --init also takes the released ckpt/pointcloud_model/model.safetensors (the same weights).
Fixed settings: the renders data/renders (scripts/data.py) of the train characters (data/train_cards_meta.json), the
train cards (data/cards_train.npz) of hairstyles with up to 192 cards; the head frame data/head_frame_train.npz
(hairstyles without a head fit left out) and data/token_stats_head.json (scripts/data.py), the latents
ckpt/<vae>/latents_train_head.npy (scripts/train/export_latents.py). DINOv3 tokens of the --res px crop (CLS, 4 storage
tokens, (res / 16)^2 patches), DiT dim 768, 12 layers, 12 heads; --init copies every tensor of the point-cloud DiT whose
name and shape match (all but the condition input). AdamW lr 1e-4 (betas 0.9 / 0.99, no weight decay), warmup 1000, EMA
0.9995, CFG dropout 0.1, logit-normal t, loss weights root 4 / existence 4 / normal 1, bf16 autocast, gradient clip 1.0.
Epoch e: every view of the kept hairstyles once, in random order; the crop and augmentation of a view depend only on e
and the view. Seed 0 (torch.manual_seed) sets the rest: the new weights of the condition input, the
order of the views and the flow-matching noise. The released run set no seed (torch then draws a random one for each
process, and it was not recorded), so its initial condition-input weights and its order of views cannot be reproduced.
Outputs: ckpt/<name>/{style_aug.json, last.pt, ema_XXXXXX.pt}, logs/<name>.log.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RELEASE)
from hairdeck import paths  # noqa: E402
from hairdeck.ckpt import training_dir  # noqa: E402
from hairdeck.data.hairset import HairSet, TargetGather  # noqa: E402
from hairdeck.data.imgdata import RenderViews, load_style_cfg  # noqa: E402
from hairdeck.models.imgmodel import DinoEncoder, build_img_model, init_from  # noqa: E402
from hairdeck.models.tokenizer import build_tokenizer  # noqa: E402
from hairdeck.sampling.flow import fm_loss  # noqa: E402


def train(a, cfg):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'
    odir = training_dir(a.name)
    os.makedirs(odir, exist_ok=True)
    os.makedirs(paths.logs(), exist_ok=True)
    out = f'{odir}/style_aug.json'
    if os.path.exists(out) and json.load(open(out)) != cfg:
        raise SystemExit(f'{out} exists with a different config; refusing to mix configs in one run')
    json.dump(cfg, open(out, 'w'), indent=1)
    log = open(paths.logs(f'{a.name}.log'), 'a')

    def P(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    P('style_aug', json.dumps(cfg))
    P(json.dumps(vars(a)))
    tr = HairSet('train', a.nmax, np.load(paths.ckpt(a.vae, 'latents_train_head.npy')), train=True)
    hz = np.load(paths.data('head_frame_train.npz'))
    tr.items = tr.items[hz['valid'][tr.items]]
    tok = build_tokenizer(a.vae, head=True, device=dev)
    wvec = tok.loss_weights({'root': a.w_root, 'exist': a.w_exist, 'nrm': a.w_nrm})
    gather = TargetGather(tr, (hz['k'], hz['b']), dev)
    enc = DinoEncoder(a.dino, a.res).to(dev)
    model = build_img_model(a, tok.dim).to(dev)
    if a.init:
        skipped = init_from(model, paths.resolve(a.init))
        P('init from', a.init, '; not initialised:', skipped)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    P('token dim', tok.dim, 'DiT params %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), 'ctx tokens', enc.n_tokens)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    keep = set(tr.items.tolist())  # hairstyles with <= nmax cards (and a head fit)
    meta_json = paths.data('train_cards_meta.json')
    step, t0, losses, epoch = 0, time.time(), [], 0
    while step < a.steps:
        ds = RenderViews(paths.data('renders'), meta_json, out=a.res, train=True, cfg=cfg, seed=epoch)
        ds.keep(keep)
        ds.epoch = epoch
        P(f'epoch {epoch}: {len(ds)} views of {len({it[0] for it in ds.items})} hairstyles')
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                        persistent_workers=False, prefetch_factor=4)
        for b in dl:
            hid = b['hid'].to(dev, non_blocking=True)
            img = b['img'].to(dev, non_blocking=True)
            tgt = gather(hid)
            x1 = tok.encode(tgt['res'], tgt['nrm'], tgt['n'], tgt['shape'], tgt['mask'], tgt['lat'])
            tokens = enc(img)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ctx = model.encode_cond(tokens)
                loss = fm_loss(model, x1, ctx, wvec, a.p_drop, 'logitnorm')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - a.ema)
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                P(f'step {step} loss {np.mean(losses):.5f} gn {gn.item():.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 100:.3f}s/it')
                losses, t0 = [], time.time()
            if step % a.ckpt_every == 0 or step == a.steps:
                torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'step': step,
                            'args': vars(a)}, f'{odir}/last.pt')
                torch.save({'ema': ema.state_dict(), 'step': step, 'args': vars(a)}, f'{odir}/ema_{step:06d}.pt')
            if step >= a.steps:
                break
        epoch += 1
    P('done', step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--style_cfg', default=os.path.join(RELEASE, 'configs', 'style_aug.json'),
                    help='appearance augmentation (hairdeck.data.imgdata.load_style_cfg)')
    ap.add_argument('--vae', default='card_vae')
    ap.add_argument('--res', type=int, default=512, help='crop size of the DINOv3 input')
    ap.add_argument('--init', default='', help='point-cloud DiT checkpoint to start the DiT from')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--steps', type=int, default=80000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    o = ap.parse_args()
    cfg = load_style_cfg(o.style_cfg)
    # the args record of the released checkpoint: the options above and the fixed settings of the docstring (head: the
    # cards are in the head frame)
    a = argparse.Namespace(name=o.name, rep='rgb', render_root='data/renders', vae=o.vae, K=64, nmax=192, dim=768, depth=12,
                           heads=12, dino='vitl16', res=o.res, init=o.init, bs=o.bs, lr=1e-4, steps=o.steps, warmup=1000,
                           ema=0.9995, p_drop=0.1, w_root=4.0, w_exist=4.0, w_nrm=1.0, workers=o.workers,
                           ckpt_every=o.ckpt_every, resume=False, max_steps_this_run=0, head=True, aug=True)
    train(a, cfg)


if __name__ == '__main__':
    main()
