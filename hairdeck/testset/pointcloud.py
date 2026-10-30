"""Point-cloud test (scripts/test.py pointcloud) on CHARM's 100 test point clouds (data/charm_test_pc), per sampling
seed in OUT/seed<k> (RUN):
  1. the released condition of every test point cloud, the Dora-VAE 1.1 tokens (hairdeck.encoders.dora.condition_tokens),
     computed in memory once for all seeds;
  2. sample: all 100 conditions in one batch, Euler 50, CFG 2, bf16 autocast -> RUN/JsonResults/{idx:03d}.{json,glb}
     (CHARM-format cards and their mesh), RUN/timing.json;
  3. evaluate: the CHARM-protocol scan of every hairstyle (hairdeck.sampling.scan, EGL) -> RUN/PointClouds; the point
     metrics against CHARM's test point clouds with its protocol (hairdeck.eval.points: CD, EMD, HD, IoU); against the
     original meshes of all 100 test cases (data/test_pc_orig_idx_all.json): the orientation along the cards
     (hairdeck.eval.orient) and the card metrics (hairdeck.eval.cards without interpenetration and card-level EMD: roots,
     root orientation, duplicates, folds, card counts) -> RUN/eval.json.
report: evaluate, then RUN/eval.json with the table (CD, EMD, HD, IoU, Orient.); run: the whole test for the seeds,
then OUT/eval.json with every seed's table and their mean, and the printed table (Table 1).
"""
import json

import numpy as np
import torch

from .. import paths
from ..cards.tokens import LatentDecoder, tokens_to_cards
from ..encoders.dora import condition_tokens
from ..eval import cards as card_eval, orient, points
from ..models.pc_dit import load_pc_dit
from ..sampling.flow import fm_sample
from ..sampling.scan import scan_runs
from .common import SEEDS, save_eval, seed_dir, summarize, write_cards

CKPT = 'ckpt/pointcloud_model/model.safetensors'
TABLES = (("Table 1: CHARM's 100 test point clouds (CD, EMD, HD, IoU with its protocol; Orient. in degrees, on the "
           'original meshes)', (('CD', 4), ('EMD', 4), ('HD', 4), ('IoU', 3), ('Orient.', 1))),)


def sample(ckpt, cond, out, seed=0, steps=50, cfg=2.0, idxs=None, device='cuda'):
    """Hairstyles of a point-cloud card DiT for the test conditions cond [N, 512, 768] (the rows idxs, default all), in
    one batch; writes the cards and OUT/timing.json and returns them."""
    model, tok, ra = load_pc_dit(paths.resolve(ckpt), device)
    vae = LatentDecoder(paths.ckpt(ra.vae), device)
    idxs = list(range(len(cond))) if idxs is None else list(idxs)
    c = torch.from_numpy(cond[idxs]).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
        x = fm_sample(model, model.encode_cond(c), 192, tok.dim, steps, cfg, 1.0, g, 'euler')
    hairs = tokens_to_cards(tok, x, vae)
    write_cards(out, idxs, hairs)
    json.dump(dict(model='pointcloud', ckpt=ckpt, cond=ra.cond, seed=seed, steps=steps, cfg=cfg,
                   n_cards_mean=float(np.mean([len(h) for h in hairs]))), open(f'{out}/timing.json', 'w'), indent=1)
    return hairs


def evaluate(run, workers=12):
    """The scan (point clouds already in RUN/PointClouds are kept), then the metrics:
    {'points': ..., 'orient': ..., 'cards': ...}."""
    scan_runs([run])
    idxs = json.load(open(paths.data('test_pc_orig_idx_all.json')))
    return dict(points=points.evaluate_run(run, idxs, workers=workers), orient=orient.evaluate_run(run, idxs, workers),
                cards=card_eval.summarize(card_eval.run_eval(run, idxs, workers=workers, fast=True)))


def table(parts):
    m = parts['points']['mean']
    return {'CD': m['cd'], 'EMD': m['emd'], 'HD': m['hd'], 'IoU': m['iou'], 'Orient.': parts['orient']['mean']['sec_mean']}


def report(run, settings, workers=12):
    """evaluate, then RUN/eval.json; returns the table."""
    print(f'{run}: scan and evaluation ...', flush=True)
    parts = evaluate(run, workers)
    failed = parts['points']['failed']
    if failed:
        print(f'{run}: {len(failed)} of {len(parts["points"]["per_sample"])} test rows have no hair cards (empty generation; '
              f'rows {failed}); they are left out of the means of the table and listed in eval.json', flush=True)
    t = table(parts)
    save_eval(run, 'pointcloud', settings, t, parts)
    return t


def run(ckpt=CKPT, out=None, seeds=SEEDS, workers=12, device='cuda'):
    """The whole test for the seeds: the condition, then per seed sample and report into OUT/seed<k> (default OUT
    results/pointcloud), then OUT/eval.json and the printed table."""
    out = paths.resolve(out or 'results/pointcloud')
    cond = condition_tokens('test', device=device)
    print(f'condition: {len(cond)} test point clouds', flush=True)
    torch.cuda.empty_cache()
    for s in seeds:
        sample(ckpt, cond, seed_dir(out, s), s, device=device)
        print('sampled ->', seed_dir(out, s), flush=True)
        report(seed_dir(out, s), dict(ckpt=ckpt, seed=s, steps=50, cfg=2.0), workers)
    return summarize(out, 'pointcloud', list(seeds), TABLES)
