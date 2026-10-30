"""Chain test (scripts/test.py chains) on the 58 test hairstyles with chains (hairdeck.spring.chaindit.Data), per sampling
seed in OUT/seed<k> (RUN):
  1. sample: the chain DiT on the cards of each hairstyle, Euler 50, CFG 2, bf16 autocast; generated and GT chains go
     back from the head frame to the dataset frame with each test character's own head fit -> RUN/chains.json =
     {test row: {pred, gt[, n_cards]}};
  2. evaluate: the chains against the characters' own chains (hairdeck.eval.chains: CD, recall, precision); the hair
     motion they give the artists' cards against the characters' own spring bones (hairdeck.eval.chain_motion, only for
     the artists' cards: the motion is measured on them); how well they fit the cards they were made for
     (hairdeck.eval.chain_fit).
Cards, the condition of the chain DiT (head-frame card tokens):
  gt                the artists' cards (data/cards_test.npz with the head-frame latents ckpt/<vae>/latents_test_head.npy)
  the output folder of scripts/test.py pointcloud (CARDS): chain seed k runs on the cards of CARDS/seed<k>
                    (RUN/JsonResults, dataset frame) moved into the head frame with each test character's own head fit
                    (root -> k * root + b; the root-relative card scaled by k and encoded again by the card VAE), root
                    normals signed by the rule of the training tokens
report: evaluate, then RUN/eval.json with the table: CD, recall, precision, motion (the correlation of the per-point
motion with that under the characters' own spring bones, over the four head motions; the artists' cards only) and its
amplitude, and the fit (share of chain points on the cards); run: the whole test for the seeds, then OUT/eval.json with
every seed's table and their mean, and the printed table (Table 4).
"""
import json
import os

import numpy as np
import torch

from .. import paths
from ..cards.codec import sign_root_normals
from ..cards.vae import load_vae, pad_seqs
from ..eval import chain_fit, chain_motion, chains as chain_eval
from ..eval.cards import load_cards
from ..sampling.flow import fm_sample
from ..spring.chaindit import NCARD, Data, chain_stats, chains_head_to_dataset, decode, load_chain_dit, make_tok
from .common import SEEDS, save_eval, seed_dir, summarize

CKPT = 'ckpt/chain_model/model.safetensors'
COLUMNS = (('CD', 3), ('Rec.', 3), ('Prec.', 3), ('Motion', 3), ('Ampl.', 4), ('Fit', 3))


def check_run(run):
    """A point-cloud run of scripts/test.py (RUN/timing.json)."""
    model = json.load(open(f'{run}/timing.json')).get('model')
    if model != 'pointcloud':
        raise ValueError(f'{run}: not a point-cloud run of scripts/test.py (timing.json model {model!r})')


def pointcloud_run_cards(run, hs, tok, vae):
    """Head-frame card tokens (len(hs), NCARD, 40) of the cards of a point-cloud run (RUN/JsonResults, dataset frame, the
    first NCARD cards with at least 3 points) for the test rows hs, and their card counts."""
    hz = np.load(paths.data('head_frame_test.npz'))
    assert all(hz['valid'][h] for h in hs)
    B, N = len(hs), NCARD
    root = np.zeros((B, N, 1, 3), np.float32)
    nrm = np.zeros((B, N, 3), np.float32)
    shape = np.zeros((B, N), np.int64)
    lat = np.zeros((B, N, vae.zdim), np.float32)
    mask = np.zeros((B, N), bool)
    n_cards = []
    for b, h in enumerate(hs):
        cards = load_cards(f'{run}/JsonResults/{h:03d}.json')[:N]
        assert all(s.shape[1] >= 8 for s, _ in cards), 'needs cards with their root normals (8 columns)'
        k, sh = float(hz['k'][h]), hz['b'][h].astype(np.float64)
        n_cards.append(len(cards))
        if not cards:
            continue
        X, lens = pad_seqs([s for s, _ in cards], vae.L)
        lat[b, :len(cards)] = vae.encode_padded(X * k, lens)
        root[b, :len(cards), 0] = np.stack([k * s[0, :3] + sh for s, _ in cards])
        nrm[b, :len(cards)] = sign_root_normals(np.stack([s[0, 5:8] for s, _ in cards]), np.stack([s[0, :3] for s, _ in cards]))[0]
        shape[b, :len(cards)] = [c for _, c in cards]
        mask[b, :len(cards)] = True
    t = lambda x: torch.from_numpy(x).to(vae.device)  # noqa: E731
    return tok.encode(t(root), t(nrm), None, t(shape), t(mask), t(lat)), n_cards


def sample(ckpt, out, cards='gt', seed=0, steps=50, cfg=2.0, device='cuda'):
    """Chains of a chain DiT for the test hairstyles with chains on `cards` ('gt' or a point-cloud run directory); writes
    OUT/chains.json and returns it."""
    model, ra, dim = load_chain_dit(paths.resolve(ckpt), device)
    stats = chain_stats()
    tok = make_tok(ra['vae'], device)
    D = Data('test', tok, ra['vae'], stats, device)
    hs, n_cards = [int(h) for h in D.items], None
    if cards == 'gt':
        ctok = D.cards(torch.from_numpy(D.items).to(device), perm=False)
    else:
        check_run(cards)
        ctok, n_cards = pointcloud_run_cards(cards, hs, tok, load_vae(paths.ckpt(ra['vae']), device))
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
        x = fm_sample(model, model.encode_cond(ctok), D.nmax, D.dim, steps, cfg, 1.0, g, 'euler').float().cpu().numpy()
    gt = D.chains(torch.tensor(hs, device=device), perm=False).cpu().numpy()
    hz = np.load(paths.data('head_frame_test.npz'))
    res = {}
    for j, (h, xp, xg) in enumerate(zip(hs, x, gt)):
        r = dict(pred=decode(xp, stats), gt=decode(xg, stats))
        if n_cards is not None:
            r['n_cards'] = n_cards[j]
        chains_head_to_dataset(r['pred'] + r['gt'], float(hz['k'][h]), hz['b'][h].astype(np.float64))
        res[h] = r
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f'{out}/chains.json', 'w'))
    return res


def evaluate(run, cards='gt', workers=12):
    """{'chains': ..., 'motion': ... (cards 'gt' only), 'fit': ...} of RUN/chains.json made on `cards`."""
    out = dict(chains=chain_eval.evaluate_run(run))
    if cards == 'gt':
        out['motion'] = chain_motion.evaluate_run(run, workers)
    out['fit'] = chain_fit.evaluate_run(run, cards, 'pred', workers)[1]
    return out


def report(run, settings, workers=12):
    """evaluate (on settings['cards']), then RUN/eval.json; returns the table."""
    print(f'{run}: evaluation ...', flush=True)
    parts = evaluate(run, settings['cards'], workers)
    c, motion = parts['chains']['mean'], parts.get('motion', {}).get('mean', {}).get('all', {})
    table = {'CD': c['cd'], 'Rec.': c['recall'], 'Prec.': c['precision'], 'Motion': motion.get('corr_pred'),
             'Ampl.': motion.get('amp_pred'), 'Fit': parts['fit']['mean']['on_hair']}
    save_eval(run, 'chains', settings, table, parts)
    return table


def run(ckpt=CKPT, out=None, cards='gt', seeds=SEEDS, workers=12, device='cuda'):
    """The whole test for the seeds: per seed k sample (on CARDS/seed<k> for a test folder CARDS) and report into
    OUT/seed<k> (default OUT results/chains, on a test folder results/chains_on_<its name>), then OUT/eval.json and the
    printed table."""
    cards = cards if cards == 'gt' else paths.resolve(cards).rstrip('/')
    out = paths.resolve(out or ('results/chains' if cards == 'gt' else f'results/chains_on_{os.path.basename(cards)}'))
    for s in seeds:
        on = cards if cards == 'gt' else seed_dir(cards, s)
        if on != 'gt' and not os.path.isfile(f'{on}/timing.json'):
            raise FileNotFoundError(f'{on}: no run of scripts/test.py pointcloud for seed {s}')
        res = sample(ckpt, seed_dir(out, s), on, s, device=device)
        print(f'sampled {len(res)} test hairstyles ->', seed_dir(out, s), flush=True)
        report(seed_dir(out, s), dict(ckpt=ckpt, seed=s, cards=on, steps=50, cfg=2.0), workers)
    on = "the artists' cards" if cards == 'gt' else f'the cards of {cards}/seed<k>'
    return summarize(out, 'chains', list(seeds), ((f'Table 4: the 58 test hairstyles with chains, on {on}', COLUMNS),))
