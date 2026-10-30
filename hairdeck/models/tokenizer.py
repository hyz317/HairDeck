"""Card tokens of the set DiTs: [root (3), card-VAE latent (zdim, standardised), root width direction (3), shape class
(+-1), existence (+-1)]; with zdim 32 a token is 40-d. Roots are standardised with the token stats file
(data/token_stats.json for the point-cloud model, data/token_stats_head.json for the head-frame image and chain models),
latents with ckpt/<vae>/latent_stats[_head].json. The point count of a card comes from the card VAE's count head at
decoding time.
"""
import json
import os

import numpy as np
import torch

from .. import paths


class LatentTokenizer:
    def __init__(self, stats, lstats, device='cuda'):
        self.s = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in stats.items()}
        self.lmu = torch.as_tensor(lstats['mu'], dtype=torch.float32, device=device)
        self.lsd = torch.as_tensor(lstats['sd'], dtype=torch.float32, device=device)
        self.zdim = len(lstats['mu'])
        self.dim = 3 + self.zdim + 3 + 2
        self.groups = {'root': slice(0, 3), 'shape': slice(3, 3 + self.zdim), 'nrm': slice(3 + self.zdim, 6 + self.zdim),
                       'cls': slice(6 + self.zdim, 7 + self.zdim), 'exist': slice(7 + self.zdim, 8 + self.zdim)}

    def encode(self, res, nrm, n, shape, mask, lat):
        """res (B,N,K>=1,5) (only the root res[...,0,:3] is used), nrm (B,N,3), n (B,N), shape (B,N), mask (B,N) bool,
        lat (B,N,zdim) -> tokens (B,N,dim)."""
        root = (res[..., 0, :3] - self.s['root_mu']) / self.s['root_sd']
        z = (lat - self.lmu) / self.lsd
        tok = torch.cat([root, z, nrm, shape.float()[..., None] * 2 - 1, torch.ones_like(root[..., :1])], -1)
        tok = torch.where(mask[..., None], tok, torch.zeros_like(tok))
        tok[..., -1] = torch.where(mask, 1.0, -1.0)
        return tok

    def decode(self, tok):
        g = self.groups
        return dict(root=tok[..., g['root']] * self.s['root_sd'] + self.s['root_mu'], z=tok[..., g['shape']] * self.lsd + self.lmu,
                    nrm=tok[..., g['nrm']], shape=(tok[..., g['cls']][..., 0] > 0).long(), exist=tok[..., g['exist']][..., 0] > 0)

    def loss_weights(self, w):
        wv = torch.ones(self.dim, device=self.lmu.device)
        for k, sl in self.groups.items():
            wv[sl] = w.get(k, 1.0)
        return wv


def compute_stats(cards):
    """Token stats from the fixed-K card features {'res': (C,K,5), 'n': (C,)} (400K random cards, seed 0)."""
    res, n = cards['res'], cards['n']
    idx = np.random.default_rng(0).choice(len(res), min(len(res), 400000), replace=False)
    r = res[idx].astype(np.float64)
    rel = r[:, 1:, :3] - r[:, :1, :3]
    return dict(root_mu=r[:, 0, :3].mean(0), root_sd=r[:, 0, :3].std(0), rel_sd=rel.std(),
                w_mu=r[:, :, 3].mean(), w_sd=r[:, :, 3].std(), t_mu=r[:, :, 4].mean(), t_sd=r[:, :, 4].std(),
                logn_mu=np.log(n[idx]).mean(), logn_sd=np.log(n[idx]).std())


def build_tokenizer(vae, head=False, train_ds=None, device='cuda'):
    """Tokenizer for card-VAE `vae` (a checkpoint directory name under the checkpoint root).
    Dataset frame (the point-cloud model): data/token_stats.json (computed from train_ds and written if missing) and
    ckpt/<vae>/latent_stats.json. Head frame (head=True; the image model and the chain model):
    data/token_stats_head.json and ckpt/<vae>/latent_stats_head.json (scripts/data.py and scripts/train/export_latents.py)."""
    if head:
        return LatentTokenizer(json.load(open(paths.data('token_stats_head.json'))),
                               json.load(open(paths.ckpt(vae, 'latent_stats_head.json'))), device=device)
    stats_path = paths.data('token_stats.json')
    if not os.path.isfile(stats_path):
        st = compute_stats(dict(res=train_ds.res, n=train_ds.n))
        json.dump({k: np.asarray(v).tolist() for k, v in st.items()}, open(stats_path, 'w'))
    stats = json.load(open(stats_path))
    return LatentTokenizer(stats, json.load(open(paths.ckpt(vae, 'latent_stats.json'))), device=device)
