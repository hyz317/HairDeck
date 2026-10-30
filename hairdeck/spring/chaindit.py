"""Spring-bone chain generator: a small set DiT over chain tokens, conditioned on the hairstyle's card tokens.

Chain token (normalised by data/spring/chain_stats_head.json): [M = 8 arc-length samples x 3, joint count,
stiffness] + existence flag (+1 / -1); NMAX = 48 slots, flow matching exactly like the card DiT (sampling/flow.py).
Condition: the hairstyle's card tokens as produced by models.tokenizer.LatentTokenizer (192 x 40, existence flag
included, so empty card slots can be ignored), linearly projected and cross-attended; the pooled projection feeds adaLN.
At inference the card tokens can come straight from a card DiT (no decoding needed; see cards.tokens.clean_tokens).
Everything is in the head frame of the image model: card tokens with head-frame roots, head-frame latents and token
stats, chains from scripts/data.py.
Drivers: scripts/train/train_chain_dit.py, scripts/test.py chains.
"""
import json

import numpy as np
import torch
import torch.nn as nn

from .. import paths
from ..ckpt import load as load_ckpt
from ..models.setdit import SetDiT
from ..models.tokenizer import build_tokenizer

CARD_DIM, NCARD = 40, 192


class ChainDiT(SetDiT):
    def __init__(self, token_dim, dim=384, depth=8, heads=6):
        super().__init__(token_dim, dim, depth, heads, cond_dim=CARD_DIM, head_dim=CARD_DIM, n_ctx=NCARD)
        self.card_norm = nn.LayerNorm(CARD_DIM)
        self.card_proj = nn.Linear(CARD_DIM, dim)
        self.card_pos = nn.Parameter(torch.zeros(1, 1, dim))  # cards are a set: no per-slot positions

    def encode_cond(self, cards):
        """cards: (B, NCARD, CARD_DIM) card tokens."""
        return self.card_proj(self.card_norm(cards.float())) + self.card_pos


def make_tok(vae, dev):
    return build_tokenizer(vae, head=True, device=dev)


def chain_stats():
    return json.load(open(paths.data('spring', 'chain_stats_head.json')))


class Data:
    """Card tokens and chain tokens of hairstyles that have chains, on the GPU; random slot permutation per sample."""

    def __init__(self, split, tok, vae, stats, dev):
        z = np.load(paths.data(f'cards_{split}.npz'))
        lat = np.load(paths.ckpt(vae, f'latents_{split}_head.npy'))
        ch = np.load(paths.data('spring', f'chains_{split}_head.npz'))
        hp = z['hair_ptr']
        nc = np.diff(hp)
        self.items = np.nonzero(ch['valid'][:len(nc)] & (nc <= NCARD) & (nc > 0))[0]
        self.dev, self.tok = dev, tok
        self.hp = torch.from_numpy(hp).to(dev)
        root = z['res'][:, :1].copy()
        # card roots into the head frame (latents / chains are already head-frame files)
        hz = np.load(paths.data(f'head_frame_{split}.npz'))
        card_h = np.repeat(np.arange(len(hp) - 1), nc)
        root[:, 0, :3] = hz['k'][card_h, None] * root[:, 0, :3] + hz['b'][card_h]
        self.root = torch.from_numpy(root).to(dev)
        self.nrm = torch.from_numpy(z['nrm']).to(dev)
        self.n = torch.from_numpy(z['n'].astype(np.int64)).to(dev)
        self.shape = torch.from_numpy(z['shape'].astype(np.int64)).to(dev)
        self.lat = torch.from_numpy(lat).to(dev)
        mu, sd = torch.tensor(stats['mu'], device=dev), torch.tensor(stats['sd'], device=dev)
        self.ctok = (torch.from_numpy(ch['tok']).to(dev) - mu) / sd
        self.ccount = torch.from_numpy(ch['count'].astype(np.int64)).to(dev)
        self.nmax = self.ctok.shape[1]
        self.dim = self.ctok.shape[2] + 1

    def cards(self, hid, g=None, perm=True):
        B, N, dev = len(hid), NCARD, self.dev
        a = self.hp[hid]
        cnt = torch.minimum(self.hp[hid + 1] - a, torch.tensor(N, device=dev))
        ar = torch.arange(N, device=dev)
        if perm:
            slot = torch.argsort(torch.rand(B, N, device=dev, generator=g), 1)
        else:
            slot = ar[None].expand(B, N)
        val = torch.where(ar[None] < cnt[:, None], a[:, None] + ar[None], torch.full((B, N), -1, device=dev))
        cid = torch.full((B, N), -1, dtype=torch.long, device=dev).scatter_(1, slot, val)
        mask = cid >= 0
        safe = cid.clamp(min=0)
        return self.tok.encode(self.root[safe] * mask[..., None, None], self.nrm[safe] * mask[..., None],
                               torch.where(mask, self.n[safe], torch.full_like(safe, 3)), self.shape[safe] * mask, mask,
                               self.lat[safe] * mask[..., None])

    def chains(self, hid, g=None, perm=True):
        B, N, dev = len(hid), self.nmax, self.dev
        c = self.ccount[hid]
        ar = torch.arange(N, device=dev)
        x = torch.zeros(B, N, self.dim, device=dev)
        src = self.ctok[hid]
        if perm:
            slot = torch.argsort(torch.rand(B, N, device=dev, generator=g), 1)
        else:
            slot = ar[None].expand(B, N)
        keep = ar[None] < c[:, None]
        x[..., :-1] = torch.zeros_like(src).scatter_(1, slot[..., None].expand_as(src), src * keep[..., None])
        ex = torch.zeros(B, N, dtype=torch.bool, device=dev).scatter_(1, slot, keep)
        x[..., -1] = torch.where(ex, 1.0, -1.0)
        x[..., :-1] *= ex[..., None]
        return x


def decode(x, stats, M=8):
    """Chain tokens (N, D) -> list of dict(path (M, 3), joints (n, 3), n_joints, stiffness) for existing slots."""
    mu, sd = np.array(stats['mu']), np.array(stats['sd'])
    out = []
    for t in x:
        if t[-1] <= 0:
            continue
        v = t[:-1] * sd + mu
        path = v[:M * 3].reshape(M, 3)
        nj = int(np.clip(round(v[M * 3]), 2, 20))
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        tt = np.linspace(0, d[-1], nj)
        joints = np.stack([np.interp(tt, d, path[:, k]) for k in range(3)], 1)
        out.append(dict(path=path.tolist(), joints=joints.tolist(), n_joints=nj, stiffness=float(np.clip(v[M * 3 + 1], 0, 1))))
    return out


def chains_head_to_dataset(chains, k, b):
    """Decoded head-frame chains back to the dataset frame (in place) with a character's head fit (k, b)."""
    for c in chains:
        c['path'] = ((np.asarray(c['path']) - b) / k).tolist()
        c['joints'] = ((np.asarray(c['joints']) - b) / k).tolist()
    return chains


def load_chain_dit(ckpt_path, dev='cuda'):
    """(model, run args dict, token dim) of a chain DiT checkpoint (EMA weights)."""
    ck = load_ckpt(ckpt_path)
    ra = ck['args']
    model = ChainDiT(ck['dim'], ra['dim'], ra['depth'], ra['heads']).to(dev).eval()
    model.load_state_dict(ck['ema'])
    return model, ra, ck['dim']
