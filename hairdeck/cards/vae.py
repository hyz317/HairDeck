"""Card VAE: variable-length card -> latent -> continuous curve decoder.

Encoder: transformer over the original control points (root-relative xyz, w, t, arc-length fraction u), padding masked.
Decoder: queries are positions along the card given by their arc-length fraction u in [0,1] (Fourier features), plus
the latent; a transformer (here the adaLN-modulated AdaDecoder, --dec ada) over the queries outputs (rel xyz, w, t) at
each u. The output head runs in fp32. The point count n comes from a classification head on z.
Training queries the decoder at uniform u_j = j/(n-1) with linearly interpolated targets (--uniform_target), exactly
what decoding asks for. Training driver: scripts/train/train_vae.py.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import paths
from ..ckpt import load as load_ckpt, vae_file
from .codec import resample
from .packed import Packed

BUCKETS = [16, 24, 32, 48, 64, 96, 128]  # length buckets for training batches


def _amp(enabled):
    """bf16 autocast on CUDA (a no-op for CPU tensors, as in the original code)."""
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(enabled) and torch.cuda.is_available())


# ---------------------------------------------------------------- data helpers
def pad_seqs(seqs, L):
    """Cards (n, >=5) -> (C,L,5) float32 root-relative features (xyz minus the root, w, t), (C,) lengths. Cards longer
    than L are arc-length resampled to L points first; the root is subtracted in the cards' own precision (float32
    templates, float64 generated cards)."""
    out = np.zeros((len(seqs), L, 5), np.float32)
    lens = np.zeros(len(seqs), np.int64)
    for j, s in enumerate(seqs):
        s = s[:, :5]
        if len(s) > L:
            s = resample(s.astype(np.float64), L).astype(s.dtype)
        out[j, :len(s)] = s
        out[j, :len(s), :3] -= s[0, :3]
        lens[j] = len(s)
    return out, lens


def pad_cards(P, L, idx=None):
    """Packed templates -> (C,L,5) float32 root-relative features, (C,) lengths (pad_seqs of the cards idx, default
    all)."""
    idx = np.arange(len(P.card_ptr) - 1) if idx is None else idx
    return pad_seqs([P.points[P.card_ptr[c]:P.card_ptr[c + 1]] for c in idx], L)


def build_pad_cache(L=128):
    """data/cards_train_pad{L}.npz from data/train_templates.npz (root-relative, padded); returns its path."""
    out = paths.data(f'cards_train_pad{L}.npz')
    if not os.path.isfile(out):
        X, lens = pad_cards(Packed(paths.data('train_templates.npz')), L)
        np.savez(out, X=X, lens=lens)
    return out


def arc_u(X, lens):
    """X (C,L,5) root-relative -> (C,L) arc-length fraction of each point (0 at root, 1 at tip; 0 for padding)."""
    seg = np.linalg.norm(np.diff(X[:, :, :3], axis=1), axis=-1)
    L = X.shape[1]
    valid_seg = np.arange(1, L)[None] < lens[:, None]
    seg = seg * valid_seg
    s = np.concatenate([np.zeros((len(X), 1), np.float32), np.cumsum(seg, 1)], 1)
    tot = np.maximum(s[np.arange(len(X)), lens - 1], 1e-9)
    u = s / tot[:, None]
    u[~(np.arange(L)[None] < lens[:, None])] = 0
    return u.astype(np.float32)


def uniform_targets(X, lens, U, chunk=20000):
    """Resample every card (padded X, arc fractions U) to its own n points at uniform u_j = j/(n-1) by linear
    interpolation along the original polyline. This is exactly what the decoder is asked for at inference."""
    out = np.zeros_like(X)
    for n in np.unique(lens):
        idx = np.nonzero(lens == n)[0]
        uj = np.linspace(0, 1, n).astype(np.float32)
        for s in range(0, len(idx), chunk):
            b = idx[s:s + chunk]
            Ub, Xb = U[b, :n], X[b, :n]
            k = np.clip((Ub[:, :, None] <= uj[None, None, :]).sum(1) - 1, 0, max(n - 2, 0))  # (B, n) segment index
            u0 = np.take_along_axis(Ub, k, 1)
            u1 = np.take_along_axis(Ub, np.minimum(k + 1, n - 1), 1)
            f = np.clip((uj[None] - u0) / np.maximum(u1 - u0, 1e-12), 0, 1)[..., None]
            x0 = np.take_along_axis(Xb, k[..., None], 1)
            x1 = np.take_along_axis(Xb, np.minimum(k + 1, n - 1)[..., None], 1)
            out[b, :n] = x0 * (1 - f) + x1 * f
    return out


class Norm:
    def __init__(self, stats, device):
        self.mu = torch.tensor(stats['mu'], device=device, dtype=torch.float32)
        self.sd = torch.tensor(stats['sd'], device=device, dtype=torch.float32)

    def fwd(self, x):
        return (x - self.mu) / self.sd

    def inv(self, x):
        return x * self.sd + self.mu


# ---------------------------------------------------------------- model
class Fourier(nn.Module):
    def __init__(self, nf=16):
        super().__init__()
        self.register_buffer('f', (2.0 ** torch.arange(nf)) * math.pi, persistent=False)

    def forward(self, u):
        a = u[..., None] * self.f
        return torch.cat([u[..., None], torch.sin(a), torch.cos(a)], -1)


class ResMLP(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim)) for _ in range(depth)])

    def forward(self, h, src_key_padding_mask=None):
        for l in self.layers:
            h = h + l(h)
        return h


class AdaBlock(nn.Module):
    """Pre-LN transformer block whose LayerNorms are modulated by the latent (adaLN-Zero, as in DiT)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, keep):
        s1, sc1, g1, s2, sc2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x) * (1 + sc1) + s1).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        x = x + g1 * self.o(a.transpose(1, 2).reshape(B, N, D))
        return x + g2 * self.mlp(self.n2(x) * (1 + sc2) + s2)


class AdaDecoder(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        self.blocks = nn.ModuleList([AdaBlock(dim, heads) for _ in range(depth)])

    def forward(self, h, c, keep):
        for b in self.blocks:
            h = b(h, c, keep)
        return h


class CardVAE(nn.Module):
    def __init__(self, L=128, dim=256, depth=4, heads=8, zdim=64, nf=16, dec='tf', integ=False):
        super().__init__()
        self.integ = integ  # xyz channels are dp/du; positions = integral from the root (p(0) = 0 exactly)
        self.L, self.zdim = L, zdim
        self.four = Fourier(nf)
        fd = 1 + 2 * nf
        self.inp = nn.Linear(5 + fd, dim)
        self.pe_enc = nn.Parameter(torch.randn(L + 1, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        mk = lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.0, 'gelu', batch_first=True, norm_first=True), depth)
        self.enc = mk()
        self.to_z = nn.Linear(dim, 2 * zdim)
        self.from_z = nn.Sequential(nn.Linear(zdim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.q_in = nn.Linear(fd, dim)
        self.dec_type = dec
        self.dec = mk() if dec == 'tf' else ResMLP(dim, 2 * depth) if dec == 'mlp' else AdaDecoder(dim, depth, heads)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 5)
        self.n_head = nn.Sequential(nn.Linear(zdim, dim), nn.SiLU(), nn.Linear(dim, L))

    def encode(self, x, u, mask):
        B, Lb = x.shape[:2]
        h = self.inp(torch.cat([x, self.four(u)], -1))
        h = torch.cat([self.cls.expand(B, -1, -1), h], 1) + self.pe_enc[None, :Lb + 1]
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.enc(h, src_key_padding_mask=kpm)
        mu, logvar = self.to_z(h[:, 0]).chunk(2, -1)
        return mu, logvar.clamp(-12, 6)

    def decode(self, z, u, mask):
        c = self.from_z(z)
        h = self.q_in(self.four(u)) + c[:, None]
        h = self.dec(h, c, mask) if self.dec_type == 'ada' else self.dec(h, src_key_padding_mask=~mask)
        with torch.autocast('cuda', enabled=False):
            o = self.out(self.norm(h.float()))
            if not self.integ:
                return o
            # queries are sorted by u along each card (true u in training, uniform at decode); trapezoid rule
            du = torch.diff(u.float(), dim=1, prepend=u[:, :1].float()) * mask
            d = o[..., :3] * mask[..., None]
            seg = 0.5 * (d + torch.cat([d[:, :1], d[:, :-1]], 1)) * du[..., None]
            return torch.cat([torch.cumsum(seg, 1), o[..., 3:]], -1)


class LoadedVAE:
    """A trained card VAE (ckpt directory with model.safetensors or model.pt + stats.json): encode padded cards, decode
    latents to cards."""

    def __init__(self, vae_dir, device='cuda'):
        ck = load_ckpt(vae_file(vae_dir))
        a = ck['args']
        self.model = CardVAE(a['L'], a['dim'], a['depth'], 8, a['zdim'], a.get('nf', 16), a.get('dec', 'tf'), a.get('integ', False)).to(device).eval()
        self.model.load_state_dict(ck['model'])
        self.norm = Norm(json.load(open(f'{vae_dir}/stats.json')), device)
        self.L, self.zdim, self.device = a['L'], a['zdim'], device
        self.bf16 = not a.get('fp32', False)
        if not self.bf16:
            torch.backends.cuda.matmul.allow_tf32 = False

    @torch.no_grad()
    def encode_padded(self, X, lens, bs=8192):
        """X (C,L,5) root-relative padded cards, lens (C,) -> posterior means (C, zdim) float32."""
        U = arc_u(X, lens)
        out = np.zeros((len(X), self.zdim), np.float32)
        ar = torch.arange(self.L, device=self.device)
        for s in range(0, len(X), bs):
            x = torch.from_numpy(X[s:s + bs]).to(self.device)
            u = torch.from_numpy(U[s:s + bs]).to(self.device)
            n = torch.from_numpy(lens[s:s + bs]).to(self.device)
            mask = ar[None] < n[:, None]
            with _amp(self.bf16):
                mu, _ = self.model.encode(self.norm.fwd(x) * mask[..., None], u, mask)
            out[s:s + bs] = mu.float().cpu().numpy()
        return out

    @torch.no_grad()
    def decode_cards(self, z, root):
        """z (B, zdim) torch, root (B, 3) numpy -> list of (n, 5) float64 cards in world coordinates."""
        z = z.float()
        ns = (self.model.n_head(z).argmax(-1) + 1).clamp(min=3)
        ar = torch.arange(self.L, device=z.device)
        mask = ar[None] < ns[:, None]
        u = (ar[None].float() / (ns[:, None].float() - 1).clamp(min=1)).clamp(max=1.0) * mask
        with _amp(self.bf16):
            feat = self.model.decode(z, u, mask)
        feat = self.norm.inv(feat.float()).cpu().numpy().astype(np.float64)
        ns = ns.cpu().numpy()
        out = []
        for b in range(z.shape[0]):
            s = feat[b, :ns[b]].copy()
            s[:, :3] += root[b]
            s[:, 3:] = np.clip(s[:, 3:], 0, None)
            out.append(s)
        return out


def load_vae(vae_dir, device='cuda'):
    """Load a card VAE checkpoint directory (model.safetensors or model.pt with 'model' and 'args', stats.json)."""
    return LoadedVAE(vae_dir, device)
