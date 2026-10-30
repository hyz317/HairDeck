"""Point-cloud conditioned set DiT: the condition is a cached token set [B, T, C] from a frozen point-cloud encoder
(the released model: Dora-VAE 1.1 pre-KL tokens, T = 512, C = 768, data/cond_dora_{split}.npy), LayerNorm + Linear to
the DiT width. Training / test drivers: scripts/train/train_pc_dit.py, scripts/test.py pointcloud.
"""
import argparse

import numpy as np

from .. import paths
from ..ckpt import load as load_ckpt
from .setdit import SetDiT
from .tokenizer import build_tokenizer


def cond_path(cond, split):
    """Condition cache of encoder `cond` ('dora' for the released model) for a split."""
    return paths.data(f'cond_{cond}_{split}.npy')


class GenericCondDiT(SetDiT):
    """Condition = a cached token set [B, T, C] from a frozen encoder."""

    def __init__(self, token_dim, dim, depth, heads, T, C):
        super().__init__(token_dim, dim, depth, heads, cond_dim=C, n_ctx=T)
        del self.head_norm, self.head_proj  # no separate head row

    def encode_cond(self, cond):
        return self.cond_proj(self.cond_norm(cond.float()))


PcDiT = GenericCondDiT


def build_model(a, tok_dim, T=None, C=None):
    """Model for run args `a` (cond, dim, depth, heads). T, C default to the shape of the test condition cache."""
    if a.cond in ('miche', 'pointtok'):
        raise ValueError(f'cond {a.cond!r}: the released point-cloud model takes the Dora condition (cond dora)')
    if T is None or C is None:
        T, C = np.load(cond_path(a.cond, 'test'), mmap_mode='r').shape[1:]
    return GenericCondDiT(tok_dim, a.dim, a.depth, a.heads, T, C)


def load_pc_dit(ckpt_path, dev='cuda'):
    """(model, tokenizer, run args) of a point-cloud card DiT checkpoint (EMA weights)."""
    ck = load_ckpt(ckpt_path)
    ra = argparse.Namespace(**ck['args'])
    tok = build_tokenizer(ra.vae, device=dev)
    T, C = ck['ema']['null_ctx'].shape[0], ck['ema']['cond_proj.weight'].shape[1]  # condition shape from the weights
    model = build_model(ra, tok.dim, T, C).to(dev).eval()
    model.load_state_dict(ck['ema'])
    return model, tok, ra
