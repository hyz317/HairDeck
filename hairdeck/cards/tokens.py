"""Generated card tokens -> cards.

LatentDecoder turns the decoded token fields (root, latent z, existence) into card polylines with the card VAE;
tokens_to_cards assembles cards (n, 8) = xyz, width, thickness, unit root width direction (the predicted normal
token), which mesh.py meshes with the Bishop frame. clean_tokens puts generated tokens into the training-time format
(used when one model's output conditions another, e.g. card tokens -> chain DiT); sign_token_normals also gives their
root normals the sign rule of the training tokens.
"""
import numpy as np
import torch

from .codec import apply_root_normal, sign_root_normals
from .vae import load_vae


class LatentDecoder:
    def __init__(self, vae_dir, device='cuda'):
        self.vae = load_vae(vae_dir, device)

    @torch.no_grad()
    def decode_tokens(self, d, fix=True):
        """d = tokenizer.decode(x) -> same dict plus d['seq'][b] = {slot: (n,5) card}. fix: the root-normal fix of
        CHARM's 5-column cards (codec.apply_root_normal); cards with a root direction (tokens_to_cards) use fix=False."""
        d = dict(d)
        d['seq'] = []
        for b in range(d['exist'].shape[0]):
            js = torch.nonzero(d['exist'][b])[:, 0]
            seqs = self.vae.decode_cards(d['z'][b, js].float(), d['root'][b, js].cpu().numpy().astype(np.float64)) if len(js) else []
            out = {}
            for j, s in zip(js.tolist(), seqs):
                out[j] = apply_root_normal(s, d['nrm'][b, j].cpu().numpy().astype(np.float64)) if fix else s
            d['seq'].append(out)
        return d


def tokens_to_cards(tok, x, vae, root_dir=True, fix=False):
    """x: (B,N,D) tokens -> list (per hairstyle) of list of (seq, shape_id). root_dir: seq (n,8) with the predicted unit
    root normal appended to every row. (root_dir=False returns the (n,5) VAE output, with the root-normal fix of CHARM's
    5-column cards if fix=True.)"""
    d = tok.decode(x.float())
    d = vae.decode_tokens(d, fix=fix and not root_dir)
    out = []
    for b in range(x.shape[0]):
        cards = []
        ex = d['exist'][b].cpu().numpy()
        for j in np.nonzero(ex)[0]:
            nrm = d['nrm'][b, j].cpu().numpy().astype(np.float64)
            seq = d['seq'][b][j]
            if root_dir:
                nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
                seq = np.concatenate([seq[:, :5], np.repeat(nrm[None], len(seq), 0)], 1)
            cards.append((seq, int(d['shape'][b, j])))
        out.append(cards)
    return out


def cards_head_to_dataset(cards, k, b):
    """Head-frame cards (p_head = k * p + b) back to the dataset frame; width and thickness scale by 1 / k."""
    back = []
    for seq, sh in cards:
        seq = np.array(seq, np.float64)
        seq[:, :3] = (seq[:, :3] - b) / k
        seq[:, 3:5] /= k
        back.append((seq, sh))
    return back


def clean_tokens(tok, x):
    """Generated tokens in the training-time format: slots with existence <= 0 zeroed, existence and card class
    snapped to +-1, root normal unit length."""
    g = tok.groups
    ex = x[..., g['exist']][..., 0] > 0
    y = torch.where(ex[..., None], x, torch.zeros_like(x))
    n = y[..., g['nrm']]
    y[..., g['nrm']] = torch.where(ex[..., None], n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6), n)
    y[..., g['cls']] = torch.where(ex[..., None], torch.sign(y[..., g['cls']]) + (y[..., g['cls']] == 0).float(), y[..., g['cls']])
    y[..., g['exist']] = torch.where(ex, 1.0, -1.0)[..., None]
    return y


def sign_token_normals(tok, x):
    """Cleaned card tokens x (clean_tokens) with the root normal of every existing card signed by the rule of the
    training tokens (codec.sign_root_normals on the roots decoded from the tokens)."""
    g = tok.groups
    y = x.clone()
    for b in range(len(y)):
        ex = y[b, :, g['exist']][:, 0] > 0
        if not ex.any():
            continue
        rows = y[b, ex]
        roots = tok.decode(rows)['root'].cpu().numpy().astype(np.float64)
        nrm, _ = sign_root_normals(rows[:, g['nrm']].cpu().numpy(), roots)
        rows[:, g['nrm']] = torch.from_numpy(nrm).to(rows.device)
        y[b, ex] = rows
    return y
