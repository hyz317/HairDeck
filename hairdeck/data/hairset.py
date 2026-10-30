"""Training data of the set DiTs: per-hairstyle card sets and their condition rows, assembled on the GPU.

Files (data root): cards_{split}.npz from scripts/data.py (res (C,K,5), nrm (C,3), n, shape, hair_ptr,
cond_idx = row of the hairstyle in the condition cache); card latents from scripts/train/export_latents.py,
ckpt/<vae>/latents_{split}.npy (point-cloud model) or the head-frame ckpt/<vae>/latents_{split}_head.npy (image
model); condition caches (e.g. cond_dora_{split}.npy, scripts/data.py).
"""
import time

import numpy as np
import torch

from .. import paths


class HairSet:
    """One item = one hairstyle: padded card features (N_max slots, random slot order in training) + its condition row."""

    def __init__(self, split, nmax, latents=None, train=True, cond_path=None):
        z = np.load(paths.data(f'cards_{split}.npz'))
        self.res, self.nrm, self.n, self.shape, self.hp = z['res'], z['nrm'], z['n'], z['shape'], z['hair_ptr']
        self.cidx = z['cond_idx'] if 'cond_idx' in z.files else np.arange(len(self.hp) - 1)  # hairstyle -> row of the cond cache
        self.lat = latents
        self.nmax = nmax
        nc = np.diff(self.hp)
        self.items = np.nonzero(nc <= nmax)[0] if train else np.arange(len(nc))
        self.cond_path = cond_path
        self.cond = None
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        if self.cond is None:
            self.cond = np.load(self.cond_path, mmap_mode='r')
        i = self.items[j]
        a, b = self.hp[i], min(self.hp[i + 1], self.hp[i] + self.nmax)
        m = b - a
        perm = np.random.permutation(self.nmax)[:m] if self.train else np.arange(m)
        K = self.res.shape[1]
        res = np.zeros((self.nmax, K, 5), np.float32)
        nrm = np.zeros((self.nmax, 3), np.float32)
        n = np.full((self.nmax,), 3, np.int64)
        sh = np.zeros((self.nmax,), np.int64)
        mask = np.zeros((self.nmax,), bool)
        res[perm], nrm[perm], n[perm], sh[perm], mask[perm] = self.res[a:b], self.nrm[a:b], self.n[a:b], self.shape[a:b], True
        out = dict(res=res, nrm=nrm, n=n, shape=sh, mask=mask, idx=i)
        if self.lat is not None:
            lat = np.zeros((self.nmax, self.lat.shape[1]), np.float32)
            lat[perm] = self.lat[a:b]
            out['lat'] = lat
        out['cond'] = np.asarray(self.cond[self.cidx[i]])
        return out

    def load_cond_ram(self):
        t0 = time.time()
        self.cond = np.load(self.cond_path)
        return time.time() - t0


class GPUBatcher:
    """All batch assembly in the main thread: card features live on the GPU; random slot assignment and the card gather
    are torch ops on the GPU; only the condition rows are gathered from RAM with index_select."""

    def __init__(self, ds, bs, dev='cuda', seed=0):
        self.bs, self.dev, self.nmax = bs, dev, ds.nmax
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.items = torch.from_numpy(ds.items).to(dev)
        self.hp = torch.from_numpy(ds.hp).to(dev)
        self.res = torch.from_numpy(ds.res).to(dev) if ds.lat is None else torch.from_numpy(np.ascontiguousarray(ds.res[:, :1])).to(dev)
        self.nrm = torch.from_numpy(ds.nrm).to(dev)
        self.n = torch.from_numpy(ds.n.astype(np.int64)).to(dev)
        self.shape = torch.from_numpy(ds.shape.astype(np.int64)).to(dev)
        self.lat = torch.from_numpy(ds.lat).to(dev) if ds.lat is not None else None
        self.cond = torch.from_numpy(ds.cond)  # (H, T, C) fp16 in RAM
        self.cidx = torch.from_numpy(ds.cidx)

    def __iter__(self):
        B, N, dev = self.bs, self.nmax, self.dev
        ar = torch.arange(N, device=dev)
        while True:
            js = torch.randint(0, len(self.items), (B,), device=dev, generator=self.g)
            hid = self.items[js]
            a = self.hp[hid]
            nc = torch.minimum(self.hp[hid + 1] - a, torch.tensor(N, device=dev))
            perm = torch.argsort(torch.rand(B, N, device=dev, generator=self.g), 1)  # perm[b,k] = slot of card k
            val = torch.where(ar[None] < nc[:, None], a[:, None] + ar[None], torch.full_like(perm, -1))
            cid = torch.full((B, N), -1, dtype=torch.long, device=dev).scatter_(1, perm, val)
            mask = cid >= 0
            safe = cid.clamp(min=0)
            out = dict(res=self.res[safe] * mask[..., None, None], nrm=self.nrm[safe] * mask[..., None],
                       n=torch.where(mask, self.n[safe], torch.full_like(safe, 3)), shape=self.shape[safe] * mask, mask=mask, idx=hid)
            if self.lat is not None:
                out['lat'] = self.lat[safe] * mask[..., None]
            out['cond'] = torch.index_select(self.cond, 0, self.cidx[hid.cpu()]).to(dev)
            yield out


def to_tokens(tok, batch, dev):
    return tok.encode(batch['res'].to(dev), batch['nrm'].to(dev), batch['n'].to(dev), batch['shape'].to(dev), batch['mask'].to(dev), batch['lat'].to(dev))


class TargetGather:
    """Card targets of given hairstyles gathered on the GPU (same slot permutation scheme as GPUBatcher), in the head
    frame; used by the image DiT, whose batches come from an image loader that only yields hairstyle ids."""

    def __init__(self, ds, head, dev='cuda', seed=0):
        """head: (k, b) per hairstyle (data/head_frame_{split}.npz); roots become k * root + b. The card latents of ds
        are the head-frame ones (each card scaled by its k before encoding)."""
        self.dev, self.nmax = dev, ds.nmax
        self.k = torch.from_numpy(head[0]).float().to(dev)
        self.b = torch.from_numpy(head[1]).float().to(dev)
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.hp = torch.from_numpy(ds.hp).to(dev)
        self.res = torch.from_numpy(np.ascontiguousarray(ds.res[:, :1])).to(dev)
        self.nrm = torch.from_numpy(ds.nrm).to(dev)
        self.n = torch.from_numpy(ds.n.astype(np.int64)).to(dev)
        self.shape = torch.from_numpy(ds.shape.astype(np.int64)).to(dev)
        self.lat = torch.from_numpy(ds.lat).to(dev)

    def __call__(self, hid, shuffle=True):
        B, N, dev = len(hid), self.nmax, self.dev
        ar = torch.arange(N, device=dev)
        a = self.hp[hid]
        nc = torch.minimum(self.hp[hid + 1] - a, torch.tensor(N, device=dev))
        perm = torch.argsort(torch.rand(B, N, device=dev, generator=self.g), 1) if shuffle else ar[None].expand(B, N)
        val = torch.where(ar[None] < nc[:, None], a[:, None] + ar[None], torch.full((B, N), -1, device=dev))
        cid = torch.full((B, N), -1, dtype=torch.long, device=dev).scatter_(1, perm.contiguous(), val)
        mask = cid >= 0
        safe = cid.clamp(min=0)
        res = self.res[safe]
        res = torch.cat([self.k[hid][:, None, None, None] * res[..., :3] + self.b[hid][:, None, None, :], res[..., 3:]], -1)
        return dict(res=res * mask[..., None, None], nrm=self.nrm[safe] * mask[..., None],
                    n=torch.where(mask, self.n[safe], torch.full_like(safe, 3)), shape=self.shape[safe] * mask, mask=mask,
                    lat=self.lat[safe] * mask[..., None])
