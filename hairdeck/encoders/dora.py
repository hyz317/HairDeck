"""Forward-pass re-implementation of the Dora-VAE 1.1 shape encoder (Seed3D/Dora, code and weights Apache-2.0),
loading the released weights unchanged (shape_model weights of dora_vae_1_1.ckpt re-saved as safetensors by
scripts/setup/convert_dora_ckpt.py). The same class also runs the Step1X-3D VAE encoder (same layout, qk-norm).

Why a re-implementation: the upstream encoder code paths contain modules adapted from Michelangelo (GPL-3.0; the
Fourier embedder and the attention blocks). This file was written from the checkpoint key layout and the architecture
described in the paper; it imports only torch and safetensors and copies no upstream file. Deterministic: farthest
point sampling always starts at index 0.

encode(coarse [B, N, 6], sharp [B, M, 6], t) -> pre-KL tokens [B, 2t, 768] (the encoder output after ln_post). The
released condition (condition_tokens: the training cache of scripts/data.py, and the test condition of scripts/test.py
pointcloud) uses t = 256 (512 tokens) with sharp = coarse: the mesh-based sharp-edge samples Dora expects cannot be
computed from a point cloud, and feeding the uniform points again reconstructed better than the point-cloud crease proxy
(common.sharp_proxy).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .. import paths
from .common import condition_plys, condition_points

DORA_WEIGHTS = ('dora_vae_1_1', 'dora_vae_1_1_shape_model.safetensors')
# the released checkpoint DORA_WEIGHTS is converted from: Hugging Face repository, revision, file, sha256
DORA_CKPT = ('Seed3D/Dora-VAE-1.1', '7187b01c78e1bd4a2ddf8301f145f5dc98ac2c22', 'dora_vae_1_1.ckpt',
             '13e1d66f690c9c3f8e2a584af0966060a06175aa63525a6e3f33d1f00ad9ce84')
COND_SHAPE = (512, 768)  # the released condition per hairstyle: tokens, width
BATCH = 16  # point clouds per forward pass of the released condition


def fourier_xyz(xyz, n_freq=8, with_pi=False):
    """[..., 3] -> [..., 3 + 6 * n_freq]: the raw coordinates, then sin, then cos of every coordinate times 2^k
    (k = 0..n_freq-1), coordinate-major (x f0..f7, y f0..f7, z f0..f7)."""
    f = 2.0 ** torch.arange(n_freq, device=xyz.device, dtype=torch.float32)
    if with_pi:
        f = f * math.pi
    ang = (xyz[..., :, None] * f.to(xyz.dtype)).flatten(-2)
    return torch.cat([xyz, ang.sin(), ang.cos()], -1)


@torch.no_grad()
def farthest_point_sample(xyz, k, start=0):
    """Greedy farthest point sampling. xyz [B, N, 3] -> indices [B, k] (int64); the first index is `start`, every next
    one maximises the squared distance to the already chosen set (first maximum on ties)."""
    B, N, _ = xyz.shape
    xyz = xyz.float()
    ar = torch.arange(B, device=xyz.device)
    out = torch.empty(B, k, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float('inf'), device=xyz.device)
    cur = torch.full((B,), start, dtype=torch.long, device=xyz.device)
    for i in range(k):
        out[:, i] = cur
        d = xyz - xyz[ar, cur][:, None]
        d = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]
        dist = torch.minimum(dist, d)
        cur = dist.argmax(-1)
    return out


def _fps_torch_cluster_half(xyz, k):
    """Step1X-3D's query selection: torch_cluster.fps(pos.half(), batch, ratio=k/N, random_start=False)."""
    try:
        import torch_cluster
    except ImportError:
        return farthest_point_sample(xyz.half().float(), k)
    B, N, _ = xyz.shape
    batch = torch.arange(B, device=xyz.device).repeat_interleave(N)
    idx = torch_cluster.fps(xyz.reshape(-1, 3).half(), batch, ratio=k / N, random_start=False)
    return idx.view(B, -1) - torch.arange(B, device=xyz.device)[:, None] * N


def gather_rows(x, idx):
    """x [B, N, C], idx [B, k] -> [B, k, C]."""
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))


def _lin(x, W, p):
    return F.linear(x, W[p + '.weight'], W.get(p + '.bias'))


def _ln(x, W, p, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), W[p + '.weight'], W[p + '.bias'], eps)


def _attend(q, k, v):
    """q [B, n, H, d], k/v [B, m, H, d] -> [B, n, H * d] (softmax(q k^T / sqrt(d)) v)."""
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return o.transpose(1, 2).flatten(2)


def _heads(x, H, parts):
    """Projected tensor [B, n, parts * C] whose channels are laid out per head as [part0 | part1 | ...] (each C / H wide)
    -> list of `parts` tensors [B, n, H, C / H]."""
    B, n, c = x.shape
    return list(x.view(B, n, H, c // H).chunk(parts, -1))


def _mlp(x, W, p):
    return _lin(F.gelu(_lin(x, W, p + '.c_fc')), W, p + '.c_proj')


def _self_block(x, W, p, H, qk_norm=False):
    """Pre-LN transformer block: x + attn(ln_1 x), then + mlp(ln_2 x). Fused qkv projection, per-head [q | k | v]."""
    q, k, v = _heads(_lin(_ln(x, W, p + '.ln_1'), W, p + '.attn.c_qkv'), H, 3)
    if qk_norm:
        q = _ln(q, W, p + '.attn.attention.q_norm', 1e-6)
        k = _ln(k, W, p + '.attn.attention.k_norm', 1e-6)
    x = x + _lin(_attend(q, k, v), W, p + '.attn.c_proj')
    return x + _mlp(_ln(x, W, p + '.ln_2'), W, p + '.mlp')


def _cross_block(x, data, W, p, H, qk_norm=False):
    """x + attn(ln_1 x -> q, ln_2 data -> [k | v] per head), then + mlp(ln_3 x)."""
    (q,) = _heads(_lin(_ln(x, W, p + '.ln_1'), W, p + '.attn.c_q'), H, 1)
    k, v = _heads(_lin(_ln(data, W, p + '.ln_2'), W, p + '.attn.c_kv'), H, 2)
    if qk_norm:
        q = _ln(q, W, p + '.attn.attention.q_norm', 1e-6)
        k = _ln(k, W, p + '.attn.attention.k_norm', 1e-6)
    x = x + _lin(_attend(q, k, v), W, p + '.attn.c_proj')
    return x + _mlp(_ln(x, W, p + '.ln_3'), W, p + '.mlp')


class ShapeVAE:
    """Dora-VAE 1.1 (and Step1X-3D VAE): point-query perceiver with a second ("sharp-edge") input branch.

    encode(coarse [B, N, 6], sharp [B, M, 6], t) -> pre-KL tokens [B, 2t, 768] where the first t query points are the
    FPS of the coarse points and the next t the FPS of the sharp points; both point sets are cross-attended by the full
    query set (one block each) and the two block outputs are summed, followed by 8 self-attention blocks and a
    LayerNorm. kl_mean(tokens) -> [B, 2t, 64]. sdf(latent_width, xyz) queries the decoder.
    """

    def __init__(self, kind, W, device='cuda', heads=12, qk_norm=False, with_pi=False, n_enc=8, n_dec=16):
        assert kind in ('dora', 'step1x3d')
        self.kind, self.H, self.qk_norm, self.with_pi = kind, heads, qk_norm, with_pi
        self.n_enc, self.n_dec = n_enc, n_dec
        self.W = {k: v.to(device=device, dtype=torch.float32) for k, v in W.items()}
        if kind == 'dora':
            self.k_in, self.k_in_sharp = 'encoder.input_proj', 'encoder.input_proj1'
            self.k_ca, self.k_ca_sharp = 'encoder.cross_attn', 'encoder.cross_attn1'
        else:
            self.k_in, self.k_in_sharp = 'encoder.input_proj', 'encoder.input_proj_sharp'
            self.k_ca, self.k_ca_sharp = 'encoder.cross_attn', 'encoder.cross_attn_sharp'

    def _embed(self, pts, proj):
        return _lin(torch.cat([fourier_xyz(pts[..., :3], 8, self.with_pi), pts[..., 3:]], -1), self.W, proj)

    def query_indices(self, coarse, sharp, t):
        """FPS indices, start 0. Dora: exact fp32 FPS (picks the same points as upstream's torch_cluster call).
        Step1X-3D runs torch_cluster.fps on float16 positions upstream; it is called the same way when installed."""
        if self.kind == 'step1x3d':
            return _fps_torch_cluster_half(coarse[..., :3], t), _fps_torch_cluster_half(sharp[..., :3], t)
        return farthest_point_sample(coarse[..., :3], t), farthest_point_sample(sharp[..., :3], t)

    @torch.no_grad()
    def encode(self, coarse, sharp, t, idx=None):
        W, H = self.W, self.H
        coarse, sharp = coarse.float(), sharp.float()
        dc, ds = self._embed(coarse, self.k_in), self._embed(sharp, self.k_in_sharp)
        ic, is_ = idx if idx is not None else self.query_indices(coarse, sharp, t)
        q = torch.cat([gather_rows(dc, ic), gather_rows(ds, is_)], 1)
        x = _cross_block(q, dc, W, self.k_ca, H, self.qk_norm) + _cross_block(q, ds, W, self.k_ca_sharp, H, self.qk_norm)
        for i in range(self.n_enc):
            x = _self_block(x, W, f'encoder.self_attn.resblocks.{i}', H, self.qk_norm)
        return _ln(x, W, 'encoder.ln_post')

    def kl_mean(self, tokens):
        return _lin(tokens.float(), self.W, 'pre_kl')[..., :64]

    @torch.no_grad()
    def latents_for_decoder(self, mean):
        x = _lin(mean.float(), self.W, 'post_kl')
        for i in range(self.n_dec):
            x = _self_block(x, self.W, f'transformer.resblocks.{i}', self.H, self.qk_norm)
        return x

    @torch.no_grad()
    def sdf(self, lat, xyz):
        """lat [B, T, 768] from latents_for_decoder, xyz [B, Q, 3] -> [B, Q] (TSDF-like field, surface at 0)."""
        q = _lin(fourier_xyz(xyz.float(), 8, self.with_pi), self.W, 'decoder.query_proj')
        x = _cross_block(q, lat, self.W, 'decoder.cross_attn_decoder', self.H, self.qk_norm)
        return _lin(_ln(x, self.W, 'decoder.ln_post'), self.W, 'decoder.output_proj')[..., 0]


def load_state(path):
    """Weights dict from a safetensors file (or a torch checkpoint), with the 'shape_model.' prefix stripped."""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location='cpu', weights_only=True)
        sd = sd.get('state_dict', sd)
    pre = 'shape_model.'
    if any(k.startswith(pre) for k in sd):
        sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    return sd


def build_dora(path=None, device='cuda'):
    """Dora-VAE 1.1 encoder from paths.weights('dora_vae_1_1/dora_vae_1_1_shape_model.safetensors') or `path`."""
    W = load_state(path or paths.weights(*DORA_WEIGHTS))
    return ShapeVAE('dora', W, device, qk_norm=any('q_norm' in k for k in W))


@torch.no_grad()
def encode_dora(model, x, t=512, sharp='copy'):
    """Condition tokens of point clouds x [B, N, 6] (xyz * 2, normals): t total tokens, t/2 per branch."""
    if sharp == 'proxy':
        from .common import sharp_proxy
        s = sharp_proxy(x)
    else:
        s = x
    return model.encode(x, s, t // 2)


@torch.no_grad()
def condition_tokens(split, out=None, device='cuda', workers=0, log=None, plys=None):
    """The released point-cloud condition of every row of a split: the Dora tokens of its condition point clouds
    (common.condition_plys and condition_points), BATCH clouds per forward pass in row order, both branches fed the same
    points, fp32 forward with TF32 off, stored as float16 [N, 512, 768]. out: an array of that shape to write into (the
    training cache is a memmap), else a new one is returned. workers > 1: read the clouds in that many processes (forked
    before the encoder is loaded). log(text): progress every 200 batches. plys: the point clouds instead of
    condition_plys(split) (scripts/data.py: hair_reform.ply of every row of both splits)."""
    plys = condition_plys(split) if plys is None else plys
    n = len(plys)
    out = np.zeros((n,) + COND_SHAPE, np.float16) if out is None else out
    pool = None
    if workers > 1:
        import multiprocessing
        pool = multiprocessing.get_context('fork').Pool(workers)
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        pts = pool.imap(condition_points, plys, chunksize=16) if pool else map(condition_points, plys)
        model = build_dora(device=device)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        for b, i in enumerate(range(0, n, BATCH)):
            x = np.stack([next(pts) for _ in range(min(BATCH, n - i))])
            out[i:i + len(x)] = encode_dora(model, torch.from_numpy(x).to(device), COND_SHAPE[0], 'copy').half().cpu().numpy()
            if log and b % 200 == 0:
                log(f'{i}/{n}')
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        if pool:
            pool.terminate()
    return out
