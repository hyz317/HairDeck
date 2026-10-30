"""Card-level set DiT.

Each hairstyle is a set of N_max card tokens (no positional embedding, so the model is permutation-equivariant). Each
token carries an existence flag; empty slots are zeros with flag -1. The condition is a set of context tokens read by
cross-attention in every block; their mean, projected, is added to the timestep embedding that drives adaLN.
The condition-specific subclasses (point cloud: models/pc_dit.py, image: models/imgmodel.py, cards for the chain DiT:
spring/chaindit.py) only replace encode_cond. Flow-matching loss and sampler: hairdeck/sampling/flow.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


class Attention(nn.Module):
    def __init__(self, dim, heads, ctx_dim=None):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim or dim, 2 * dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.qn, self.kn = RMSNorm(dim // heads), RMSNorm(dim // heads)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        B, N, C = x.shape
        M = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, C // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.o(o.transpose(1, 2).reshape(B, N, C))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, cross=True):
        super().__init__()
        self.cross = cross
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        if cross:
            self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.xattn = Attention(dim, heads)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate='tanh'), nn.Linear(h, dim))
        self.ada = nn.Linear(dim, (9 if cross else 6) * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    fp32_res = False  # keep the residual stream in fp32 under bf16 autocast (matmuls stay bf16)

    def forward(self, x, c, ctx):
        m = self.ada(F.silu(c)).chunk(9 if self.cross else 6, -1)
        cast = (lambda t: t.float()) if self.fp32_res else (lambda t: t)
        x = cast(x) + cast(m[2][:, None] * self.attn(modulate(self.n1(x), m[0], m[1])))
        if self.cross:
            x = x + cast(m[5][:, None] * self.xattn(modulate(self.n2(x), m[3], m[4]), ctx))
            x = x + cast(m[8][:, None] * self.mlp(modulate(self.n3(x), m[6], m[7])))
        else:
            x = x + cast(m[5][:, None] * self.mlp(modulate(self.n3(x), m[3], m[4])))
        return x


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class SetDiT(nn.Module):
    """Base set DiT. The base encode_cond expects a (B, n_ctx, cond_dim) cache whose row 0 holds a separate 768-d
    head vector (zero-padded); the released subclasses all override encode_cond and delete head_norm / head_proj."""

    def __init__(self, token_dim, dim=768, depth=12, heads=12, cond_dim=1536, head_dim=768, n_ctx=257, mlp_ratio=4.0, fp32_res=False):
        super().__init__()
        self.fp32_res = fp32_res
        self.dim = dim
        self.inp = nn.Linear(token_dim, dim)
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_norm = nn.LayerNorm(cond_dim)
        self.head_norm = nn.LayerNorm(head_dim)
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.head_proj = nn.Linear(head_dim, dim)
        self.pool_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null_ctx = nn.Parameter(torch.randn(n_ctx, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        for b in self.blocks:
            b.fp32_res = fp32_res
        self.fn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.f_ada = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, token_dim)
        for m in (self.f_ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_cond(self, cond):
        """cond: (B, n_ctx, cond_dim); row 0 = head vector (768, zero-padded)."""
        cond = cond.float()
        head = self.head_proj(self.head_norm(cond[:, 0, :768]))[:, None]
        lat = self.cond_proj(self.cond_norm(cond[:, 1:]))
        return torch.cat([head, lat], 1)

    def forward(self, x, t, ctx, drop=None):
        """ctx: encoded cond (B, n_ctx, dim). drop: (B,) bool -> use the learned null condition (CFG)."""
        if drop is not None:
            ctx = torch.where(drop[:, None, None], self.null_ctx[None].to(ctx.dtype).expand_as(ctx), ctx)
        c = self.t_mlp(timestep_embedding(t, 256)) + self.pool_mlp(ctx.mean(1))
        h = self.inp(x).float() if self.fp32_res else self.inp(x)
        for b in self.blocks:
            h = b(h, c, ctx)
        # output head in fp32: under bf16 autocast the per-dim output would carry ~0.4% quantization noise
        with torch.autocast('cuda', enabled=False):
            c, h = c.float(), h.float()
            sh, sc = self.f_ada(F.silu(c)).chunk(2, -1)
            return self.out(modulate(self.fn(h), sh, sc))
