"""Fast Euler/CFG sampling of the set DiTs (hairdeck.models.setdit.SetDiT and its subclasses): the operations of
hairdeck.sampling.flow.fm_sample / hairdeck.infer.image.euler_cfg with Euler steps, restructured for small batches on
GPUs whose small GEMMs and elementwise kernels are bound by memory bandwidth and launch overhead (AMD Strix Halo:
LPDDR5X shared by the CPU and the Radeon 8060S).

- The condition is fixed during sampling, so each block's cross-attention keys / values (after their RMSNorm) and the
  pooled condition embedding are computed once per call instead of once per step.
- The adaLN conditioning c = t_mlp(t) + pool_mlp(mean(ctx)) depends only on the step, so every block's modulation
  ada(silu(c)) and the output head's are computed for all steps in one GEMM each (M = steps * batch instead of a
  bandwidth-bound M = batch GEMM per block and step).
- Linear weights are read from a bf16 copy made once per model instead of autocast's per-call cast of each fp32 weight.
- One Euler step (the doubled CFG batch, the guidance combination and the update) is compiled by torch.compile:
  inductor fuses the norms, modulations, residuals and activations between the GEMMs into Triton kernels (on Strix
  Halo half of the eager step's time is those small kernels).

The tensor operations are the module's under the same autocast, so up to the batching of the precomputed GEMMs the
result is the original's; tools/rocm/check_fast.py compares them and the test-set metrics are unchanged
(docs/rocm.md).

HAIRDECK_FAST: 1 on, 0 off; default on for ROCm builds of PyTorch, off otherwise (the CUDA release keeps its own path).
HAIRDECK_COMPILE: 1 torch.compile the step, 0 eager; default as HAIRDECK_FAST. The first call per model and batch
size compiles (about a minute cold; seconds from the inductor cache on disk); a compilation error falls back to eager.
HAIRDECK_GRAPHS=1: also capture the step in a HIP / CUDA graph (default off: slower than plain launches on ROCm 7.2 in
our measurements; always off while PyTorch TunableOp is enabled, as hipBLASLt cannot tune or look up inside a capture).
"""
import os

import torch
import torch.nn.functional as F

from ..models.setdit import SetDiT, modulate, timestep_embedding


def enabled(model):
    want = os.environ.get('HAIRDECK_FAST')
    on = bool(torch.version.hip) if want is None else want == '1'
    # Only the base forward is restructured here; a subclass with its own forward keeps the generic path.
    return (on and isinstance(model, SetDiT) and type(model).forward is SetDiT.forward
            and next(model.parameters()).device.type == 'cuda' and torch.is_autocast_enabled('cuda')
            and torch.get_autocast_dtype('cuda') == torch.bfloat16)


def _graphs_on():
    tunable = torch.version.hip and torch.cuda.tunable.is_enabled()
    return os.environ.get('HAIRDECK_GRAPHS', '0') == '1' and not tunable


def _compile_on():
    want = os.environ.get('HAIRDECK_COMPILE')
    return bool(torch.version.hip) if want is None else want == '1'


_COMPILE_FAILED = []


def _compiled(fn):
    """torch.compile(fn), or fn itself after a compilation error (reported once)."""
    if _COMPILE_FAILED:
        return fn
    c = torch.compile(fn, dynamic=False, fullgraph=True)

    def call(*a):
        if _COMPILE_FAILED:
            return fn(*a)
        try:
            return c(*a)
        except Exception as e:  # noqa: BLE001  (inductor / Triton toolchain problems: keep sampling, eagerly)
            _COMPILE_FAILED.append(e)
            print(f'hairdeck.sampling.fast: torch.compile failed ({type(e).__name__}: {e}); sampling eagerly',
                  flush=True)
            return fn(*a)
    return call


def _bf16(m):
    return (m.weight.detach().to(torch.bfloat16), None if m.bias is None else m.bias.detach().to(torch.bfloat16))


class _Weights:
    """bf16 copies of a SetDiT's autocast Linear weights, kept with the module and remade when its parameters change
    (training steps, load_state_dict, .to())."""

    def __init__(self, model):
        self.key = self._key(model)
        mods = [m for b in model.blocks for m in b.modules() if isinstance(m, torch.nn.Linear)]
        mods += [m for m in (model.inp, *model.t_mlp, *model.pool_mlp) if isinstance(m, torch.nn.Linear)]
        self.w = {m: _bf16(m) for m in mods}

    @staticmethod
    def _key(model):
        ps = list(model.parameters())
        return tuple(p.data_ptr() for p in ps[:2]), sum(p._version for p in ps), ps[0].device

    def fresh(self, model):
        return self._key(model) == self.key


def _weights(model):
    w = getattr(model, '_hairdeck_fast_w', None)
    if w is None or not w.fresh(model):
        w = _Weights(model)
        object.__setattr__(model, '_hairdeck_fast_w', w)  # not a submodule / buffer: stays out of state_dict
    return w


class _Denoiser:
    def __init__(self, model, ctx, cfg, T):
        """T: (steps, B) the t of every step; the per-step terms are precomputed for all of them."""
        self.m = model
        self.W = _weights(model).w
        self.cfg = cfg
        B = ctx.shape[0]
        if cfg != 1.0:  # the doubled batch of fm_sample: rows B.. use the learned null context
            ctx = torch.cat([ctx, ctx])
            drop = torch.cat([torch.zeros(B, dtype=torch.bool, device=ctx.device),
                              torch.ones(B, dtype=torch.bool, device=ctx.device)])
            ctx = torch.where(drop[:, None, None], model.null_ctx[None].to(ctx.dtype).expand_as(ctx), ctx)
            T = torch.cat([T, T], 1)
        S, B2 = T.shape
        self.kv = []
        for b in model.blocks:
            if b.cross:
                a = b.xattn
                M, C = ctx.shape[1], model.dim
                k, v = self.lin(a.kv, ctx).view(B2, M, 2, a.h, C // a.h).permute(2, 0, 3, 1, 4)
                self.kv.append((a.kn(k), v))
            else:
                self.kv.append(None)
        # c for every step: (S, B2, dim), then every block's modulation (S, B2, n * dim) in one GEMM per block
        pooled = self._seq(model.pool_mlp, ctx.mean(1))
        c = self._seq(model.t_mlp, timestep_embedding(T.reshape(-1), 256)).view(S, B2, -1) + pooled
        sc = F.silu(c).reshape(S * B2, -1)
        self.ada = [self.lin(b.ada, sc).view(S, B2, -1) for b in model.blocks]
        with torch.autocast('cuda', enabled=False):
            self.f_ada = model.f_ada(F.silu(c.float()).reshape(S * B2, -1)).view(S, B2, -1)

    def lin(self, m, x):
        w, b = self.W[m]
        return F.linear(x, w, b)

    def _seq(self, seq, x):
        for m in seq:
            x = self.lin(m, x) if isinstance(m, torch.nn.Linear) else m(x)
        return x

    def _attn(self, a, x, kv=None):
        B, N, C = x.shape
        q = self.lin(a.q, x).view(B, N, a.h, C // a.h).transpose(1, 2)
        if kv is None:
            k, v = self.lin(a.kv, x).view(B, N, 2, a.h, C // a.h).permute(2, 0, 3, 1, 4)
            k = a.kn(k)
        else:
            k, v = kv
        o = F.scaled_dot_product_attention(a.qn(q), k, v)
        return self.lin(a.o, o.transpose(1, 2).reshape(B, N, C))

    def _block(self, b, x, ada, kv):
        m = ada.chunk(9 if b.cross else 6, -1)
        cast = (lambda t: t.float()) if b.fp32_res else (lambda t: t)
        x = cast(x) + cast(m[2][:, None] * self._attn(b.attn, modulate(b.n1(x), m[0], m[1])))
        if b.cross:
            x = x + cast(m[5][:, None] * self._attn(b.xattn, modulate(b.n2(x), m[3], m[4]), kv))
            x = x + cast(m[8][:, None] * self._seq(b.mlp, modulate(b.n3(x), m[6], m[7])))
        else:
            x = x + cast(m[5][:, None] * self._seq(b.mlp, modulate(b.n3(x), m[3], m[4])))
        return x

    def forward(self, x, ada, f_ada):
        """SetDiT.forward for the (doubled) batch at one step, with that step's precomputed modulations."""
        m = self.m
        h = self.lin(m.inp, x).float() if m.fp32_res else self.lin(m.inp, x)
        for b, a, kv in zip(m.blocks, ada, self.kv):
            h = self._block(b, h, a, kv)
        with torch.autocast('cuda', enabled=False):
            sh, sc = f_ada.chunk(2, -1)
            return m.out(modulate(m.fn(h.float()), sh, sc))

    def step(self, x, dt, ada, f_ada):
        """One Euler step of fm_sample: x + (tn - t) * v with v the CFG combination (float32)."""
        if self.cfg == 1.0:
            v = self.forward(x, ada, f_ada).float()
        else:
            vc, vu = self.forward(torch.cat([x, x]), ada, f_ada).float().chunk(2)
            v = vu + self.cfg * (vc - vu)
        return x + dt * v


def euler(model, ctx, x, steps, cfg):
    """Euler integration from the noise x (t = 1) to t = 0 in `steps` uniform steps with CFG `cfg`, as fm_sample."""
    B = x.shape[0]
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    # fm_sample's per-step scalars: t as torch.full((B,), float(t)), the update (tn - t) in float32
    T = ts[:-1, None].expand(steps, B).contiguous()
    DT = (ts[1:] - ts[:-1]).contiguous()
    d = _Denoiser(model, ctx, cfg, T)
    step = d.step
    if _compile_on():
        step = _compiled(d.step)
    if not _graphs_on():
        for i in range(steps):
            x = step(x, DT[i], [a[i] for a in d.ada], d.f_ada[i])
        return x
    # static buffers for the captured step: x, dt, and the step's modulations, copied in before each replay
    xs, dtb = x.clone(), DT[0].clone()
    ada = torch.stack([a for a in d.ada], 1)  # (S, blocks, B2, n * dim)
    adab, fb = ada[0].clone(), d.f_ada[0].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm-up outside capture (lazy init, kernel selection, compilation)
        for _ in range(2):
            step(xs, dtb, list(adab.unbind(0)), fb)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step(xs, dtb, list(adab.unbind(0)), fb)
    for i in range(steps):
        dtb.copy_(DT[i])
        adab.copy_(ada[i])
        fb.copy_(d.f_ada[i])
        g.replay()
        xs.copy_(out)
    return xs.clone()
