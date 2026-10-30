"""Rectified-flow training loss and sampler shared by the card DiTs and the chain DiT.

x_t = (1 - t) x1 + t eps (t = 0 data, t = 1 noise); the model predicts v = eps - x1. Classifier-free guidance: the
condition is replaced by the model's learned null context with probability p_drop in training; at sampling time
v = v_uncond + cfg * (v_cond - v_uncond), both evaluated in one doubled batch.
"""
import torch


def fm_loss(model, x1, ctx, wvec, p_drop=0.1, t_mode='logitnorm'):
    """Weighted MSE flow-matching loss; t ~ logit-normal (default) or uniform. wvec: per-token-dim weights."""
    B = x1.shape[0]
    if t_mode == 'logitnorm':
        t = torch.sigmoid(torch.randn(B, device=x1.device))
    else:
        t = torch.rand(B, device=x1.device)
    eps = torch.randn_like(x1)
    xt = (1 - t[:, None, None]) * x1 + t[:, None, None] * eps
    drop = torch.rand(B, device=x1.device) < p_drop
    v = model(xt, t, ctx, drop)
    per = ((v.float() - (eps - x1)) ** 2) * wvec
    return per.mean()


@torch.no_grad()
def fm_sample(model, ctx, n_tokens, token_dim, steps=50, cfg=1.0, shift=1.0, generator=None, sampler='euler'):
    """Integrate from noise (t = 1) to data (t = 0) in `steps` uniform steps (optionally time-shifted); Euler or Heun."""
    B = ctx.shape[0]
    x = torch.randn(B, n_tokens, token_dim, device=ctx.device, generator=generator)
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    if shift != 1.0:
        ts = shift * ts / (1 + (shift - 1) * ts)

    def vel(x, t):
        tt = torch.full((B,), float(t), device=ctx.device)
        if cfg == 1.0:
            return model(x, tt, ctx).float()
        both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([ctx, ctx]),
                     torch.cat([torch.zeros(B, dtype=torch.bool, device=ctx.device), torch.ones(B, dtype=torch.bool, device=ctx.device)])).float()
        vc, vu = both.chunk(2)
        return vu + cfg * (vc - vu)

    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        v = vel(x, t)
        if sampler == 'heun' and tn > 0:
            x2 = x + (tn - t) * v
            v2 = vel(x2, tn)
            x = x + (tn - t) * 0.5 * (v + v2)
        else:
            x = x + (tn - t) * v
    return x
