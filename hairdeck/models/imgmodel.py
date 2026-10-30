"""Image-conditioned set DiT: frozen DINOv3 image tokens as the cross-attention context.

DinoEncoder builds DINOv3 ViT-L/16 with the model code of the official repository (torch.hub, a local clone at
paths.dinov3_repo()) and loads the weights of the Hugging Face repository facebook/dinov3-vitl16-pretrain-lvd1689m
(paths.weights('dinov3', 'model.safetensors'), in the transformers layout; hub_state_dict maps them). DINOv3 is released
under the DINOv3 License (not included here); the released image DiT checkpoints contain no DINOv3 weights.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .. import paths
from ..ckpt import load as load_ckpt
from .setdit import SetDiT
from .tokenizer import build_tokenizer

DINO = {'vitl16': ('dinov3_vitl16', 1024, 16)}  # torch.hub entry, width, attention heads
DINO_WEIGHTS = 'model.safetensors'


def dino_tokens(name, res):
    """Number of context tokens and their width for a DINOv3 variant at input resolution res: CLS + 4 storage tokens +
    (res / 16)^2 patch tokens."""
    return 1 + 4 + (res // 16) ** 2, DINO[name][1]


def hub_state_dict(sd, depth, heads):
    """State dict of the transformers DINOv3ViTModel -> that of the torch.hub model. The transformers file has separate
    q / k / v projections without a k bias, and leaves out two buffers; they get the values of Meta's original
    checkpoint, the weights the image models were trained with: the rope periods rounded to bfloat16 and a zero
    qkv bias_mask (the q and v biases of the file are zero too)."""
    out = {'cls_token': sd['embeddings.cls_token'], 'storage_tokens': sd['embeddings.register_tokens'],
           'mask_token': sd['embeddings.mask_token'].reshape(1, -1),
           'patch_embed.proj.weight': sd['embeddings.patch_embeddings.weight'],
           'patch_embed.proj.bias': sd['embeddings.patch_embeddings.bias'],
           'norm.weight': sd['norm.weight'], 'norm.bias': sd['norm.bias']}
    names = {'norm1': 'norm1', 'norm2': 'norm2', 'attention.o_proj': 'attn.proj', 'mlp.up_proj': 'mlp.fc1',
             'mlp.down_proj': 'mlp.fc2'}
    for i in range(depth):
        s, t = f'layer.{i}.', f'blocks.{i}.'
        for a, b in names.items():
            for p in ('weight', 'bias'):
                out[f'{t}{b}.{p}'] = sd[f'{s}{a}.{p}']
        out[f'{t}ls1.gamma'] = sd[f'{s}layer_scale1.lambda1']
        out[f'{t}ls2.gamma'] = sd[f'{s}layer_scale2.lambda1']
        q, k, v = (f'{s}attention.{x}_proj' for x in 'qkv')
        out[f'{t}attn.qkv.weight'] = torch.cat([sd[f'{q}.weight'], sd[f'{k}.weight'], sd[f'{v}.weight']])
        out[f'{t}attn.qkv.bias'] = torch.cat([sd[f'{q}.bias'], torch.zeros_like(sd[f'{q}.bias']), sd[f'{v}.bias']])
        out[f'{t}attn.qkv.bias_mask'] = torch.zeros_like(out[f'{t}attn.qkv.bias'])
    d = sd['norm.weight'].numel() // heads  # head width; rope base 100 as in the torch.hub ViT-L/16
    out['rope_embed.periods'] = (100.0 ** (2 * torch.arange(d // 4, dtype=torch.float32) / (d // 2))).to(torch.bfloat16).float()
    return out


class DinoEncoder(nn.Module):
    """uint8 (B, 3, H, W) -> tokens (B, 1 + 4 + (res/16)^2, C): CLS, 4 storage tokens, patch tokens (all normed)."""

    def __init__(self, name='vitl16', res=512, repo=None, weights=None):
        super().__init__()
        hub, self.dim, heads = DINO[name]
        with torch.random.fork_rng(devices=[]):  # layers that ignore device= are initialised on the CPU
            net = torch.hub.load(repo or paths.dinov3_repo(), hub, source='local', pretrained=False, device='meta')
        sd = hub_state_dict(load_file(weights or paths.weights('dinov3', DINO_WEIGHTS)), len(net.blocks), heads)
        # tensors of their own: views into the file's read buffer gave other float32 results on the CPU
        net.load_state_dict({k: v.clone() for k, v in sd.items()}, strict=True, assign=True)
        self.net = net.eval().requires_grad_(False)
        self.res = res
        self.n_tokens = 1 + 4 + (res // 16) ** 2
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):  # always frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, img):
        x = img.float() / 255
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=self.res, mode='bilinear', antialias=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = self.net.forward_features((x - self.mean) / self.std)
        # DINOv3 already applies the final norm to CLS, storage and patch tokens in forward_features
        out = torch.cat([o['x_norm_clstoken'][:, None], o['x_storage_tokens'], o['x_norm_patchtokens']], 1)
        assert out.shape[1] == self.n_tokens, out.shape
        return out.to(torch.bfloat16)


class ImgSetDiT(SetDiT):
    """SetDiT with the cross-attention context = projected image tokens + a learned per-token position embedding."""

    def __init__(self, token_dim, dim=768, depth=12, heads=12, enc_dim=1024, n_ctx=1029, **kw):
        super().__init__(token_dim, dim, depth, heads, cond_dim=enc_dim, n_ctx=n_ctx, **kw)
        del self.head_norm, self.head_proj
        self.ctx_pos = nn.Parameter(torch.zeros(n_ctx, dim))

    def encode_cond(self, tokens):
        return self.cond_proj(self.cond_norm(tokens.float())) + self.ctx_pos


def build_img_model(a, tok_dim):
    """ImgSetDiT for run args `a` (dim, depth, heads, dino, res) without loading DINOv3."""
    n_ctx, enc_dim = dino_tokens(a.dino, a.res)
    return ImgSetDiT(tok_dim, a.dim, a.depth, a.heads, enc_dim=enc_dim, n_ctx=n_ctx)


def load_img_dit(path, device='cuda'):
    """(model, head-frame tokenizer, run args) of an image DiT checkpoint (EMA weights; ema_XXXXXX.pt or
    model.safetensors)."""
    ck = load_ckpt(path)
    a = argparse.Namespace(**ck['args'])
    tok = build_tokenizer(a.vae, head=True, device=device)
    model = build_img_model(a, tok.dim).to(device).eval()
    model.load_state_dict(ck['ema'])
    return model, tok, a


def init_from(model, ckpt_path):
    """Copy every tensor of a trained point-cloud SetDiT (EMA) whose name and shape match; returns the skipped names."""
    sd = load_ckpt(ckpt_path)
    sd = sd.get('ema', sd.get('model', sd))
    own = model.state_dict()
    take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(take, strict=False)
    return sorted(set(own) - set(take))
