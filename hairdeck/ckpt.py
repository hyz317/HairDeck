"""Checkpoint files. The training scripts write torch .pt files (a DiT: ckpt/<run>/ema_XXXXXX.pt with 'ema', 'step',
'args' and for the chain DiT 'dim'; the card VAE: ckpt/<run>/model.pt with 'model', 'args'). A release holds the same
content as safetensors (ckpt/<model>/model.safetensors: the state dict as the tensors, every other entry in one JSON
string under the metadata key 'hairdeck', with 'weights' naming the entry of the state dict; one key because the
metadata of a safetensors file is a hash map written in a random order, so several keys would give other bytes on every
export). load() reads either form into the same dict. A training run writes into a folder of its own (training_dir).
"""
import json
import os

import torch

from . import paths

META_KEY = 'hairdeck'


def load(path):
    """A checkpoint as a dict: {'ema' or 'model': state dict (CPU tensors), 'args': dict, ...}."""
    if path.endswith('.safetensors'):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, 'pt') as f:
            out = json.loads(f.metadata()[META_KEY])
        out[out.pop('weights')] = load_file(path)
        return out
    return torch.load(path, map_location='cpu')


def vae_file(vae_dir):
    """The weights file of a card VAE folder: model.safetensors (a release) or model.pt (a training run)."""
    p = os.path.join(vae_dir, 'model.safetensors')
    return p if os.path.isfile(p) else os.path.join(vae_dir, 'model.pt')


def training_dir(name):
    """ckpt/<name> of a training run. A folder that holds a released checkpoint (model.safetensors, from
    scripts/setup/download_weights.py) is refused, so the files of a run never mix with the downloaded ones."""
    d = paths.ckpt(name)
    if os.path.isfile(os.path.join(d, 'model.safetensors')):
        raise SystemExit(f'{d} holds a released checkpoint; train under another --name (the launch scripts use train_<model>)')
    return d
