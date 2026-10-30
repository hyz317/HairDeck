"""Convert the released Dora-VAE 1.1 checkpoint (Seed3D/Dora-VAE-1.1 dora_vae_1_1.ckpt, a Lightning checkpoint) once
into a safetensors file with only its shape_model weights, the file hairdeck/encoders/dora.py loads, so nothing
downstream unpickles the checkpoint.

The checkpoint's hyper-parameters hold omegaconf / craftsman objects, so torch.load(weights_only=True) refuses it. It is
unpickled once, and only after its sha256 matched the Hugging Face listing (hairdeck.encoders.dora.DORA_CKPT); every
name the pickle loads outside torch and collections (the hyper-parameters' classes, and the builtins they are rebuilt
with) becomes an inert stand-in, so no training code is imported or run.
The tensors are written unchanged. The metadata is one key (source: repository, revision, file, sha256), so every run
writes the same bytes (safetensors writes several metadata keys in a random order).

python scripts/setup/convert_dora_ckpt.py [--src CKPT] [--dst SAFETENSORS]
       (defaults: paths.weights('dora_vae_1_1/dora_vae_1_1.ckpt') -> paths.weights('dora_vae_1_1/
       dora_vae_1_1_shape_model.safetensors'); scripts/setup/download_weights.py --only dora downloads the checkpoint,
       runs this and checks the output's sha256)
"""
import argparse
import hashlib
import os
import pickle
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hairdeck import paths  # noqa: E402
from hairdeck.encoders.dora import DORA_CKPT, DORA_WEIGHTS  # noqa: E402

SAFE = ('torch', 'collections')  # the modules the checkpoint's pickle may load from for real (the state dict)


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


class _Inert:
    """Stand-in for every other name of the pickle (the hyper-parameters: craftsman, omegaconf, typing, the builtins they
    are rebuilt with): keeps its arguments' state, runs no code."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self._state = s


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split('.')[0] in SAFE:
            return super().find_class(mod, name)
        return _Inert


def convert(src, dst):
    repo, rev, fname, want = DORA_CKPT
    got = sha256(src)
    if got != want:
        raise SystemExit(f'{src}: sha256 {got} != {want} (the {fname} of {repo} at {rev[:8]}); not unpickled')
    import torch
    from safetensors.torch import save_file
    pm = types.ModuleType('pickle_inert')
    pm.Unpickler, pm.load, pm.__name__ = _Unpickler, lambda f, **kw: _Unpickler(f, **kw).load(), 'pickle'
    ck = torch.load(src, map_location='cpu', weights_only=False, pickle_module=pm)
    sd = ck.get('state_dict', ck)
    out = {k[len('shape_model.'):]: v.contiguous() for k, v in sd.items() if k.startswith('shape_model.')}
    if not out:
        raise SystemExit(f'{src}: no shape_model weights')
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    save_file(out, dst + '.part', metadata={'source': f'{repo}@{rev} {fname} sha256 {want}, shape_model weights'})
    os.replace(dst + '.part', dst)
    print(f'{dst}: {len(out)} tensors, {sum(v.numel() for v in out.values())} parameters, sha256 {sha256(dst)}',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='', help=f'default: weights/{DORA_WEIGHTS[0]}/{DORA_CKPT[2]}')
    ap.add_argument('--dst', default='', help=f'default: weights/{DORA_WEIGHTS[0]}/{DORA_WEIGHTS[1]}')
    a = ap.parse_args()
    convert(a.src or paths.weights(DORA_WEIGHTS[0], DORA_CKPT[2]), a.dst or paths.weights(*DORA_WEIGHTS))


if __name__ == '__main__':
    main()
