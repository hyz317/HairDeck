"""Where data, checkpoints and third-party weights live.

Nothing is hard-coded: every location comes from an environment variable, else from a JSON config file, else it is
derived from the workspace root. Keys (environment variable / config key):

  HAIRDECK_ROOT         root          workspace root; the defaults below are directories under it
  HAIRDECK_DATA         data          data directory                         (default ROOT/data)
  HAIRDECK_CKPT         ckpt          checkpoints                            (default ROOT/ckpt)
  HAIRDECK_LOGS         logs          training logs                          (default ROOT/logs)
  HAIRDECK_RESULTS      results       sample and evaluation outputs          (default ROOT/results)
  HAIRDECK_SCRATCH      scratch       large intermediate caches              (default ROOT/scratch)
  HAIRDECK_WEIGHTS      weights       third-party weights, e.g. dora_vae_1_1/, dinov3/  (default ROOT/third_party/weights)
  HAIRDECK_DINOV3_REPO  dinov3_repo   local clone of facebookresearch/dinov3 for torch.hub (default ROOT/third_party/dinov3_src)
  HAIRDECK_SAM3_REPO    sam3_repo     local clone of facebookresearch/sam3 (hairdeck.prep; not needed when the sam3
                                       package is installed; default ROOT/third_party/sam3_src)
  HAIRDECK_CONFIG       -             JSON file with any of the keys above (default: configs/paths.json in the release
                                       tree, if it exists); environment variables take precedence over the file.

Example: export HAIRDECK_ROOT=/path/to/workspace   (see configs/paths.example.json for the file form)
"""
import json
import os

_ENV = {'root': 'HAIRDECK_ROOT', 'data': 'HAIRDECK_DATA', 'ckpt': 'HAIRDECK_CKPT', 'logs': 'HAIRDECK_LOGS',
        'results': 'HAIRDECK_RESULTS', 'scratch': 'HAIRDECK_SCRATCH', 'weights': 'HAIRDECK_WEIGHTS',
        'dinov3_repo': 'HAIRDECK_DINOV3_REPO', 'sam3_repo': 'HAIRDECK_SAM3_REPO'}
_UNDER_ROOT = {'data': 'data', 'ckpt': 'ckpt', 'logs': 'logs', 'results': 'results', 'scratch': 'scratch',
               'weights': os.path.join('third_party', 'weights'), 'dinov3_repo': os.path.join('third_party', 'dinov3_src'),
               'sam3_repo': os.path.join('third_party', 'sam3_src')}
_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'paths.json')


def _config():
    fn = os.environ.get('HAIRDECK_CONFIG') or _DEFAULT_CONFIG
    if os.path.isfile(fn):
        with open(fn) as f:
            return json.load(f)
    return {}


def get(key):
    """Absolute directory for one key (see the module docstring); raises if it cannot be resolved."""
    v = os.environ.get(_ENV[key])
    if v:
        return v
    cfg = _config()
    if cfg.get(key):
        return cfg[key]
    if key != 'root':
        return os.path.join(get('root'), _UNDER_ROOT[key])
    v = cfg.get('root')
    if not v:
        raise RuntimeError('hairdeck.paths: set HAIRDECK_ROOT (or HAIRDECK_DATA, HAIRDECK_CKPT, ...) or write '
                           f'{_DEFAULT_CONFIG} (see configs/paths.example.json)')
    return v


def root(*p):
    return os.path.join(get('root'), *p)


def data(*p):
    return os.path.join(get('data'), *p)


def ckpt(*p):
    return os.path.join(get('ckpt'), *p)


def logs(*p):
    return os.path.join(get('logs'), *p)


def results(*p):
    return os.path.join(get('results'), *p)


def scratch(*p):
    return os.path.join(get('scratch'), *p)


def weights(*p):
    return os.path.join(get('weights'), *p)


def dinov3_repo():
    return get('dinov3_repo')


def sam3_repo():
    return get('sam3_repo')


def resolve(p):
    """Command-line paths such as 'ckpt/NAME/model.safetensors' or 'results/RUN' are relative to the workspace layout: a
    path under data/, ckpt/, logs/, results/, scratch/ or third_party/weights/ goes to that folder (HAIRDECK_DATA, ...
    apply), any other to the root; absolute paths are returned unchanged."""
    if os.path.isabs(p):
        return p
    for key, base in sorted(_UNDER_ROOT.items(), key=lambda kv: -len(kv[1])):
        if key in ('data', 'ckpt', 'logs', 'results', 'scratch', 'weights') and (p == base or p.startswith(base + os.sep)):
            return os.path.join(get(key), p[len(base) + 1:]) if p != base else get(key)
    return root(p)
