"""The avatar body of the VRM export and the head-turn preview, read from a JSON file (default data/avatar/body.json: a
VRoid Studio template avatar) with the fields
  h          index of the body in the per-body tables of the export (tables(); any integer)
  key        name of the body;  vrm  its VRM 0.x file;  dir  a folder with the body's own hair as hair.glb, in the dataset
             frame (as the dataset folders hold it)
  fit        hairdeck.data.head_frame.fit_one of the body (Blender rest-pose hair / face boxes -> dataset frame)
  k, b       head-frame transform of the body, p_head = k * p + b (k = W / w, b = O - k * anchor, with the canonical head
             W, O of CANONICAL_HEAD)
  file_tag   name of the downloadable VRM (<file_tag>_seed<s>.vrm) and of the export's intermediate files (FILES:
             <file_tag>_gen.vrm, ...); one per body
  vrm_meta   (optional) VRM 0.x meta fields written into the exported VRM, by their VRM json names (the default body:
             title "HairDeck avatar", author hyz317, Apache-2.0 as licenseName Other + otherLicenseUrl, use by everyone
             and commercial use allowed)
Paths in the file are relative to the workspace root (hairdeck.paths.root()) or absolute. This module imports only json,
os and numpy besides hairdeck.paths, so the Blender scripts under blender/ import it too (apply_vrm_meta).

python -m hairdeck.vrm.body build VRM DATASET_DIR KEY H FILE_TAG OUT_JSON
  writes the body file of a VRM (without vrm_meta): Blender rest-pose boxes (blender/body_boxes.py, run like the export),
  fit_one, k and b. DATASET_DIR holds the body's hair as hair.glb.
"""
import json
import os

import numpy as np

from .. import paths

DEFAULT_BODY = ('avatar', 'body.json')  # under paths.data()
CANONICAL_HEAD = 'head_frame_train.npz'  # under paths.data(): W, O of the canonical head (head frame)
# intermediate files of an export (under out_dir/vrm_seed<s>/ and in the worker cache), named by the body's file_tag
FILES = dict(cards='{tag}_cards.json', prep='{tag}_prep', gen='{tag}_gen.vrm', export='{tag}_export.json',
             orig='{tag}_orig.vrm', nohair='{tag}_nohair')
_CACHE = {}

# VRM json name of a meta field -> property name of the VRM Add-on for Blender (vrm_addon_extension.vrm0.meta)
META_PROPS = dict(title='title', version='version', author='author', contactInformation='contact_information',
                  reference='reference', allowedUserName='allowed_user_name', violentUssageName='violent_ussage_name',
                  sexualUssageName='sexual_ussage_name', commercialUssageName='commercial_ussage_name',
                  otherPermissionUrl='other_permission_url', licenseName='license_name', otherLicenseUrl='other_license_url')


def default_path():
    return paths.data(*DEFAULT_BODY)


def file_name(kind, b):
    """Name of one intermediate file of an export on body b (FILES: cards, prep (+ .npz / .json), gen, export, orig,
    nohair (+ .blend / .json))."""
    return FILES[kind].format(tag=b['file_tag'])


def body_path(path=None):
    """Absolute path of a body file (relative paths follow the workspace layout, paths.resolve; default default_path())."""
    return os.path.abspath(paths.resolve(path) if path else default_path())


def _abs(p):
    """a path of a body file: relative ones follow the workspace layout (data/... under HAIRDECK_DATA, paths.resolve)"""
    return os.path.abspath(paths.resolve(p))


def body(path=None):
    """The body of a body file (default default_path()): vrm and dir as absolute paths, k and b as Python floats, h as an
    int; read once per file."""
    path = body_path(path)
    if path not in _CACHE:
        b = json.load(open(path))
        b.update(vrm=_abs(b['vrm']), dir=_abs(b['dir']), b=list(map(float, b['b'])), k=float(b['k']), h=int(b['h']))
        _CACHE[path] = b
    return _CACHE[path]


def tables(b):
    """(meta, rl, fits) of one body, indexed like the test-set tables (meta[h]['key'], meta[h]['dir'], rl[key],
    fits[key])."""
    return {b['h']: dict(key=b['key'], dir=b['dir'])}, {b['key']: b['vrm']}, {b['key']: b['fit']}


def apply_vrm_meta(arm, b=None):
    """Blender side: write the body's vrm_meta (b: a body dict, default the default body) into the armature's VRM 0.x
    meta before the export; returns the fields written."""
    over = (b if b is not None else body()).get('vrm_meta') or {}
    meta = arm.data.vrm_addon_extension.vrm0.meta
    for k, v in over.items():
        setattr(meta, META_PROPS[k], v)
    return dict(over)


def build(vrm, ddir, key, h, tag, out_json):
    """Blender rest-pose boxes of the body (blender/body_boxes.py: the fields blender/gen_data.py writes into meta.json),
    then hairdeck.data.head_frame.fit_one and the head-frame transform k = W / w, b = O - k * anchor. Writes OUT_JSON
    and OUT_JSON's name + _blender.json (the boxes); returns the body dict."""
    import subprocess
    from ..data.head_frame import anchor, fit_one
    from .export import SCRIPTS, blender_cmd
    meta_json = os.path.splitext(out_json)[0] + '_blender.json'
    r = subprocess.run(blender_cmd() + ['-b', '--python', os.path.join(SCRIPTS, 'body_boxes.py'), '--', vrm, meta_json],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(meta_json):
        raise RuntimeError('body_boxes.py failed: ' + (r.stdout + r.stderr)[-2000:])
    _, fit = fit_one((key, ddir, meta_json))
    if fit.get('status') != 'ok':
        raise RuntimeError(f'head fit failed: {fit}')
    hz = np.load(paths.data(CANONICAL_HEAD))
    W, O = float(hz['W']), hz['O'].astype(np.float64)
    k = W / fit['w']
    b = O - k * anchor(fit)
    root = paths.root()

    def rel(p):  # relative only when it reads back as the same file (paths.resolve, e.g. with another HAIRDECK_DATA)
        p = os.path.abspath(p)
        r = os.path.relpath(p, root)
        return r if not r.startswith('..') and os.path.abspath(paths.resolve(r)) == p else p
    out = dict(h=int(h), key=key, vrm=rel(vrm), dir=rel(ddir), fit=fit, k=float(k), b=b.tolist(), file_tag=tag,
               blender_meta=rel(meta_json), made_by='hairdeck.vrm.body build')
    json.dump(out, open(out_json, 'w'), indent=1)
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 8 and sys.argv[1] == 'build':
        print(json.dumps(build(*sys.argv[2:5], int(sys.argv[5]), sys.argv[6], sys.argv[7]), indent=1))
    else:
        sys.exit(__doc__)
