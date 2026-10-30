"""Rendered views of a character and the input view of the test-set evaluation.

A character's renders (scripts/data.py) live in RENDER_ROOT/<key with '/' -> '_'>/: meta.json, v<i>.png (RGBA)
and v<i>_lab.png (labels, hairdeck.data.crop) for the views i = 0..7. meta.json['views'][i]['view'] holds the view's
bucket ('front', 'side', 'back') and az_rel, its azimuth relative to the head.
"""
import os

VIEW_TARGETS = {'q34': ('front', 40.0), 'side': ('side', 90.0), 'back': ('back', 180.0)}


def render_dir(render_root, key):
    return f'{render_root}/{key.replace("/", "_")}'


def dir_to_key(d):
    """'9_7067719180331438579' -> '9/7067719180331438579'; 'vroid-hair_2_114...' -> 'vroid-hair/2/114...'."""
    return '/'.join(os.path.basename(d.rstrip('/')).rsplit('_', 2))


def pick_view(m, how='front'):
    """m: a render meta.json. 'front': the most frontal view of the front bucket (view 0 if the character has none);
    'q34' / 'side' / 'back': the view of that bucket whose |az_rel| is closest to 40 / 90 / 180 deg (None if the
    character has none); else a view index."""
    if how not in ('front', *VIEW_TARGETS):
        return int(how)
    bucket, target = VIEW_TARGETS.get(how, ('front', 0.0))
    cand = [i for i, v in enumerate(m['views']) if v['view']['bucket'] == bucket]
    if not cand:
        return 0 if how == 'front' else None
    return min(cand, key=lambda i: abs(abs(m['views'][i]['view']['az_rel']) - target))
