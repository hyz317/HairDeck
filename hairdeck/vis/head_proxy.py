"""Head proxy: an ellipsoid standing in for the skull at the canonical head of the head frame (face width W, face-top
anchor O; hairdeck.data.head_frame), shown with generated hair in the GLB files of hairdeck.infer (the head-turn preview
draws the same ellipsoid).
"""
import numpy as np
import trimesh

from .. import paths


def canonical(path=None):
    """W, O of the head frame (default: data/head_frame_train.npz)."""
    z = np.load(path or paths.data('head_frame_train.npz'))
    return float(z['W']), z['O']


def head_proxy(W=None, O=None):
    """Ellipsoid (icosphere, subdivision 4) with radii (0.405, 0.5, 0.5) W, centred 0.1 W above and 0.2 W behind the face
    centre. Grid search on 1500 train hairstyles in the head frame: hair roots sit 0.047 W outside it (median; 0.090 W
    for centre +0.05 W, x radius 0.45 W) and 3.2% of hair points are more than 0.02 W inside (4.2%)."""
    if W is None:
        W, O = canonical()
    face_c = O - np.array([0, 0.547 * W, 0])  # face-top anchor -> face centre (face height ~1.094 W)
    c = face_c + np.array([0, 0.10 * W, -0.2 * W])
    m = trimesh.creation.icosphere(subdivisions=4)
    m.apply_scale([0.405 * W, 0.5 * W, 0.5 * W])
    m.apply_translation(c)
    return m
