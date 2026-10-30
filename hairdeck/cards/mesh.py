"""Card -> triangle mesh (no CHARM code).

A card is (n, 8): xyz, width, thickness, root width direction (row 0 of columns 5:8 is used). The width direction
field is the Bishop transport of the root direction along the 5-point-stencil tangents (frames.py). Ring i has 4
vertices around p_i, with n_i the unit width direction, g_i the tangent made orthogonal to n_i, and b_i = n_i x g_i:
  square:   p + w/2 n, p + t/2 b, p - w/2 n, p - t/2 b
  triangle: p + w/2 n, p + t/3 b, p - w/2 n, p - t/6 b
Faces (vertex 4i+k = corner k of ring i): root cap (0,2,1), (0,3,2); for every ring i >= 1 eight side triangles
joining ring i-1 and ring i, then a cap (4i, 4i+1, 4i+2), (4i, 4i+2, 4i+3). The vertex and face order is the one used
to build and evaluate the original card data (checked bitwise), so seeded surface sampling gives the same points.

UVs (card_uv, cards_to_mesh(..., uv=True)): u runs across the width (0 at the +width corner, 1 at the -width corner,
0.5 at both thickness corners, so the two faces of a card are mirror images and the mesh needs no seam), v runs along
the card (arc length of the control points from the root over the card length). Vertices and faces are the same with
and without UVs.

5-column cards (CHARM's hair.json templates) are not handled here: their normal field is CHARM's least-squares solve.
Every function that meshes cards takes an optional `mesher` for other card formats.
"""
import numpy as np
import trimesh

from .frames import bishop, tangents

SHAPE_NAMES = ['square', 'triangle']


def _side_faces(i):
    """8 triangles between ring i-1 and ring i (i >= 1)."""
    cur, prev = 4 * i, 4 * i - 4
    out = [[cur + k, prev + k, cur + (k + 1) % 4] for k in range(4)]
    out += [[prev + (k + 1) % 4, cur + (k + 1) % 4, prev + k] for k in (0, 3, 2, 1)]
    return out


def ring_faces(n_rings):
    f = [[0, 2, 1], [0, 3, 2]]
    for i in range(1, n_rings):
        c = 4 * i
        f += _side_faces(i) + [[c, c + 1, c + 2], [c, c + 2, c + 3]]
    return np.array(f, dtype=np.int64)


def mesh_from_normals(seq, normals, shape):
    """seq (n, >=5): xyz, width, thickness; normals (n,3) width directions -> (vertices (4n,3), faces)."""
    T = tangents(seq[:, :3])
    up, down = (3, 6) if shape == 'triangle' else (2, 2)  # thickness divisors on the +b / -b side
    verts = []
    for i in range(len(seq)):
        p, w, t = seq[i, :3], seq[i, 3], seq[i, 4]
        n = normals[i] / (np.linalg.norm(normals[i]) + 1e-12)
        g = T[i] - np.dot(T[i], n) * n
        g /= np.linalg.norm(g) + 1e-12
        b = np.cross(n, g)
        verts += [p + w / 2 * n, p + t / up * b, p - w / 2 * n, p - t / down * b]
    return np.array(verts), ring_faces(len(seq))


def bishop_mesh(seq8, shape):
    """seq8: (n, 8) card -> (vertices, faces), the root width direction transported with Bishop frames."""
    return mesh_from_normals(seq8, bishop(seq8[0, 5:8], tangents(seq8[:, :3])), shape)


def card_mesh_vf(seq, sh):
    """(vertices float64, faces int64) of one card; sh = shape id (0 square, 1 triangle)."""
    if seq.shape[1] < 8:
        raise ValueError('5-column cards (CHARM hair.json templates) need CHARM\'s own mesher (mesher=); hairdeck '
                         'meshes 8-column cards')
    v, f = bishop_mesh(seq, SHAPE_NAMES[sh])
    return np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(f, dtype=np.int64)


def card_uv(seq):
    """(4n, 2) float64 UVs of card_mesh_vf's vertices (4 per ring, ring i = control point i), see the module docstring."""
    p = np.asarray(seq, np.float64)[:, :3]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    v = s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, len(p))
    u = np.array([0.0, 0.5, 1.0, 0.5])
    return np.stack([np.tile(u, len(p)), np.repeat(v, 4)], 1)


def _uv_after_cleanup(verts, v0, uv0):
    """UVs of the vertices left after trimesh's cleanup (construction merges coincident vertices, e.g. the corners of a
    ring of zero width, and keeps the coordinates of one of them; unreferenced vertices are dropped): the mean UV of the
    original vertices with exactly the same coordinates."""
    at = {}
    for k, row in enumerate(np.ascontiguousarray(v0, np.float64)):
        at.setdefault(row.tobytes(), []).append(k)
    return np.array([uv0[at[row.tobytes()]].mean(0) for row in np.ascontiguousarray(verts, np.float64)])


def cards_to_mesh(cards, mesher=None, uv=False):
    """cards: list of (seq, shape_id) -> trimesh.Scene (one mesh per card with >= 3 points); uv=True adds card_uv as
    texture coordinates (exported to GLB as TEXCOORD_0), geometry unchanged; a vertex merged by the cleanup takes the
    mean UV of the corners it replaces."""
    mesher = mesher or card_mesh_vf
    meshes = []
    for seq, sh in cards:
        if len(seq) < 3:
            continue
        v, f = mesher(seq, sh)
        m = trimesh.Trimesh(vertices=v, faces=f)
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        if uv:
            if len(v) != 4 * len(seq):
                raise ValueError('card_uv needs 4 vertices per control point (card_mesh_vf layout)')
            m.visual = trimesh.visual.TextureVisuals(uv=_uv_after_cleanup(m.vertices, v, card_uv(seq)))
        meshes.append(m)
    return trimesh.Scene(meshes)


def cards_to_json(cards):
    """CHARM-style json list: [{'shape': 'square' | 'triangle', 'seq': rows}, ...]."""
    return [{'shape': SHAPE_NAMES[sh], 'seq': seq.tolist()} for seq, sh in cards]
