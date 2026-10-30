"""Card topology for the template extraction: end groups, end classification, cross-section rings.

Clean-room replacement for the three CHARM `hair_utils.mesh2tmp` routines (`split_two_ends`, `check_end_type`,
`regularize`) that the original card extraction used. Written from a behavioural spec obtained by running the reference
functions as black boxes (real VRoid cards, synthetic tubes, duck-typed meshes whose `vertex_degree` was forced). No
reference code was copied. On the 36,440 train hairstyles the outputs are bitwise identical to the reference.

Input convention: one hair card as a `trimesh.Trimesh` after the extractor's clean-up (vertices merged at 1e-8,
connected component, unique and non-degenerate faces). A VRoid card is a closed triangulated tube of 4-vertex
(square/diamond) or 3-vertex (triangle) cross-sections; interior vertices touch 6 faces; each end is a flat cap
(two triangles / one triangle), a pyramid tip, or (rarely) a collapsed ring.

Spec
----
"face count" fc[v] = number of faces using vertex v (trimesh `vertex_degree`).

end_groups(mesh) -> (g1, g2), two lists of vertex ids:
  1. C = vertices with fc != 6, ascending.
  2. g1 = breadth-first order of the connected part of C that contains C[0] (adjacency = sharing a face; only
     vertices of C are entered; each vertex's neighbours are visited in ascending id). g2 = the rest of C, ascending.
  3. if len(g1) == 1: sklearn KMeans(n_clusters=2, random_state=0) on the positions of C (ascending order);
     g1 = members of cluster 0, g2 = members of cluster 1 (both ascending). Fewer than 2 points raises sklearn's error.
  4. if the sizes are {2, 4}: the 2-member group is grown with vertices outside g1+g2 lying within 1e-4 of it.
     P = vertices outside g1+g2 (ascending); M = |group| x |P| Euclidean distances (rows in group order).
     At most twice: (r, c) = first minimum of M in row-major order; if M[r, c] < 1e-4 append P[c] and delete
     column c from M (P itself keeps all entries, so a second pick right of the first is mapped one id too low and
     can repeat the first id; reproduced on purpose); otherwise stop.
  5. if g2 is empty and fc over g1 is exactly {4: 3, 5: 6} (a 2-ring square card, one flat cap, one tip): g1 becomes
     4 vertices: seed = g1[i] for the first minimum (i, j) of the pairwise distance matrix of g1 (g1 order, diagonal
     excluded), followed by the 3 members nearest to the seed (ascending distance, lower position on ties).

end_type(fc, group) -> (shape, top, rest). Decided only by the multiset of fc over the group:
  {3,3,3} {4,4,4}                      -> 'triangle', no top
  {3,5,5,5}                            -> 'triangle', top = the fc-3 vertex
  {3,3,6} {5,5,6} {3,3,3,3} {3,3,6,6}
  {3,4,4,4} {4,4,5,5}                  -> 'square', no top
  {4,5,5,5,5}                          -> 'square', top = the fc-4 vertex
  rest = the other members in input order (duplicates kept). Anything else raises Exception('Unknown end type',
  bincount(fc)). (For unknown groups whose face counts are all <= 5 the reference often raises IndexError instead:
  same failure, different message.)

cross_sections(mesh, g1, g2) -> (shape, centroids (n,3) float64, rings: list of (k,3) arrays):
  start = g1, unless np.var(positions of g1) < np.var(positions of g2) (variance over all 3k coordinates; an empty
  group never wins), then start = g2. shape, top, rest = end_type(fc, start).
  Rings: [top] if any, then the set of `rest`, then repeatedly the set of not-yet-reached neighbours
  (trimesh `vertex_neighbors`) of the previous ring, until every vertex is reached. Ring members are listed in
  CPython set iteration order of a set filled in this order: first ring from `rest`; next ring by walking the previous
  ring in its listed order and each vertex's `vertex_neighbors` in trimesh order. This order fixes the floating-point
  sum of the centroid and the sign of the width direction downstream, so it is reproduced exactly.
  centroid = mean of the ring's positions.
  Validity (the reference's least-squares normals and width/thickness are discarded by the extractor, but their
  input checks decide which hairstyles fail, so they are kept):
    fewer than 3 rings                   -> ValueError('At least 3 points are needed to compute normals')
    first ring (in walk order) of size 0, 2 or >= 5 -> Exception('Unknown shape', size)
    a 3-vertex ring is accepted if its longest side is below 1e-6 (collapsed ring) or two of its side lengths
      differ by less than 0.1 x the mean side length, else Exception('Not a isosceles triangle')
  Not reproduced: the reference can raise TypeError for 4-vertex rings that are badly irregular relative to its
  least-squares normal; it never fired on the full train/test extraction. The
  reference loops forever when some vertex is unreachable from the start ring; here that raises RuntimeError.
"""
from collections import deque

import numpy as np

NEAR = 1e-4
TINY_SIDE = 1e-6
_END_TABLE = {
    (3, 3, 3): ('triangle', None),
    (4, 4, 4): ('triangle', None),
    (3, 5, 5, 5): ('triangle', 3),
    (3, 3, 6): ('square', None),
    (5, 5, 6): ('square', None),
    (3, 3, 3, 3): ('square', None),
    (3, 3, 6, 6): ('square', None),
    (3, 4, 4, 4): ('square', None),
    (4, 4, 5, 5): ('square', None),
    (4, 5, 5, 5, 5): ('square', 4),
}


def _face_neighbours(faces, nv):
    """Sorted neighbour id arrays (vertices sharing a face)."""
    f = np.asarray(faces, dtype=np.int64)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    keep = a != b
    pair = np.unique(np.stack([a[keep], b[keep]], 1), axis=0)  # sorted by (a, b)
    starts = np.searchsorted(pair[:, 0], np.arange(nv + 1))
    return [pair[starts[v]:starts[v + 1], 1] for v in range(nv)]


def _dist(p, q):
    """Euclidean distance of two 3-vectors, computed as a 1-D vector norm (same rounding as the reference)."""
    return np.linalg.norm(p - q)


def _cluster_split(V, cand):
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=2, random_state=0).fit(V[cand]).labels_
    return [int(v) for v, l in zip(cand, lab) if l == 0], [int(v) for v, l in zip(cand, lab) if l == 1]


def _grow_pair(V, grp, others, nv):
    taken = set(grp) | set(others)
    pool = [v for v in range(nv) if v not in taken]
    M = np.zeros((len(grp), len(pool)))
    for r, a in enumerate(grp):
        for c, b in enumerate(pool):
            M[r, c] = _dist(V[a], V[b])
    out = list(grp)
    for _ in range(2):
        if M.size == 0:
            break
        r, c = np.unravel_index(np.argmin(M), M.shape)
        if not M[r, c] < NEAR:
            break
        out.append(pool[c])
        M = np.delete(M, c, axis=1)
    return out


def _tight_four(V, grp):
    n = len(grp)
    M = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = _dist(V[grp[i]], V[grp[j]])
    i = np.unravel_index(np.argmin(M), M.shape)[0]
    row = M[i].copy()
    out = [grp[i]]
    for _ in range(3):
        k = int(np.argmin(row))
        out.append(grp[k])
        row[k] = np.inf
    return out


def end_groups(mesh):
    """Vertex ids of the two card ends (see module spec, end_groups)."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    fc = np.asarray(mesh.vertex_degree)
    nv = len(V)
    cand = [int(v) for v in np.flatnonzero(fc != 6)]
    if not cand:
        raise IndexError('card has no end vertices (every vertex touches 6 faces)')
    is_cand = np.zeros(nv, bool)
    is_cand[cand] = True
    nbr = _face_neighbours(mesh.faces, nv)
    order, seen, q = [cand[0]], {cand[0]}, deque([cand[0]])
    while q:
        for u in nbr[q.popleft()]:
            u = int(u)
            if is_cand[u] and u not in seen:
                seen.add(u)
                order.append(u)
                q.append(u)
    g1, g2 = order, [v for v in cand if v not in seen]
    if len(g1) == 1:
        g1, g2 = _cluster_split(V, cand)
    if sorted((len(g1), len(g2))) == [2, 4]:
        if len(g1) == 2:
            g1 = _grow_pair(V, g1, g2, nv)
        else:
            g2 = _grow_pair(V, g2, g1, nv)
    if not g2 and _fc_multiset(fc, g1) == (4, 4, 4, 5, 5, 5, 5, 5, 5):
        g1 = _tight_four(V, g1)
    return g1, g2


def _fc_multiset(fc, grp):
    return tuple(sorted(int(fc[v]) for v in grp))


def end_type(fc, group):
    """(shape, top, rest) for one end group (see module spec, end_type). fc: per-vertex face counts."""
    fc = np.asarray(fc)
    key = _fc_multiset(fc, group)
    if key not in _END_TABLE:
        raise Exception('Unknown end type', np.bincount(np.asarray([fc[v] for v in group], dtype=np.int64)))
    shape, top_fc = _END_TABLE[key]
    if top_fc is None:
        return shape, None, list(group)
    top = next(v for v in group if fc[v] == top_fc)
    return shape, top, [v for v in group if v != top]


def _triangle_ok(r):
    s = [np.linalg.norm(r[1] - r[0]), np.linalg.norm(r[2] - r[1]), np.linalg.norm(r[0] - r[2])]
    if max(s) < TINY_SIDE:
        return True
    tol = 0.1 * (s[0] + s[1] + s[2]) / 3
    return abs(s[0] - s[1]) < tol or abs(s[1] - s[2]) < tol or abs(s[0] - s[2]) < tol


def _check_rings(rings):
    if len(rings) < 3:
        raise ValueError('At least 3 points are needed to compute normals')
    for r in rings:
        k = len(r)
        if k in (1, 4):
            continue
        if k == 3:
            if not _triangle_ok(r):
                raise Exception('Not a isosceles triangle')
            continue
        raise Exception('Unknown shape', k)


def _spread(V, grp):
    return np.var(V[np.asarray(grp, dtype=np.int64)]) if len(grp) else np.nan


def cross_sections(mesh, g1, g2):
    """(shape, centroids (n,3), rings) walking from the chosen end (see module spec, cross_sections)."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    start = g2 if _spread(V, g1) < _spread(V, g2) else g1
    shape, top, rest = end_type(mesh.vertex_degree, start)
    adj = mesh.vertex_neighbors
    nv = len(V)
    reached = set()
    cents, rings = [], []
    if top is not None:
        reached.add(int(top))
        rings.append(V[[int(top)]])
        cents.append(V[int(top)])
    ring = set(int(v) for v in rest)
    while len(reached) < nv:
        if not ring:
            raise RuntimeError('card vertices unreachable from the start ring')
        reached |= ring
        nxt = set()
        for v in ring:
            for u in adj[v]:
                if u not in reached:
                    nxt.add(int(u))
        pts = V[list(ring)]
        rings.append(pts)
        cents.append(pts.mean(0))
        ring = nxt
    _check_rings(rings)
    return shape, np.array(cents, dtype=np.float64), rings


def card_sections(card):
    """end_groups + cross_sections for one cleaned card mesh."""
    g1, g2 = end_groups(card)
    return cross_sections(card, g1, g2)
