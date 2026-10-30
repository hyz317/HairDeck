"""Per-control-point frames of a card polyline (same arithmetic as the implementation that built the card data, so
outputs are bitwise identical to it).

tangents(p): (n,3) unit tangents from the 5-point stencil (-1/12, 2/3, 0, -2/3, 1/12) over p[i-2..i+2]
  (window clamped to the first / last five points). The stencil is the negated 4th-order central difference, so the
  tangent points from tip towards root. Cards with fewer than 5 points use the root->tip chord for every point.
perp(v, t): v with its component along unit t removed, normalised.
bishop(n0, T): rotation-minimising (parallel-transport) field: N[0] = perp(n0, T[0]); N[i] = N[i-1] rotated by the
  rotation taking T[i-1] to T[i] (Rodrigues; skipped when the tangents are parallel to 1e-9), then re-orthogonalised.
true_dir_wt(r): (unit width direction, width, thickness) of a 3- or 4-vertex cross-section ring, None otherwise.
  4 vertices: the diagonals are the farthest pair and the remaining pair; width = longer diagonal (its direction is
  r[j] - r[i] for the first farthest pair i < j in ring order), thickness = shorter. 3 vertices: width = longest side
  (first maximum over sides r1-r0, r2-r1, r0-r2), thickness = distance of the opposite vertex from that side.
"""
import numpy as np

_STENCIL = np.array([-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12])


def tangents(p):
    n = len(p)
    if n < 5:
        t = p[-1] - p[0]
        return np.repeat((t / (np.linalg.norm(t) + 1e-12))[None], n, 0)
    out = np.zeros_like(p)
    for i in range(n):
        lo = min(max(i - 2, 0), n - 5)
        t = (_STENCIL[:, None] * p[lo:lo + 5]).sum(0)
        out[i] = t / (np.linalg.norm(t) + 1e-12)
    return out


def perp(v, t):
    v = v - np.dot(v, t) * t
    return v / (np.linalg.norm(v) + 1e-12)


def bishop(n0, T):
    out = [perp(n0, T[0])]
    for i in range(1, len(T)):
        a, b = T[i - 1], T[i]
        ax = np.cross(a, b)
        s, c = np.linalg.norm(ax), np.clip(np.dot(a, b), -1, 1)
        n = out[-1]
        if s > 1e-9:
            k = ax / s
            n = n * c + np.cross(k, n) * s + k * np.dot(k, n) * (1 - c)
        out.append(perp(n, b))
    return np.array(out)


def true_dir_wt(r):
    if len(r) == 4:
        d = np.linalg.norm(r[:, None] - r[None], axis=-1)
        i, j = np.unravel_index(d.argmax(), d.shape)
        k, l = [m for m in range(4) if m not in (i, j)]
        d1, d2 = r[j] - r[i], r[l] - r[k]
        a, b = (d1, d2) if np.linalg.norm(d1) >= np.linalg.norm(d2) else (d2, d1)
        return a / (np.linalg.norm(a) + 1e-12), np.linalg.norm(a), np.linalg.norm(b)
    if len(r) == 3:
        sides = [(r[1] - r[0], r[2]), (r[2] - r[1], r[0]), (r[0] - r[2], r[1])]
        base, apex = max(sides, key=lambda s: np.linalg.norm(s[0]))
        bl = np.linalg.norm(base)
        foot = np.linalg.norm(np.cross(base, apex - r[0])) / (bl + 1e-12)
        return base / (bl + 1e-12), bl, foot
    return None
