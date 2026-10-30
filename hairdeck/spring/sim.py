"""Spring-bone simulation of hair cards driven by chains (generated or from the VRM), without Blender.

  skinning: a card follows the chain whose path is nearest to the lower 60% of its centreline when that mean distance is
        below ASSIGN face widths, else it stays rigid with the head; each point binds to the nearest segment of that
        chain, with weight ramping from 0 at the chain root to 1 over RAMP face widths of arc length; with BLEND the
        binding is blended linearly between adjacent segments around each joint (no tearing at joints)
  simulation: the VRM 0.x spring-bone update (UniVRM VRMSpringBoneLogic: verlet with drag, stiffness pull toward the
        parent-rotated rest direction, gravity, length constraint; no colliders), 60 fps, meters
  motions about the head joint (2.5 s each): turn (yaw 50 deg in 0.35 s, hold), nod (pitch 25 deg down and back in
        0.5 s), jump (0.15 m up and down in 0.6 s), sway (sideways 6 cm at 1.5 Hz for two cycles, then stop)
Chains: list of dict(J (n, 3) joint positions in meters, stiff, drag, grav (3,)). Used by eval/chain_motion.py and
eval/chain_fit.py.
The update follows UniVRM (https://github.com/vrm-c/UniVRM; MIT License, Copyright (c) 2020 VRM Consortium, Copyright
(c) 2018 Masataka SUMI for MToon; the full notice is in blender/springbone.py).
"""
import numpy as np

FPS, NF = 60, 150
# face widths; tuned against the VRoid characters' own skin weights on 200 train characters
ASSIGN, RAMP = 0.4, 0.05
BLEND = True
NPT = 16
MOTIONS = ('turn', 'nod', 'jump', 'sway')


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def motion(name):
    """Head rotation (NF, 3, 3) and translation (NF, 3) in meters; dataset frame (y up, face toward +z)."""
    t = np.arange(1, NF + 1) / FPS
    R = np.repeat(np.eye(3)[None], NF, 0)
    T = np.zeros((NF, 3))
    if name == 'turn':
        x = np.clip(t / 0.35, 0, 1)
        R = np.stack([rot_y(np.radians(50) * xx * xx * (3 - 2 * xx)) for xx in x])
    elif name == 'nod':
        R = np.stack([rot_x(np.radians(25) * np.sin(np.pi * tt / 0.5) if tt < 0.5 else 0.0) for tt in t])
    elif name == 'jump':
        T[:, 1] = np.where(t < 0.6, 0.15 * 4 * (t / 0.6) * (1 - t / 0.6), 0)
    elif name == 'sway':
        T[:, 0] = np.where(t < 4 / 3, 0.06 * np.sin(2 * np.pi * 1.5 * t), 0)
    return R, T


def rotdiff(a, b):
    """Rotation matrix taking unit vector a to unit vector b (shortest arc)."""
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-12:
        return np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def clean(J):
    """Joint polyline with zero-length segments removed."""
    J = np.asarray(J, np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(J, axis=0), axis=1) > 1e-6]
    return J[keep]


def simulate(chains, R, T, piv, dt=1.0 / FPS):
    """chains: list of dict(J (n, 3) meters, stiff, drag, grav (3,)). Returns per chain (heads (NF, n-1, 3),
    rots (NF, n-1, 3, 3)): bone head positions and rest-to-current world rotations."""
    out = []
    for c in chains:
        J = c['J']
        d = np.diff(J, axis=0)
        L = np.linalg.norm(d, axis=1)
        D = d / L[:, None]
        cur = J[1:].copy()
        prev = cur.copy()
        heads = np.zeros((NF, len(L), 3))
        rots = np.zeros((NF, len(L), 3, 3))
        for f in range(NF):
            Wp = R[f]
            head = R[f] @ (J[0] - piv) + piv + T[f]
            for k in range(len(L)):
                rest = Wp @ D[k]
                nxt = cur[k] + (cur[k] - prev[k]) * (1 - c['drag']) + rest * (c['stiff'] * dt) + c['grav'] * dt
                v = nxt - head
                n = np.linalg.norm(v)
                v = v / n if n > 1e-12 else rest
                nxt = head + v * L[k]
                prev[k], cur[k] = cur[k], nxt
                Wk = rotdiff(rest, v) @ Wp
                heads[f, k], rots[f, k] = head, Wk
                Wp, head = Wk, nxt
        out.append((heads, rots))
    return out


def project(P, J):
    """Points P (m, 3) onto polyline J (n, 3): distance, segment index, arc length from J[0] (clipped to the line)."""
    a, ab = J[:-1], np.diff(J, axis=0)
    l2 = (ab ** 2).sum(1)
    u = np.clip(((P[:, None] - a[None]) * ab[None]).sum(-1) / l2[None], 0, 1)
    dist = np.linalg.norm(P[:, None] - (a[None] + u[..., None] * ab[None]), axis=-1)
    k = dist.argmin(1)
    ar = np.arange(len(P))
    cum = np.r_[0, np.cumsum(np.sqrt(l2))]
    return dist[ar, k], k, cum[k] + u[ar, k] * np.sqrt(l2[k])


def skin(cards_pts, chains, W):
    """cards_pts: list of (m, 3) card centreline points, root first (meters). Returns per point: chain (-1 = rigid),
    segment, weight (concatenated over cards)."""
    ch, sg, wt = [], [], []
    for P in cards_pts:
        m = len(P)
        lo = P[int(0.4 * m):]
        best, bd = -1, ASSIGN * W
        for ci, c in enumerate(chains):
            dm = project(lo, c['J'])[0].mean()
            if dm < bd:
                best, bd = ci, dm
        if best < 0:
            ch.append(np.full(m, -1))
            sg.append(np.zeros(m, int))
            wt.append(np.zeros(m))
            continue
        _, k, arc = project(P, chains[best]['J'])
        ch.append(np.full(m, best))
        sg.append(k)
        wt.append(np.clip(arc / (RAMP * W), 0, 1))
    return np.concatenate(ch), np.concatenate(sg), np.concatenate(wt)


def skin_blend(cards_pts, chains, W):
    """skin() plus a second segment and blend factor per point: on segment k at parameter u in [0, 1], the point follows
    k with weight alpha and its neighbour with 1 - alpha, alpha = 0.5 + u toward k - 1 (u < 0.5) or 1.5 - u toward k + 1
    (u > 0.5), so the binding is continuous across joints. Returns (chain, segment, weight, segment2, alpha)."""
    ch, sg, wt = skin(cards_pts, chains, W)
    P = np.concatenate(cards_pts)
    sg2, al = sg.copy(), np.ones(len(sg))
    for ci, c in enumerate(chains):
        sel = np.nonzero(ch == ci)[0]
        if not len(sel):
            continue
        J = c['J']
        a, ab = J[:-1], np.diff(J, axis=0)
        k = sg[sel]
        u = np.clip(((P[sel] - a[k]) * ab[k]).sum(-1) / (ab[k] ** 2).sum(-1), 0, 1)
        nseg = len(ab)
        lo = (u < 0.5) & (k > 0)
        hi = (u > 0.5) & (k < nseg - 1)
        sg2[sel[lo]], al[sel[lo]] = k[lo] - 1, 0.5 + u[lo]
        sg2[sel[hi]], al[sel[hi]] = k[hi] + 1, 1.5 - u[hi]
    return ch, sg, wt, sg2, al


def bind(cards_pts, chains, W):
    return skin_blend(cards_pts, chains, W) if BLEND else skin(cards_pts, chains, W)


def deform(V, binding, chains, sim, R, T, piv):
    """Positions (NF, m, 3) of rest points V under the simulated rig. binding: skin() or skin_blend() output."""
    ch, sg, wt = binding[:3]
    sg2, al = (binding[3], binding[4]) if len(binding) == 5 else (sg, np.ones(len(sg)))
    rigid = np.einsum('fij,mj->fmi', R, V - piv) + piv + T[:, None]
    X = rigid.copy()
    for ci in range(len(chains)):
        sel = np.nonzero(ch == ci)[0]
        if not len(sel):
            continue
        heads, rots = sim[ci]
        J = chains[ci]['J']
        k, k2, a = sg[sel], sg2[sel], al[sel][None, :, None]
        follow = heads[:, k] + np.einsum('fmij,mj->fmi', rots[:, k], V[sel] - J[k])
        follow2 = heads[:, k2] + np.einsum('fmij,mj->fmi', rots[:, k2], V[sel] - J[k2])
        follow = a * follow + (1 - a) * follow2
        w = wt[sel][None, :, None]
        X[:, sel] = (1 - w) * rigid[:, sel] + w * follow
    return X


def local_disp(X, V, R, T, piv):
    """Displacement of the simulated points relative to the rigidly moving head."""
    return np.einsum('fji,fmj->fmi', R, X - piv - T[:, None]) + piv - V


def centreline(seq, n=NPT):
    """n arc-length samples of a card's control polyline."""
    p = np.asarray(seq, np.float64)[:, :3]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, p[:, k]) for k in range(3)], 1)
