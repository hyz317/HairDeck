"""Arc-length helpers for card polylines and the fixed-K card features used by the set DiT data files."""
import numpy as np


def arclen(xyz):
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg + 1e-12)])


def resample(seq, K):
    """Arc-length resample a (n, C) card to K points (endpoints kept). Linear interpolation of every column."""
    s = arclen(seq[:, :3])
    t = np.linspace(0.0, s[-1], K)
    return np.stack([np.interp(t, s, seq[:, c]) for c in range(seq.shape[1])], 1)


def decode_card(res, n, root_normal=None):
    """Resampled (K,5) -> n points uniformly spaced in arc length along the K-point polyline (optionally with the
    root-normal fix of CHARM's cards, see apply_root_normal)."""
    out = resample(res, max(int(round(n)), 3))
    if root_normal is not None:
        out = apply_root_normal(out, root_normal)
    return out


def root_normal_of(seq):
    """CHARM cards (5 columns): width direction at the root, as the smallest PCA axis of the first 3 points."""
    _, vecs = np.linalg.eigh(np.cov(seq[:3, :3].T))
    return vecs[:, 0]


def apply_root_normal(seq, nrm):
    """CHARM cards (5 columns): project points 1, 2 into the plane through p0 with normal nrm (CHARM's mesher derives
    the first width direction from the plane of the first three points)."""
    seq = seq.copy()
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
    p0 = seq[0, :3]
    for k in (1, 2):
        seq[k, :3] -= np.dot(seq[k, :3] - p0, nrm) * nrm
    # if the projected points are collinear the plane is undefined: bend p2 slightly within the target plane
    a, b = seq[1, :3] - p0, seq[2, :3] - p0
    if np.linalg.norm(np.cross(a, b)) < 1e-10 * (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) + 1e-14:
        tau = b / (np.linalg.norm(b) + 1e-12)
        side = np.cross(nrm, tau)
        seq[2, :3] += 1e-3 * np.linalg.norm(b) * side
    return seq


def sign_root_normals(nrm, roots):
    """Root width directions nrm (C, 3) of one hairstyle, each flipped where it points towards the mean of the card
    roots (C, 3): the sign rule of the training tokens (hair_features), for cards that come from another model. Returns
    the signed copy and the mask of flipped cards."""
    nrm = np.array(nrm)
    flip = (nrm.astype(np.float64) * (roots - roots.mean(0))).sum(1) < 0
    nrm[flip] *= -1
    return nrm, flip


def hair_features(cards, K):
    """Per-card fixed-K features of one hairstyle: res (C,K,5) float32 (K-point arc-length resample of xyz, w, t) and
    nrm (C,3) float32 (root width direction, sign chosen to point away from the mean card root)."""
    roots = np.stack([s[0, :3] for s, _ in cards])
    c = roots.mean(0)
    res = np.zeros((len(cards), K, 5), np.float32)
    nrm = np.zeros((len(cards), 3), np.float32)
    for j, (s, _) in enumerate(cards):
        res[j] = resample(s[:, :5], K)
        v = s[0, 5:8].copy() if s.shape[1] >= 8 else root_normal_of(s)
        if np.dot(v, s[0, :3] - c) < 0:
            v = -v
        nrm[j] = v
    return res, nrm
