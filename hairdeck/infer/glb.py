"""GLB output of the image pipeline: the cards as one mesh (vectorised card mesher, per-card UVs, hair colour), the
head proxy and the spring-bone chains as separate meshes of one scene (y up, face toward +z, metres)."""
import numpy as np
import trimesh

from ..cards.mesh import card_uv
from ..vis.head_proxy import head_proxy


def hair_color(img, hair):
    """Median colour of the darker half of the hair pixels (sRGB, 0..1)."""
    if not hair.any():
        return [0.4, 0.3, 0.25]
    px = img[hair].astype(np.float64)
    lum = px @ np.array([0.299, 0.587, 0.114])
    return (np.median(px[lum <= np.median(lum)], 0) / 255).tolist()


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _cross(a, b):
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def _unit(v):
    return v / (np.sqrt((v * v).sum(-1, keepdims=True)) + 1e-12)


_STENCIL = np.array([-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12])


def _ring_faces(n):
    f = [[0, 2, 1], [0, 3, 2]]
    for i in range(1, n):
        l = 4 * i
        f += [[l, l - 4, l + 1], [l + 1, l - 3, l + 2], [l + 2, l - 2, l + 3], [l + 3, l - 1, l], [l - 3, l + 1, l - 4],
              [l - 4, l, l - 1], [l - 1, l + 3, l - 2], [l - 2, l + 2, l - 3], [l, l + 1, l + 2], [l, l + 2, l + 3]]
    return np.array(f, np.int64)


def cards_vf(cards):
    """hairdeck.cards.mesh.card_mesh_vf (bishop_mesh) for every card, vectorised over the cards of equal point count: same
    5-point-stencil tangents, Bishop transport of the root direction, ring vertices and faces (about 20 times faster
    than the per-card loop; vertices agree to float64 rounding, faces are identical). Returns a list of (vertices,
    faces) in card order (None for < 3 points)."""
    out = [None] * len(cards)
    groups = {}
    for j, (seq, sh) in enumerate(cards):
        if len(seq) >= 3:
            groups.setdefault(len(seq), []).append(j)
    for n, js in groups.items():
        S8 = np.stack([np.asarray(cards[j][0], np.float64) for j in js])  # (C, n, 8)
        P = S8[:, :, :3]
        if n < 5:
            T = np.repeat(_unit(P[:, -1] - P[:, 0])[:, None], n, 1)
        else:
            T = np.empty_like(P)
            for i in range(n):
                lo = min(max(i - 2, 0), n - 5)
                T[:, i] = _unit((_STENCIL[None, :, None] * P[:, lo:lo + 5]).sum(1))
        N = np.empty_like(P)
        v0 = S8[:, 0, 5:8]
        N[:, 0] = _unit(v0 - (v0 * T[:, 0]).sum(-1, keepdims=True) * T[:, 0])
        for i in range(1, n):
            a, b = T[:, i - 1], T[:, i]
            ax = _cross(a, b)
            s = np.sqrt((ax * ax).sum(-1, keepdims=True))
            c = np.clip((a * b).sum(-1, keepdims=True), -1, 1)
            m = N[:, i - 1]
            k = ax / np.where(s > 1e-9, s, 1.0)
            rot = m * c + _cross(k, m) * s + k * (k * m).sum(-1, keepdims=True) * (1 - c)
            m = np.where(s > 1e-9, rot, m)
            N[:, i] = _unit(m - (m * b).sum(-1, keepdims=True) * b)
        Nn = _unit(N)
        tg = _unit(T - (T * Nn).sum(-1, keepdims=True) * Nn)
        t2 = _cross(Nn, tg)
        w, th = S8[:, :, 3:4], S8[:, :, 4:5]
        tri = np.array([cards[j][1] == 1 for j in js])[:, None, None]
        up, down = np.where(tri, 3.0, 2.0), np.where(tri, 6.0, 2.0)
        V = np.stack([P + w / 2 * Nn, P + th / up * t2, P - w / 2 * Nn, P - th / down * t2], 2).reshape(len(js), 4 * n, 3)
        F = _ring_faces(n)
        for q, j in enumerate(js):
            out[j] = (V[q], F)
    return out


def hair_trimesh(cards, color_srgb):
    """All cards as one mesh: hairdeck.cards.mesh.card_mesh_vf's mesh per card (cards_vf; cards with < 3 points skipped
    and degenerate faces dropped, as cards_to_mesh does), per-card UVs (card_uv), a PBR material with the hair colour
    (glTF wants linear colour) and a per-card brightness in COLOR_0 (0.78..1, fixed seed) so neighbouring cards stay
    visible."""
    rng = np.random.default_rng(0)
    V, F, UV, C = [], [], [], []
    off = 0
    for (seq, sh), vf in zip(cards, cards_vf(cards)):
        if vf is None:
            continue
        v, f = vf
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        f = f[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-14]
        k = float(np.clip(rng.normal(0.93, 0.07), 0.78, 1.0))
        V.append(v)
        F.append(f + off)
        UV.append(card_uv(seq))
        C.append(np.tile(np.array([k, k, k, 1.0]), (len(v), 1)))
        off += len(v)
    if not V:
        return None
    mat = trimesh.visual.material.PBRMaterial(
        name='hair', baseColorFactor=[*srgb_to_linear(color_srgb).tolist(), 1.0], metallicFactor=0.0,
        roughnessFactor=0.75, doubleSided=True)
    m = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=False,
                        visual=trimesh.visual.TextureVisuals(uv=np.concatenate(UV), material=mat))
    m.visual.vertex_attributes['color'] = (np.concatenate(C) * 255 + 0.5).astype(np.uint8)
    return m


def head_trimesh():
    m = head_proxy()
    m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name='head', baseColorFactor=[0.42, 0.42, 0.42, 1.0], metallicFactor=0.0, roughnessFactor=0.9))
    _ = m.vertex_normals  # cached -> exported as NORMAL (smooth head); the cards get none (flat)
    return m


def chain_trimesh(chains, radius=0.0022):
    """Spring-bone chains as thin tubes along the joint polyline plus a small sphere per joint (orange), one mesh."""
    parts = []
    for c in chains:
        J = np.asarray(c['joints'], np.float64)
        for a, b in zip(J[:-1], J[1:]):
            if np.linalg.norm(b - a) > 1e-6:
                parts.append(trimesh.creation.cylinder(radius=radius, segment=np.stack([a, b]), sections=8))
        for q in J:
            s = trimesh.creation.icosphere(subdivisions=1, radius=1.9 * radius)
            s.apply_translation(q)
            parts.append(s)
    if not parts:
        return None
    m = trimesh.util.concatenate(parts)
    m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name='spring_chains', baseColorFactor=[0.95, 0.30, 0.03, 1.0], metallicFactor=0.0, roughnessFactor=0.6))
    return m


def write_glb(path, hair, head=None, chains=None):
    sc = trimesh.Scene()
    sc.add_geometry(hair, node_name='hair_cards', geom_name='hair_cards')
    if head is not None:
        sc.add_geometry(head, node_name='head_proxy', geom_name='head_proxy')
    if chains is not None:
        sc.add_geometry(chains, node_name='spring_chains', geom_name='spring_chains')
    sc.export(path)
