"""Hair spring-bone chains straight from VRM 0.x files (glTF 2.0), without Blender.

Reads the glb JSON and binary chunks; rest-pose global node transforms from the node TRS; spring-bone groups from
extensions.VRM.secondaryAnimation (stiffiness [sic], gravityPower, dragForce, hitRadius, root bones, collider groups);
every root-to-leaf path below a group's root bone is one chain (VRoid exports end nodes '..._end', so the last joint is
an explicit node). Hair meshes are the meshes whose name contains 'hair'; each hair vertex is assigned to the joint
with its largest skin weight, and a chain counts as a hair chain if at least one hair vertex is assigned to one of its
nodes. Coordinates are mapped to the CHARM dataset frame with the similarity that maps the VRM hair vertices onto the
dataset's hair.glb (axis signs searched, scale + shift from the bboxes, checked by the nearest-vertex residual).

    from hairdeck.spring.vrm_spring import extract
    rec = extract(vrm_path, hair_glb_path)    # dict, see extract()
extract_record: one line of data/spring/chains_{split}.jsonl. Driver: scripts/data.py (step chains).
"""
import json
import struct

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load_glb(path):
    b = open(path, 'rb').read()
    magic, _, _ = struct.unpack('<III', b[:12])
    assert magic == 0x46546C67, 'not a glb file'
    jl, _ = struct.unpack('<II', b[12:20])
    J = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl, _ = struct.unpack('<II', b[off:off + 8])
    return J, b[off + 8:off + 8 + bl]


def accessor(J, B, i):
    a = J['accessors'][i]
    bv = J['bufferViews'][a['bufferView']]
    dt, nc = np.dtype(COMP[a['componentType']]), NCOMP[a['type']]
    start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = bv.get('byteStride', 0) or dt.itemsize * nc
    raw = np.frombuffer(B, np.uint8, count=stride * (a['count'] - 1) + dt.itemsize * nc, offset=start)
    out = np.lib.stride_tricks.as_strided(raw, (a['count'], dt.itemsize * nc), (stride, 1)).copy().view(dt).reshape(a['count'], nc)
    if a.get('normalized'):
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def trs(n):
    if 'matrix' in n:
        return np.array(n['matrix'], np.float64).reshape(4, 4).T
    t = np.array(n.get('translation', [0, 0, 0]), np.float64)
    x, y, z, w = n.get('rotation', [0, 0, 0, 1])
    s = np.array(n.get('scale', [1, 1, 1]), np.float64)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R * s[None], t
    return M


def global_mats(J):
    nodes = J['nodes']
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    G = [None] * len(nodes)

    def g(i):
        if G[i] is None:
            G[i] = (g(parent[i]) if i in parent else np.eye(4)) @ trs(nodes[i])
        return G[i]
    for i in range(len(nodes)):
        g(i)
    return np.stack(G), parent


def chains_below(J, root):
    """Root-to-leaf node paths below a root bone."""
    out, stack = [], [[root]]
    while stack:
        p = stack.pop()
        ch = J['nodes'][p[-1]].get('children', [])
        if not ch:
            out.append(p)
        for c in ch:
            stack.append(p + [c])
    return out


def hair_vertices(J, B, G):
    """World rest positions and dominant joint node of every hair-mesh vertex."""
    V, D = [], []
    for ni, n in enumerate(J['nodes']):
        if 'mesh' not in n or 'hair' not in J['meshes'][n['mesh']].get('name', '').lower():
            continue
        skin = J['skins'][n['skin']] if 'skin' in n else None
        joints = skin['joints'] if skin else None
        ibm = accessor(J, B, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1) if skin and 'inverseBindMatrices' in skin else None
        seen = set()
        for pr in J['meshes'][n['mesh']]['primitives']:
            sig = (pr['attributes']['POSITION'], pr['attributes'].get('JOINTS_0'))
            if sig in seen:  # VRoid primitives (one per material) share one vertex buffer
                continue
            seen.add(sig)
            p = accessor(J, B, pr['attributes']['POSITION']).astype(np.float64)
            if skin is not None and 'JOINTS_0' in pr['attributes']:
                jj = accessor(J, B, pr['attributes']['JOINTS_0']).astype(int)
                ww = accessor(J, B, pr['attributes']['WEIGHTS_0']).astype(np.float64)
                k = ww.argmax(1)
                dom = np.array(joints)[jj[np.arange(len(jj)), k]]
                if ibm is not None:  # linear blend skinning at rest: sum_i w_i (G_i IBM_i) p
                    S = np.einsum('jab,jbc->jac', G[np.array(joints)], ibm)
                    ph = np.c_[p, np.ones(len(p))]
                    pw = np.zeros_like(ph)
                    for c in range(jj.shape[1]):
                        pw += ww[:, c:c + 1] * np.einsum('nab,nb->na', S[jj[:, c]], ph)
                    p = pw[:, :3]
            else:
                dom = np.full(len(p), ni)
                p = (np.c_[p, np.ones(len(p))] @ G[ni].T)[:, :3]
            V.append(p)
            D.append(dom)
    return (np.concatenate(V), np.concatenate(D)) if V else (np.zeros((0, 3)), np.zeros(0, int))


def fit_to_dataset(hv, glb_path):
    """Similarity (sx, sz signs on x / z, uniform scale, shift) mapping VRM hair vertices onto the dataset hair.glb."""
    import trimesh
    from scipy.spatial import cKDTree
    sc = trimesh.load(glb_path)
    D = np.concatenate([g.vertices for g in (sc.dump() if isinstance(sc, trimesh.Scene) else [sc])])
    tree = cKDTree(D)
    sub = hv[np.random.default_rng(0).choice(len(hv), min(3000, len(hv)), replace=False)]
    best = None
    for sx in (1, -1):
        for sz in (1, -1):
            a = hv * np.array([sx, 1, sz])
            lo, hi = a.min(0), a.max(0)
            s = float(np.median((D.max(0) - D.min(0)) / np.maximum(hi - lo, 1e-9)))
            t = (D.max(0) + D.min(0)) / 2 - s * (hi + lo) / 2
            r = tree.query(s * sub * np.array([sx, 1, sz]) + t)[0].mean()
            if best is None or r < best[0]:
                best = (float(r), sx, sz, s, t)
    r, sx, sz, s, t = best
    return dict(sx=sx, sz=sz, s=s, t=t.tolist(), resid=r, resid_rel=r / float((D.max(0) - D.min(0)).max()))


def extract(vrm_path, hair_glb=None):
    """Returns dict(chains=[dict(nodes, names, pos (n, 3) glTF frame, pos_ds (n, 3) dataset frame if hair_glb,
    stiffness, gravity, drag, hit, n_hair_verts)], n_groups, n_hair_verts, frac_hair_verts_on_chains, fit)."""
    J, B = load_glb(vrm_path)
    G, _ = global_mats(J)
    sa = J.get('extensions', {}).get('VRM', {}).get('secondaryAnimation', {})
    hv, dom = hair_vertices(J, B, G)
    node_chain, chains = {}, []
    for g in sa.get('boneGroups', []):
        prm = dict(stiffness=float(g.get('stiffiness', 1.0)), gravity=float(g.get('gravityPower', 0.0)),
                   drag=float(g.get('dragForce', 0.4)), hit=float(g.get('hitRadius', 0.02)))
        for r in g.get('bones', []):
            for path in chains_below(J, r):
                for nd in path:
                    node_chain.setdefault(nd, len(chains))
                chains.append(dict(nodes=path, names=[J['nodes'][k].get('name', '') for k in path], pos=G[path][:, :3, 3], **prm))
    cnt = np.zeros(len(chains), int)
    for d in dom:
        c = node_chain.get(int(d))
        if c is not None:
            cnt[c] += 1
    hair_chains = []
    for c, ch in enumerate(chains):
        if cnt[c] > 0:
            ch['n_hair_verts'] = int(cnt[c])
            hair_chains.append(ch)
    rec = dict(chains=hair_chains, n_groups=len(sa.get('boneGroups', [])), n_chains_all=len(chains), n_hair_verts=int(len(hv)),
               frac_hair_verts_on_chains=float(cnt.sum() / max(len(hv), 1)))
    if hair_glb is not None and len(hv):
        fit = fit_to_dataset(hv, hair_glb)
        rec['fit'] = fit
        m = np.array([fit['sx'], 1, fit['sz']])
        for ch in hair_chains:
            ch['pos_ds'] = fit['s'] * ch['pos'] * m + np.array(fit['t'])
    return rec


def extract_record(item):
    """Pool worker: (key, vrm path, hair.glb path) -> the record of data/spring/chains_{split}.jsonl: key, status, fit,
    n_groups, n_chains_all, n_hair_verts, frac_hair_verts_on_chains, chains (names, pos = pos_ds rounded to 6 digits,
    stiffness, gravity, drag, hit, n_hair_verts); status 'fail' with err on an exception."""
    key, vrm, glb = item
    try:
        r = extract(vrm, glb)
        chains = [dict(names=c['names'], pos=[[round(float(v), 6) for v in p] for p in c['pos_ds']], stiffness=c['stiffness'],
                       gravity=c['gravity'], drag=c['drag'], hit=c['hit'], n_hair_verts=c['n_hair_verts']) for c in r['chains']]
        return dict(key=key, status='ok', fit=r.get('fit'), n_groups=r['n_groups'], n_chains_all=r['n_chains_all'],
                    n_hair_verts=r['n_hair_verts'], frac_hair_verts_on_chains=r['frac_hair_verts_on_chains'], chains=chains)
    except Exception as e:  # noqa: BLE001
        return dict(key=key, status='fail', err=repr(e)[:300])
