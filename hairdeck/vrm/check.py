"""File-level checks of an exported VRM (no Blender, no browser): check(vrm_dir, prep_dir, body) -> dict with 'ok' and
'problems' (empty when every check passes).

  glb      chunk layout (magic, version 2, total length, JSON then BIN chunk, 4-byte alignment, buffer length)
  indices  every accessor inside its bufferView, every bufferView inside the buffer, every node / mesh / skin / joint /
           inverseBindMatrices / material / texture / image / sampler / scene index valid, a skin's IBM count = joints
  ext      every extension name used (root, nodes, meshes, materials, textures) is in extensionsUsed, and
           extensionsRequired is part of extensionsUsed
  vrm      sanity() (VRM 0.x: weights sum to 1, JOINTS_0 inside the skin, materialProperties count, humanoid and
           spring-group indices) on the exported file
  keep     meta, humanoid (bone -> node name), blendShapeMaster (group names, presets, binds) and firstPerson of the export
           against the body's VRM: meta equal except the thumbnail texture index and the body's vrm_meta fields (checked
           against those values); binds that pointed at the removed hair meshes are the only allowed difference
  hair     no mesh of the body's hair left (meshes named like 'hair' other than Hair_gen), the body's hair bones left
  gen      when vrm_dir holds the orig file (the VRM Add-on's unchanged re-export of the body): joint positions of
           every generated chain vs the prep (Blender and dataset frame), spring-group roots and paths, stiffness, skin
           weights per vertex vs the geometric skinning (matched by position), kept non-hair groups and colliders equal
           to the re-export

python -m hairdeck.vrm.check VRM_DIR PREP_DIR [BODY_JSON]      -> JSON on stdout
"""
import json
import os
import struct

import numpy as np

from ..spring import vrm_spring as VS
from .body import body as load_body, file_name

ITEM = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_GLB = {}


def load_glb(path):
    """vrm_spring.load_glb, remembered for the files that are the same for every export (the body's VRM and its
    re-export); a generated *_gen.vrm is always read fresh."""
    if str(path).endswith('_gen.vrm'):
        return VS.load_glb(path)
    key = (os.path.realpath(path), os.path.getmtime(path))
    if key not in _GLB:
        _GLB[key] = VS.load_glb(path)
    return _GLB[key]


def glb_layout(path):
    b = open(path, 'rb').read()
    bad = []
    magic, ver, total = struct.unpack('<III', b[:12])
    if magic != 0x46546C67:
        bad.append('magic')
    if ver != 2:
        bad.append(f'version {ver}')
    if total != len(b):
        bad.append(f'header length {total} != file size {len(b)}')
    off, chunks = 12, []
    while off < len(b):
        ln, ty = struct.unpack('<II', b[off:off + 8])
        chunks.append((ty, ln))
        if ln % 4:
            bad.append(f'chunk at {off} length {ln} not 4-aligned')
        off += 8 + ln
    if off != len(b):
        bad.append('chunks do not end at the file end')
    if not chunks or chunks[0][0] != 0x4E4F534A:
        bad.append('first chunk is not JSON')
    if len(chunks) < 2 or chunks[1][0] != 0x004E4942:
        bad.append('second chunk is not BIN')
    J, B = load_glb(path)
    if J.get('buffers') and J['buffers'][0]['byteLength'] > len(B):
        bad.append('buffer 0 longer than the BIN chunk')
    return J, B, dict(size=len(b), chunks=[(hex(t), n) for t, n in chunks]), bad


def indices(J, B):
    bad = []
    n = {k: len(J.get(k, [])) for k in ('nodes', 'meshes', 'skins', 'accessors', 'bufferViews', 'materials', 'textures',
                                        'images', 'samplers', 'buffers', 'scenes')}
    blen = [bf['byteLength'] for bf in J.get('buffers', [])]
    for i, bv in enumerate(J.get('bufferViews', [])):
        if not 0 <= bv['buffer'] < n['buffers'] or bv.get('byteOffset', 0) + bv['byteLength'] > blen[bv['buffer']]:
            bad.append(f'bufferView {i} outside its buffer')
    for i, a in enumerate(J.get('accessors', [])):
        if 'bufferView' not in a:
            continue
        if not 0 <= a['bufferView'] < n['bufferViews']:
            bad.append(f'accessor {i} bufferView index')
            continue
        bv = J['bufferViews'][a['bufferView']]
        es = ITEM[a['componentType']] * VS.NCOMP[a['type']]
        stride = bv.get('byteStride', 0) or es
        end = a.get('byteOffset', 0) + stride * (a['count'] - 1) + es
        if end > bv['byteLength']:
            bad.append(f'accessor {i} ends at {end} > bufferView length {bv["byteLength"]}')
    for i, nd in enumerate(J['nodes']):
        if 'mesh' in nd and not 0 <= nd['mesh'] < n['meshes']:
            bad.append(f'node {i} mesh index')
        if 'skin' in nd and not 0 <= nd['skin'] < n['skins']:
            bad.append(f'node {i} skin index')
        if not all(0 <= c < n['nodes'] for c in nd.get('children', [])):
            bad.append(f'node {i} child index')
    parents = [c for nd in J['nodes'] for c in nd.get('children', [])]
    if len(parents) != len(set(parents)):
        bad.append('a node has two parents')
    for s in J.get('scenes', []):
        if not all(0 <= c < n['nodes'] for c in s.get('nodes', [])):
            bad.append('scene node index')
    if 'scene' in J and not 0 <= J['scene'] < n['scenes']:
        bad.append('default scene index')
    for i, sk in enumerate(J.get('skins', [])):
        if not all(0 <= j < n['nodes'] for j in sk['joints']):
            bad.append(f'skin {i} joint index')
        if 'skeleton' in sk and not 0 <= sk['skeleton'] < n['nodes']:
            bad.append(f'skin {i} skeleton index')
        if 'inverseBindMatrices' in sk:
            ib = sk['inverseBindMatrices']
            if not 0 <= ib < n['accessors'] or J['accessors'][ib]['count'] != len(sk['joints']):
                bad.append(f'skin {i} inverseBindMatrices')
    for m in J.get('meshes', []):
        for pr in m['primitives']:
            if not all(0 <= v < n['accessors'] for v in pr['attributes'].values()):
                bad.append(f'{m.get("name")}: attribute accessor index')
            if 'indices' in pr and not 0 <= pr['indices'] < n['accessors']:
                bad.append(f'{m.get("name")}: indices accessor index')
            if 'material' in pr and not 0 <= pr['material'] < n['materials']:
                bad.append(f'{m.get("name")}: material index')
            for t in pr.get('targets', []):
                if not all(0 <= v < n['accessors'] for v in t.values()):
                    bad.append(f'{m.get("name")}: morph target accessor index')
            if 'indices' in pr:
                idx = VS.accessor(J, B, pr['indices'])
                nv = J['accessors'][pr['attributes']['POSITION']]['count']
                if idx.size and idx.max() >= nv:
                    bad.append(f'{m.get("name")}: vertex index {idx.max()} >= {nv}')
    for i, t in enumerate(J.get('textures', [])):
        if 'source' in t and not 0 <= t['source'] < n['images']:
            bad.append(f'texture {i} image index')
        if 'sampler' in t and not 0 <= t['sampler'] < n['samplers']:
            bad.append(f'texture {i} sampler index')
    for i, im in enumerate(J.get('images', [])):
        if 'bufferView' in im and not 0 <= im['bufferView'] < n['bufferViews']:
            bad.append(f'image {i} bufferView index')

    def tex_refs(o):  # textureInfo objects: {'index': int, 'texCoord', 'scale', 'strength', 'extensions', 'extras'}
        if isinstance(o, dict):
            if isinstance(o.get('index'), int) and set(o) <= {'index', 'texCoord', 'scale', 'strength', 'extensions', 'extras'}:
                yield o['index']
            for k, v in o.items():
                if k != 'index':
                    yield from tex_refs(v)
        elif isinstance(o, list):
            for v in o:
                yield from tex_refs(v)
    for i, m in enumerate(J.get('materials', [])):
        if not all(0 <= t < n['textures'] for t in tex_refs(m)):
            bad.append(f'material {i} texture index')
    return n, bad


def extensions(J):
    used = set()
    for o in [J] + J.get('nodes', []) + J.get('meshes', []) + J.get('materials', []) + J.get('textures', []) + J.get('images', []):
        used |= set(o.get('extensions', {}))
    for m in J.get('materials', []):
        for k in ('pbrMetallicRoughness', 'normalTexture', 'emissiveTexture', 'occlusionTexture'):
            v = m.get(k, {})
            used |= set(v.get('extensions', {})) if isinstance(v, dict) else set()
            if isinstance(v, dict):
                for t in v.values():
                    if isinstance(t, dict):
                        used |= set(t.get('extensions', {}))
    declared = set(J.get('extensionsUsed', []))
    req = set(J.get('extensionsRequired', []))
    bad = [f'{e} used but not in extensionsUsed' for e in sorted(used - declared)]
    bad += [f'{e} required but not used' for e in sorted(req - declared)]
    return sorted(used), sorted(declared), sorted(req), bad


def sanity(J, B):
    """Structural checks a strict VRM 0.x / glTF 2.0 loader relies on; returns a list of problems (empty = fine)."""
    bad = []
    n_nodes, n_acc = len(J['nodes']), len(J['accessors'])
    if 'VRM' not in J.get('extensionsUsed', []):
        bad.append('VRM not in extensionsUsed')
    for i, bv in enumerate(J['bufferViews']):
        if bv.get('byteOffset', 0) + bv['byteLength'] > len(B):
            bad.append(f'bufferView {i} out of range')
    for i, a in enumerate(J['accessors']):
        if a.get('bufferView', -1) >= len(J['bufferViews']):
            bad.append(f'accessor {i} bad bufferView')
    for m in J['meshes']:
        for pr in m['primitives']:
            pa = pr['attributes']['POSITION']
            if 'min' not in J['accessors'][pa] or 'max' not in J['accessors'][pa]:
                bad.append(f'{m["name"]}: POSITION without min/max')
            if not all(0 <= v < n_acc for v in pr['attributes'].values()) or pr.get('indices', 0) >= n_acc:
                bad.append(f'{m["name"]}: bad accessor index')
            if pr.get('material', 0) >= len(J.get('materials', [])):
                bad.append(f'{m["name"]}: bad material index')
    for n in J['nodes']:
        if 'skin' in n:
            sk = J['skins'][n['skin']]
            if not all(0 <= j < n_nodes for j in sk['joints']):
                bad.append('skin joint out of range')
            pr = J['meshes'][n['mesh']]['primitives'][0]
            if 'JOINTS_0' in pr['attributes']:
                jj = VS.accessor(J, B, pr['attributes']['JOINTS_0'])
                ww = VS.accessor(J, B, pr['attributes']['WEIGHTS_0']).astype(np.float64)
                if jj.max() >= len(sk['joints']):
                    bad.append(f'{J["meshes"][n["mesh"]]["name"]}: JOINTS_0 beyond skin joints')
                if np.abs(ww.sum(1) - 1).max() > 1e-3:
                    bad.append(f'{J["meshes"][n["mesh"]]["name"]}: weights do not sum to 1 ({np.abs(ww.sum(1) - 1).max():.4f})')
    vrm = J['extensions']['VRM']
    if len(vrm.get('materialProperties', [])) != len(J.get('materials', [])):
        bad.append('materialProperties count != materials count')
    for hb in vrm.get('humanoid', {}).get('humanBones', []):
        if not 0 <= hb['node'] < n_nodes:
            bad.append(f'humanoid {hb["bone"]} bad node')
    sa = vrm.get('secondaryAnimation', {})
    for g in sa.get('boneGroups', []):
        if not all(0 <= b < n_nodes for b in g.get('bones', [])) or not all(0 <= c < len(sa.get('colliderGroups', [])) for c in g.get('colliderGroups', [])):
            bad.append('bone group with bad node / collider group index')
        if not g.get('bones'):
            bad.append('bone group without bones')
    return bad


def keep_checks(Js, Jg, removed_mesh_names, meta_override=None):
    vs, vg = Js['extensions']['VRM'], Jg['extensions']['VRM']
    out, bad = {}, []
    ms, mg = dict(vs.get('meta', {})), dict(vg.get('meta', {}))
    ms.update(meta_override or {})  # fields the body sets on export (its vrm_meta) are checked against those values
    ms.pop('texture', None)
    mg.pop('texture', None)
    # a field the source leaves out is written as '' by the add-on (also in its plain re-export): treated as equal
    diff = sorted(k for k in set(ms) | set(mg) if ms.get(k, '') != mg.get(k, ''))
    out['meta_fields'], out['meta_diff'] = len(ms), diff
    if diff:
        bad.append(f'meta differs: {diff}')
    hn = lambda J, v: {h['bone']: J['nodes'][h['node']]['name'] for h in v.get('humanoid', {}).get('humanBones', [])}  # noqa: E731
    hs, hg = hn(Js, vs), hn(Jg, vg)
    out['humanoid_bones'] = [len(hs), len(hg)]
    if hs != hg:
        bad.append(f'humanoid differs: {sorted(k for k in set(hs) | set(hg) if hs.get(k) != hg.get(k))[:8]}')

    def groups(J, v):
        res = []
        for g in v.get('blendShapeMaster', {}).get('blendShapeGroups', []):
            binds = sorted((J['meshes'][b['mesh']]['name'], b['index'], round(b['weight'], 3)) for b in g.get('binds', []))
            res.append((g.get('name'), g.get('presetName'), binds))
        return res
    gs, gg = groups(Js, vs), groups(Jg, vg)
    out['blend_shape_groups'] = [len(gs), len(gg)]
    strip = lambda G: [(a, b, [x for x in c if not any(r.split('.')[0] in x[0] for r in removed_mesh_names)]) for a, b, c in G]  # noqa: E731
    same_names = [(a, b) for a, b, _ in gs] == [(a, b) for a, b, _ in gg]
    nb_s, nb_g = sum(len(c) for _, _, c in gs), sum(len(c) for _, _, c in gg)
    out['blend_shape_binds'] = [nb_s, nb_g]
    out['blend_shape_binds_equal_after_removing_hair_mesh_binds'] = strip(gs) == strip(gg)
    if not same_names:
        bad.append('blendShapeMaster group names or presets differ')
    fp_s, fp_g = vs.get('firstPerson', {}), vg.get('firstPerson', {})
    out['first_person_bone'] = [Js['nodes'][fp_s['firstPersonBone']]['name'] if 'firstPersonBone' in fp_s else None,
                                Jg['nodes'][fp_g['firstPersonBone']]['name'] if 'firstPersonBone' in fp_g else None]
    if out['first_person_bone'][0] != out['first_person_bone'][1]:
        bad.append('firstPersonBone differs')
    return out, bad


# ------------------------------------------------------------ round trip of the generated hair (the 'gen' checks)
def gltf_to_blender(g):  # VRM 0.x as imported by the VRM add-on (faces -Y): blender = (-gx, gz, gy)
    g = np.asarray(g, np.float64)
    return np.stack([-g[..., 0], g[..., 2], g[..., 1]], -1)


def blender_to_ds(b, P):  # inverse of hairdeck.vrm.prep.to_blender
    a = np.stack([P['fit_sx'] * b[..., 0], b[..., 2], P['fit_sz'] * b[..., 1]], -1)
    return a * P['fit_s'] + P['fit_t']


def groups(J):
    """Spring groups: root node names and rest positions (glTF frame), params, collider node names, chains (node paths)."""
    sa = J['extensions']['VRM'].get('secondaryAnimation', {})
    cg = sa.get('colliderGroups', [])
    G, _ = VS.global_mats(J)
    out = []
    for g in sa.get('boneGroups', []):
        out.append(dict(roots=[J['nodes'][r]['name'] for r in g.get('bones', [])], root_pos=G[g.get('bones', [])][:, :3, 3],
                        stiffness=g.get('stiffiness'), drag=g.get('dragForce'), gravity=g.get('gravityPower'), hit=g.get('hitRadius'),
                        comment=g.get('comment', ''), colliders=sorted(J['nodes'][cg[i]['node']]['name'] for i in g.get('colliderGroups', [])),
                        chains=[p for r in g.get('bones', []) for p in VS.chains_below(J, r)]))
    return out


def same_groups(A, B):
    """Groups equal up to float noise (node names may differ: Blender truncates bone names to 63 characters).
    Returns (equal, max root position difference in mm)."""
    if len(A) != len(B):
        return False, None
    ok, err = True, 0.0
    for a, b in zip(A, B):
        if len(a['roots']) != len(b['roots']) or a['colliders'] != b['colliders'] or len(a['chains']) != len(b['chains']):
            ok = False
            continue
        ok &= all(abs(a[k] - b[k]) < 1e-6 for k in ('stiffness', 'drag', 'gravity', 'hit'))
        if len(a['roots']):
            err = max(err, float(np.abs(a['root_pos'] - b['root_pos']).max() * 1000))
    return ok and err < 0.01, err


def colliders(J):
    sa = J['extensions']['VRM'].get('secondaryAnimation', {})
    out = {}
    for c in sa.get('colliderGroups', []):
        nm = J['nodes'][c['node']]['name']
        out[nm] = np.array([[k['radius']] + [k['offset'][a] for a in 'xyz'] for k in c['colliders']], np.float64)
    return out


def same_colliders(A, B):
    """(equal up to 0.01 mm, max radius / offset difference in mm)."""
    if set(A) != set(B) or any(A[k].shape != B[k].shape for k in A):
        return False, None
    err = max((float(np.abs(A[k] - B[k]).max() * 1000) for k in A if len(A[k])), default=0.0)
    return err < 0.01, err


def mesh_skin(J, B, name):
    """Rest positions (glTF), joint node names and weights of every vertex of the named mesh (first primitive set)."""
    for ni, n in enumerate(J['nodes']):
        if 'mesh' in n and J['meshes'][n['mesh']]['name'] == name:
            skin = J['skins'][n['skin']]
            pr = J['meshes'][n['mesh']]['primitives'][0]
            p = VS.accessor(J, B, pr['attributes']['POSITION']).astype(np.float64)
            jj = VS.accessor(J, B, pr['attributes']['JOINTS_0']).astype(int)
            ww = VS.accessor(J, B, pr['attributes']['WEIGHTS_0']).astype(np.float64)
            names = np.array([J['nodes'][skin['joints'][k]]['name'] for k in range(len(skin['joints']))])
            return p, names[jj], ww, len(J['meshes'][n['mesh']]['primitives'])
    return None


def gen_checks(b, vrm_dir, prep_dir):
    """The exported hair and chains against the prep and against the add-on's re-export of the body (the orig file):
    every generated chain from its bone nodes (HairGen_cCC_KK ..., HairGen_cCC_end; rigid prefix included), the spring
    groups below a HairGen root, the skin weights of Hair_gen, the kept groups and colliders. Returns (gen dict,
    [sanity of the body's VRM, sanity of the re-export])."""
    from scipy.spatial import cKDTree
    src, orig, gen = b['vrm'], f"{vrm_dir}/{file_name('orig', b)}", f"{vrm_dir}/{file_name('gen', b)}"
    P = dict(np.load(f"{prep_dir}/{file_name('prep', b)}.npz"))
    Js, Bs = load_glb(src)
    Jo, Bo = load_glb(orig)
    sanity_src, sanity_orig = sanity(Js, Bs), sanity(Jo, Bo)
    co, go = colliders(Jo), groups(Jo)
    Jg, Bg = load_glb(gen)
    gg = groups(Jg)
    gen_groups = [g for g in gg if g['roots'] and g['roots'][0].startswith('HairGen')]
    kept = [g for g in gg if not (g['roots'] and g['roots'][0].startswith('HairGen'))]
    ex = json.load(open(f"{vrm_dir}/{file_name('export', b)}"))
    removed = {tuple(g['roots']) for g in ex['removed_groups']}
    nonhair_orig = [g for g in go if tuple(g['roots']) not in removed]
    G, _ = VS.global_mats(Jg)
    name2node = {n.get('name'): i for i, n in enumerate(Jg['nodes'])}
    ptr = P['chain_ptr']
    nch = len(ptr) - 1
    errs_b, errs_ds, nj_ok = [], [], 0
    for c in range(nch):
        Jin, Jds = P['chain_J'][ptr[c]:ptr[c + 1]], P['chain_J_ds'][ptr[c]:ptr[c + 1]]
        nodes = [name2node.get(f'HairGen_c{c:02d}_{k:02d}') for k in range(len(Jin) - 1)] + [name2node.get(f'HairGen_c{c:02d}_end')]
        if any(n is None for n in nodes):
            continue
        nj_ok += 1
        pos_b = gltf_to_blender(G[nodes][:, :3, 3])
        errs_b.append(np.linalg.norm(pos_b - Jin, axis=1).max())
        errs_ds.append(np.linalg.norm(blender_to_ds(pos_b, P) - Jds, axis=1).max() / P['fit_s'])
    # spring groups: root HairGen_cCC_KK, its single chain must be nodes KK .. end of chain CC
    grp_ok, stiff_err, n_sim_bones, roots = 0, 0.0, 0, {}
    for g in gen_groups:
        nm = g['roots'][0]
        c, k = int(nm.split('_c')[1][:2]), int(nm.split('_')[-1])
        roots[c] = k
        path = [Jg['nodes'][i]['name'] for i in g['chains'][0]] if len(g['chains']) == 1 else []
        want = [f'HairGen_c{c:02d}_{j:02d}' for j in range(k, ptr[c + 1] - ptr[c] - 1)] + [f'HairGen_c{c:02d}_end']
        grp_ok += path == want
        n_sim_bones += len(want) - 1
        stiff_err = max(stiff_err, abs(g['stiffness'] - float(P['stiffness'][c])))
    hs = mesh_skin(Jg, Bg, 'Hair_gen')
    wcmp = None
    if hs is not None:
        p, jn, ww, nprim = hs
        pb = gltf_to_blender(p)
        d, idx = cKDTree(P['V']).query(pb)
        names = {c: [f'HairGen_c{c:02d}_{k:02d}' for k in range(ptr[c + 1] - ptr[c] - 1)] for c in range(nch)}
        sg2 = P['vseg2'] if 'vseg2' in P else P['vseg']
        al = P['valpha'] if 'valpha' in P else np.ones(len(P['vseg']))
        err, nmax = 0.0, 0
        for i in range(len(pb)):  # expected: head 1 - w, segment k w * alpha, neighbour segment w * (1 - alpha)
            q = idx[i]
            exp = {}
            c, w = int(P['vchain'][q]), float(P['vw'][q])
            if c < 0 or w <= 0:
                exp[ex['head_bone']] = 1.0
            else:
                two = sg2[q] != P['vseg'][q] and al[q] < 1
                exp[names[c][P['vseg'][q]]] = w * (al[q] if two else 1.0)
                if two:
                    exp[names[c][sg2[q]]] = exp.get(names[c][sg2[q]], 0.0) + w * (1 - al[q])
                if w < 1:
                    exp[ex['head_bone']] = 1 - w
            got = {}
            for n, wt in zip(jn[i], ww[i]):
                if wt > 0:
                    got[n] = got.get(n, 0.0) + float(wt)
            nmax = max(nmax, len(got))
            err = max(err, max(abs(exp.get(n, 0.0) - got.get(n, 0.0)) for n in set(exp) | set(got)))
        chain_w = np.array([sum(w for n, w in zip(jn[i], ww[i]) if n.startswith('HairGen')) for i in range(len(pb))])
        head_w = np.array([sum(w for n, w in zip(jn[i], ww[i]) if n == ex['head_bone']) for i in range(len(pb))])
        wcmp = dict(n_vertices_exported=len(pb), primitives=nprim, pos_err_mm_max=float(d.max() * 1000), per_bone_weight_err_max=err,
                    max_influences=nmax, weight_sum_err_max=float(np.abs(chain_w + head_w - 1).max()),
                    frac_verts_with_chain_weight=float((chain_w > 0).mean()))
    gen = dict(chains_in=nch, chains_with_all_bone_nodes=nj_ok,
               joint_err_mm_max_blender=float(max(errs_b) * 1000) if errs_b else None,
               joint_err_mm_max_dataset=float(max(errs_ds) * 1000) if errs_ds else None,
               spring_groups_gen=len(gen_groups), spring_groups_expected=int(ex.get('n_spring_groups', nch)),
               groups_root_and_path_ok=grp_ok,
               roots_match_export=all(roots.get(c) == k for c, k in enumerate(ex.get('spring_root_index', [])) if k is not None),
               simulated_bones=n_sim_bones, stiffness_err_max=stiff_err, kept_groups=len(kept),
               kept_groups_equal_control=same_groups(kept, nonhair_orig), colliders_equal_control=same_colliders(colliders(Jg), co),
               skin=wcmp, prep_verts=int(len(P['V'])))
    return gen, [sanity_src, sanity_orig]


def check(vrm_dir, prep_dir, body=None):
    """All checks of one export (vrm_dir: the gen and export files and, for the 'gen' checks, the orig file; prep_dir:
    the prep npz; names: hairdeck.vrm.body.file_name). body: a body dict or body file (default the default body)."""
    b = body if isinstance(body, dict) else load_body(body)
    src, gen = b['vrm'], f"{vrm_dir}/{file_name('gen', b)}"
    R, problems = {}, []
    Jg, Bg, R['glb'], bad = glb_layout(gen)
    problems += bad
    R['counts'], bad = indices(Jg, Bg)
    problems += bad
    R['extensions_used'], R['extensions_declared'], R['extensions_required'], bad = extensions(Jg)
    problems += bad
    bad = sanity(Jg, Bg)
    problems += [f'sanity: {x}' for x in bad]
    Js, Bs = load_glb(src)
    ex = json.load(open(f"{vrm_dir}/{file_name('export', b)}"))
    removed = [m for _, m, _ in ex['hair_objects']] + [o for o, _, _ in ex['hair_objects']]
    R['keep'], bad = keep_checks(Js, Jg, removed, b.get('vrm_meta'))
    problems += bad
    src_mesh = [m['name'] for m in Js['meshes']]
    gen_mesh = [m['name'] for m in Jg['meshes']]
    hair_left = [m for m in gen_mesh if 'hair' in m.lower() and m != 'Hair_gen']
    src_hair_nodes = {n.get('name') for n in Js['nodes'] if 'hair' in n.get('name', '').lower()}
    gen_names = {n.get('name') for n in Jg['nodes']}
    R['hair'] = dict(source_meshes=src_mesh, export_meshes=gen_mesh, hair_meshes_left=hair_left,
                     source_hair_named_nodes=len(src_hair_nodes), source_hair_named_nodes_left=sorted(src_hair_nodes & gen_names),
                     protected_by_exporter=ex.get('bones_protected_candidates', []), hair_gen_mesh=('Hair_gen' in gen_mesh))
    if hair_left or 'Hair_gen' not in gen_mesh:
        problems.append(f'hair meshes: left {hair_left}, Hair_gen present {"Hair_gen" in gen_mesh}')
    unexpected = sorted((src_hair_nodes & gen_names) - set(ex.get('bones_protected_candidates', [])))
    if unexpected:
        problems.append(f'source hair bones left that the exporter did not protect: {unexpected[:5]}')
    if os.path.exists(f"{vrm_dir}/{file_name('orig', b)}"):
        R['gen'], R['control_sanity'] = gen_checks(b, vrm_dir, prep_dir)
        g = R['gen']
        if g['spring_groups_gen'] != g['spring_groups_expected'] or g['groups_root_and_path_ok'] != g['spring_groups_gen']:
            problems.append('spring groups do not match the export')
        if g['chains_with_all_bone_nodes'] != g['chains_in']:
            problems.append('chain bone nodes missing')
        if g['joint_err_mm_max_blender'] is not None and g['joint_err_mm_max_blender'] > 0.1:
            problems.append(f'joint error {g["joint_err_mm_max_blender"]:.3f} mm')
        if not g['kept_groups_equal_control'][0] or not g['colliders_equal_control'][0]:
            problems.append('kept spring groups or colliders differ from the re-exported original')
        sk = g['skin'] or {}
        if sk and (sk['per_bone_weight_err_max'] > 1e-3 or sk['weight_sum_err_max'] > 1e-3 or sk['pos_err_mm_max'] > 0.1):
            problems.append(f'skin weights / positions differ from the prep: {sk}')
    R['problems'] = problems
    R['ok'] = not problems
    return R


if __name__ == '__main__':
    import sys
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    print(json.dumps(check(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), default=str))
