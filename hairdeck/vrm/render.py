"""Rest-pose renders of VRM 0.x files with the skin applied (pyrender; EGL, or OSMesa on CPU): every mesh node's
vertices go through its skin at rest (sum over influences of weight x joint global matrix x inverse bind matrix;
unskinned meshes use the node's global matrix), so what is drawn is what a VRM viewer shows in the rest pose. Generated
hair (mesh Hair_gen) in its material colour, other hair (meshes named like 'hair') in orange, everything else light grey.
One row per file: whole body from the front, then the head from the front, the side and the back.

PYOPENGL_PLATFORM=egl|osmesa python -m hairdeck.vrm.render OUT.png VRM [VRM ...]
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from ..spring import vrm_spring as VS

BG = 236


def _pyrender():
    """pyrender, imported on first use (PYOPENGL_PLATFORM defaults to egl)."""
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import pyrender
    return pyrender


def linear_to_srgb(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def skinned_meshes(path):
    """Rest-pose meshes of a VRM (dicts name, V, F, colour = base colour factor) and the head bone position (glTF)."""
    J, B = VS.load_glb(path)
    G, _ = VS.global_mats(J)
    out = []
    for ni, nd in enumerate(J['nodes']):
        if 'mesh' not in nd:
            continue
        m = J['meshes'][nd['mesh']]
        sk = J['skins'][nd['skin']] if 'skin' in nd else None
        if sk is not None:
            ibm = VS.accessor(J, B, sk['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
            M = np.stack([G[j] for j in sk['joints']]) @ ibm
        for pr in m['primitives']:
            p = VS.accessor(J, B, pr['attributes']['POSITION']).astype(np.float64)
            f = VS.accessor(J, B, pr['indices']).reshape(-1, 3).astype(np.int64)
            ph = np.concatenate([p, np.ones((len(p), 1))], 1)
            if sk is not None and 'JOINTS_0' in pr['attributes']:
                jj = VS.accessor(J, B, pr['attributes']['JOINTS_0']).astype(int)
                ww = VS.accessor(J, B, pr['attributes']['WEIGHTS_0']).astype(np.float64)
                X = np.einsum('vk,vkij,vj->vi', ww, M[jj], ph)[:, :3]
            else:
                X = (ph @ G[ni].T)[:, :3]
            mat = J['materials'][pr['material']] if 'material' in pr else {}
            col = mat.get('pbrMetallicRoughness', {}).get('baseColorFactor', [0.8, 0.8, 0.8, 1])[:3]
            out.append(dict(name=m.get('name', ''), V=X, F=f, colour=col))
    hb = {h['bone']: h['node'] for h in J['extensions']['VRM']['humanoid']['humanBones']}
    return out, G[hb['head']][:3, 3]


def colour_of(m):
    if m['name'] == 'Hair_gen':
        return linear_to_srgb(m['colour'])
    if 'hair' in m['name'].lower():
        return np.array([1.0, 0.55, 0.1])
    return np.array([0.86, 0.84, 0.82])


def look(eye, target, up=(0, 1, 0)):
    """Camera / light pose at eye looking at target (OpenGL convention: -z forward)."""
    z = np.asarray(eye, np.float64) - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    P = np.eye(4)
    P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, np.cross(z, x), z, eye
    return P


def render(meshes, target, direction, half, size=360):
    """One orthographic view (half = half the view height) from target + direction, three directional lights."""
    pyrender = _pyrender()
    s = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3] * 3)
    for m in meshes:
        tm = trimesh.Trimesh(m['V'], m['F'], process=False)
        s.add(pyrender.Mesh.from_trimesh(tm, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*colour_of(m), 1.0], metallicFactor=0.0, roughnessFactor=0.85, doubleSided=True), smooth=False))
    d = np.asarray(direction, np.float64)
    d /= np.linalg.norm(d)
    eye = target + d * 6 * half
    pose = look(eye, target)
    s.add(pyrender.OrthographicCamera(xmag=half, ymag=half, znear=0.01, zfar=20 * half), pose=pose)
    for dd, k in ((d + np.array([0.5, 0.7, 0]), 2.2), (d - np.array([0.6, 0, 0]), 0.8), (-d + np.array([0, 0.5, 0]), 1.0)):
        s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=k), pose=look(target + dd, target))
    r = pyrender.OffscreenRenderer(size, size)
    rgba, depth = r.render(s, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    # OSMesa returns RGB only: coverage from the depth buffer there
    a = rgba[..., 3:4] / 255.0 if rgba.shape[-1] == 4 else (depth > 0)[..., None].astype(np.float64)
    return Image.fromarray((rgba[..., :3] * a + BG * (1 - a)).astype(np.uint8))


def render_sheet(out, paths_):
    """The contact sheet (one row per VRM) -> out; returns per file the meshes, head bone and hair bbox. Tile titles: the
    view, and with several VRMs also the file name."""
    rows, info = [], {}
    for p in paths_:
        meshes, head = skinned_meshes(p)
        allv = np.concatenate([m['V'] for m in meshes])
        lo, hi = allv.min(0), allv.max(0)
        body_c = (lo + hi) / 2
        hc = head + np.array([0, 0.07, 0])
        # VRM 0.x faces -Z in glTF
        tiles = [('body front', render(meshes, body_c, (0, 0, -1), 0.55 * (hi - lo).max())),
                 ('head front', render(meshes, hc, (0, 0, -1), 0.26)), ('head side', render(meshes, hc, (1, 0, 0), 0.26)),
                 ('head back', render(meshes, hc, (0, 0, 1), 0.26))]
        row = Image.new('RGB', (360 * len(tiles), 384), 'white')
        d = ImageDraw.Draw(row)
        for k, (nm, t) in enumerate(tiles):
            row.paste(t, (360 * k, 24))
            d.text((360 * k + 6, 6), f'{os.path.basename(p)}: {nm}' if len(paths_) > 1 else nm, fill=(0, 0, 0))
        rows.append(row)
        hair = [m for m in meshes if 'hair' in m['name'].lower()]
        info[os.path.basename(p)] = dict(meshes=[(m['name'], len(m['V'])) for m in meshes], head_bone=head.round(4).tolist(),
                                         hair_bbox=[np.concatenate([m['V'] for m in hair]).min(0).round(3).tolist(),
                                                    np.concatenate([m['V'] for m in hair]).max(0).round(3).tolist()] if hair else None)
    sheet = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), 'white')
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out)
    return info


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    print('RENDER', json.dumps(render_sheet(sys.argv[1], sys.argv[2:])))
