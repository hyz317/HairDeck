"""Head-turn preview of generated hair with its spring-bone chains (pyrender with EGL or OSMesa, ffmpeg with libx264).

The cards and chains are in the head frame. Units are converted to meters the way the VRM export does it for the body
(hairdeck.vrm.body: head frame -> the body's dataset frame with its k, b, then / its fit scale s), so the spring
parameters act as they do in the exported VRM. Skinning is the VRM export's (hairdeck.vrm.prep): a card follows the
chain nearest to the lower 60% of its centreline (hairdeck.spring.sim.skin, within ASSIGN face widths), its mesh vertices
are bound with sim.bind (ramp RAMP from the chain root, blend between neighbouring segments), every other card is rigid
with the head. Simulation: the VRM 0.x update of hairdeck.spring.sim (UniVRM: verlet with drag, stiffness pull toward
the parent-rotated rest direction, length constraint; drag 0.4, gravity 0, the predicted stiffness, as
blender/vrm_hair.py writes them), plus a sphere collider for the head (the head proxy's centre, radius 0.45 face widths,
hit radius 0.02 m). As in the VRM export, the leading joints of a chain that lie inside the collider at rest stay rigid
with the head; later joints that rest inside are not pushed out (no start-up jump). Motion about the head joint: yaw to
+45 deg, hold, to -45 deg, hold, back to 0 (smoothstep segments), 60 Hz simulation, every second frame rendered: 3 s at
30 fps. Drawn: the hair (flat shaded, the given colour), the head proxy (grey) and optionally the chain joints.

PYOPENGL_PLATFORM=egl|osmesa python -m hairdeck.vis.turn_preview CARDS_JSON CHAINS_JSON R,G,B OUT.mp4 [--chains_visible]
       [--body BODY_JSON]
(CHAINS_JSON '-' for none; colour sRGB 0..1; ffmpeg: HAIRDECK_FFMPEG or the one on PATH.) Prints 'PREVIEW' + a json
summary.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from ..cards.mesh import card_mesh_vf
from ..spring.sim import bind, centreline, clean, deform, rot_y, rotdiff, skin
from ..vrm.body import body as load_body
from ..vrm.render import look
from .head_proxy import canonical

FPS, DUR, EVERY = 60, 3.0, 2
DRAG, GRAVITY, HIT = 0.4, 0.0, 0.02  # the spring-group parameters of the VRM export (blender/vrm_hair.py)
W_HEAD = 0.12759823777166274  # face width in head-frame units
BG = np.array([238, 238, 238], np.float64)


def _pyrender():
    """pyrender, imported on first use (PYOPENGL_PLATFORM defaults to egl)."""
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import pyrender
    return pyrender


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def yaw_deg(t):
    return (45 * smooth((t - 0.15) / 0.45) - 90 * smooth((t - 0.95) / 0.65) + 45 * smooth((t - 2.15) / 0.5))


def head_proxy(W, O):
    """hairdeck.vis.head_proxy's ellipsoid at icosphere subdivision 3, and its centre."""
    face_c = O - np.array([0, 0.547 * W, 0])
    c = face_c + np.array([0, 0.10 * W, -0.2 * W])
    m = trimesh.creation.icosphere(subdivisions=3)
    m.apply_scale([0.405 * W, 0.5 * W, 0.5 * W])
    m.apply_translation(c)
    return m, c


def simulate(chains, R, piv, sphere):
    """hairdeck.spring.sim.simulate (rotation about piv, no translation) with a sphere collider (centre, radius) and a
    rigid leading run: the bones before the root (the bone starting at the last joint of the leading run of joints inside
    the collider at rest) follow the head. Returns per chain (heads, rots) and per chain dict(joints, rigid_bones,
    pushes)."""
    ctr, rad = sphere
    NF = len(R)
    out, info = [], []
    for c in chains:
        J = c['J']
        d = np.diff(J, axis=0)
        L = np.linalg.norm(d, axis=1)
        D = d / L[:, None]
        inside0 = np.linalg.norm(J - ctr, axis=1) < rad + HIT
        f = 0
        while f < len(J) and inside0[f]:
            f += 1
        root = max(f - 1, 0)  # bones before the root follow the head rigidly (the VRM export's rule)
        cur, prev = J[1:].copy(), J[1:].copy()
        heads, rots = np.zeros((NF, len(L), 3)), np.zeros((NF, len(L), 3, 3))
        pushes = 0
        for fr in range(NF):
            Wp = R[fr]
            head = R[fr] @ (J[0] - piv) + piv
            Cw = R[fr] @ (ctr - piv) + piv
            for k in range(len(L)):
                rest = Wp @ D[k]
                if k < root or f >= len(J):
                    nxt = head + rest * L[k]
                    v = rest
                else:
                    nxt = cur[k] + (cur[k] - prev[k]) * (1 - c['drag']) + rest * (c['stiff'] * (1.0 / FPS)) + c['grav'] * (1.0 / FPS)
                    v = nxt - head
                    n = np.linalg.norm(v)
                    v = v / n if n > 1e-12 else rest
                    nxt = head + v * L[k]
                    if not inside0[k + 1]:  # collider push-out, then the length constraint again (UniVRM)
                        e = nxt - Cw
                        r = np.linalg.norm(e)
                        if r < rad + HIT:
                            nxt = Cw + e / max(r, 1e-12) * (rad + HIT)
                            v = nxt - head
                            v = v / max(np.linalg.norm(v), 1e-12)
                            nxt = head + v * L[k]
                            pushes += 1
                prev[k], cur[k] = cur[k], nxt
                Wk = rotdiff(rest, v) @ Wp
                heads[fr, k], rots[fr, k] = head, Wk
                Wp, head = Wk, nxt
        out.append((heads, rots))
        info.append(dict(joints=len(J), rigid_bones=int(min(root, len(L)) if f < len(J) else len(L)), pushes=pushes))
    return out, info


def flat_mesh(V, F, colour, double=True):
    """pyrender mesh with one normal per face (flat shading)."""
    pyrender = _pyrender()
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
    prim = pyrender.Primitive(positions=tri.reshape(-1, 3).astype(np.float32), normals=np.repeat(n, 3, 0).astype(np.float32),
                              material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[*colour, 1.0], metallicFactor=0.0,
                                                                          roughnessFactor=0.8, doubleSided=double))
    return pyrender.Mesh([prim])


def preview(cards_json, chains_json, colour, out, show_chains=False, body=None):
    """Simulate and render one hairstyle -> out (mp4); returns the summary dict. colour: sRGB (3,); chains_json: path or
    '-'; body: a body file (default the default body of hairdeck.vrm.body)."""
    pyrender = _pyrender()
    t_start = time.time()
    colour = np.asarray(colour, np.float64)
    bd = load_body(body)
    fit, k, b = bd['fit'], bd['k'], np.asarray(bd['b'], np.float64)
    m = 1.0 / (k * fit['s'])  # head-frame units -> meters
    W = W_HEAD * m
    piv = (np.array(fit['head_joint']) * k + b) * m
    proxy, pc = head_proxy(*canonical())
    proxy_V, proxy_F = np.asarray(proxy.vertices) * m, np.asarray(proxy.faces)
    sphere = (pc * m, 0.45 * W)
    cards = [(np.asarray(c['seq'], np.float64), 1 if c['shape'] == 'triangle' else 0) for c in json.load(open(cards_json))]
    cards = [(s, sh) for s, sh in cards if len(s) >= 3]
    chains_in = json.load(open(chains_json)) if chains_json != '-' else []
    chains = []
    for c in chains_in:
        J = clean(np.asarray(c['joints'], np.float64) * m)
        if len(J) >= 2:
            chains.append(dict(J=J, stiff=float(c['stiffness']), drag=DRAG, grav=np.array([0, -GRAVITY, 0])))
    # mesh + binding (hairdeck.vrm.prep's per-card procedure)
    Vs, Fs, off = [], [], 0
    ch, sg, wt, sg2, al = [], [], [], [], []
    rigid_mismatch = 0
    for seq, sh in cards:
        v, f = card_mesh_vf(seq, sh)
        v = v * m
        n = len(v)
        cc = int(skin([centreline(seq[:, :3]) * m], chains, W)[0][0]) if chains else -1
        kk, ww, kk2, aa = np.zeros(n, int), np.zeros(n), np.zeros(n, int), np.ones(n)
        if cc >= 0:
            bb = bind([v], [chains[cc]], W)
            if (bb[0] < 0).any():
                rigid_mismatch += 1
                cc = -1
            else:
                kk, ww = bb[1], bb[2]
                kk2, aa = (bb[3], bb[4]) if len(bb) == 5 else (kk.copy(), np.ones(n))
        ch.append(np.full(n, cc))
        sg.append(kk)
        wt.append(ww)
        sg2.append(kk2)
        al.append(aa)
        Vs.append(v)
        Fs.append(f + off)
        off += n
    V, F = np.concatenate(Vs), np.concatenate(Fs)
    binding = tuple(np.concatenate(x) for x in (ch, sg, wt, sg2, al))
    t = np.arange(1, int(DUR * FPS) + 1) / FPS
    R = np.stack([rot_y(np.radians(yaw_deg(tt))) for tt in t])
    t0 = time.time()
    sim, sinfo = simulate(chains, R, piv, sphere) if chains else ([], [])
    X = deform(V, binding, chains, sim, R, np.zeros((len(R), 3)), piv) if chains else \
        np.einsum('fij,mj->fmi', R, V - piv) + piv
    t_sim = time.time() - t0
    frames = list(range(EVERY - 1, len(R), EVERY))
    # local motion of the hair relative to the head (what the springs add), for the summary
    loc = np.einsum('fji,fmj->fmi', R, X - piv) + piv - V
    disp = np.linalg.norm(loc, axis=-1)
    # camera: fixed, slightly from the side and above, framing every frame
    P = np.concatenate([X[frames].reshape(-1, 3), (np.einsum('fij,mj->fmi', R[frames], proxy_V - piv) + piv).reshape(-1, 3)])
    lo, hi = P.min(0), P.max(0)
    tgt = (lo + hi) / 2
    half = 0.56 * (hi - lo)[:2].max()
    d = np.array([np.sin(np.radians(20)), 0.15, np.cos(np.radians(20))])
    d /= np.linalg.norm(d)
    size = 480
    r = pyrender.OffscreenRenderer(size, size)
    fdir = f'{os.path.splitext(out)[0]}_frames'
    os.makedirs(fdir, exist_ok=True)
    t0 = time.time()
    for i, fr in enumerate(frames):
        s = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.28] * 3)
        s.add(pyrender.OrthographicCamera(xmag=half, ymag=half, znear=0.01, zfar=40 * half), pose=look(tgt + d * 8 * half, tgt))
        for dd, kk in ((d + np.array([-0.5, 0.8, 0]), 2.4), (d + np.array([0.8, 0, 0]), 0.8), (-d + np.array([0, 0.5, 0]), 1.2)):
            s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=kk), pose=look(tgt + dd, tgt))
        s.add(flat_mesh(X[fr], F, colour))
        s.add(flat_mesh(np.einsum('ij,mj->mi', R[fr], proxy_V - piv) + piv, proxy_F, np.array([0.72, 0.72, 0.72]), False))
        if show_chains:
            for (heads, _), c in zip(sim, chains):
                for q in heads[fr]:
                    sp = trimesh.creation.icosphere(subdivisions=1, radius=0.006)
                    sp.apply_translation(q)
                    s.add(flat_mesh(np.asarray(sp.vertices), np.asarray(sp.faces), np.array([0.95, 0.45, 0.1]), False))
        rgba, depth = r.render(s, flags=pyrender.RenderFlags.RGBA)
        # OSMesa returns RGB only: coverage from the depth buffer there
        a = rgba[..., 3:4] / 255.0 if rgba.shape[-1] == 4 else (depth > 0)[..., None].astype(np.float64)
        im = Image.fromarray((rgba[..., :3] * a + BG * (1 - a)).astype(np.uint8))
        ImageDraw.Draw(im).text((8, 8), f't = {t[fr]:.2f} s   head yaw {yaw_deg(t[fr]):+.0f} deg   max hair motion vs head '
                                         f'{disp[fr].max() * 100:.1f} cm', fill=(40, 40, 40))
        im.save(f'{fdir}/{i:04d}.png')
    r.delete()
    t_render = time.time() - t0
    t0 = time.time()
    ffmpeg = os.environ.get('HAIRDECK_FFMPEG') or shutil.which('ffmpeg') or 'ffmpeg'
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(FPS // EVERY), '-i', f'{fdir}/%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-movflags', '+faststart', out], check=True)
    t_enc = time.time() - t0
    shutil.rmtree(fdir, ignore_errors=True)
    return dict(out=out, frames=len(frames), fps=FPS // EVERY, seconds=round(len(frames) / (FPS // EVERY), 2), n_cards=len(cards),
                n_chains=len(chains), cards_on_chains=int(sum(1 for c in ch if c[0] >= 0)),
                bind_mismatch_rigid=rigid_mismatch, chain_info=sinfo, collider_pushes=int(sum(x['pushes'] for x in sinfo)),
                max_motion_cm=round(float(disp.max() * 100), 2), mean_motion_cm=round(float(disp.mean() * 100), 3),
                meters_per_head_unit=m, face_width_m=W, sec=dict(sim=round(t_sim, 2), render=round(t_render, 2),
                                                                 encode=round(t_enc, 2), total=round(time.time() - t_start, 2)))


def main():
    ap = argparse.ArgumentParser(description='head-turn preview of generated hair with its spring-bone chains')
    ap.add_argument('cards_json')
    ap.add_argument('chains_json', help="chains json, or '-' for none")
    ap.add_argument('colour', help='R,G,B hair colour, sRGB 0..1')
    ap.add_argument('out', help='output mp4')
    ap.add_argument('--chains_visible', action='store_true', help='draw the chain joints')
    ap.add_argument('--body', default=None, help='body file (default: hairdeck.vrm.body.default_path())')
    a = ap.parse_args()
    res = preview(a.cards_json, a.chains_json, [float(x) for x in a.colour.split(',')], a.out, a.chains_visible, a.body)
    print('PREVIEW', json.dumps(res))


if __name__ == '__main__':
    main()
