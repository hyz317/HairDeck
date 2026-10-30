"""Render the image-condition training data from full VRoid characters (Blender 4.2 LTS + VRM Add-on for Blender).

Each character is imported once, then N views are rendered, each with independently sampled augmentation:
  body pose   : group mix natural 70% / hand near head 25% / T or A 5%, IK targets jittered, one-handed poses mirrored
                at random, random spine / neck / head rotation, finger curl (V sign for v_near_face)
  hair motion : VRM 0.x spring bones (springbone.py). Settled 35%, otherwise mid-motion or just stopped after a head
                turn / nod / body spin / jump / sway. Hair gravity override U(0.1, 0.6) with probability 0.3.
  view        : azimuth relative to the posed head's facing: front and 3/4 80%, side 10%, back 10%;
                elevation N(5, 12) deg clipped to [-25, 40]
  render      : wide framing on the posed hair + face bbox (margin 0.15; the final crop is done by the data loader,
                with the same rule as at test time), light preset or random sun, MToon outline on 80% (forced for
                models exported without outlines) / off 20%, RGBA 768 px, EEVEE
The target stays the rest-pose hair cards of the dataset; nothing here changes it.

Outputs, per character: OUT/<key>/v<i>.png (RGBA), v<i>_lab.png (0 background, 1 hair mesh, 2 face mesh, 3 other),
meta.json (all sampled parameters, camera and posed head matrices, timings). Characters with meta.json are skipped,
failures are appended to OUT/_failed/<key>.txt.

blender -b --python gen_data.py -- OUT LIST_FILE [--views 8] [--seed 0] [--res 768] [--max_chars 0]
LIST_FILE lines: key<TAB>vrm path (key like 9/7067719180331438579). Parallel driver: scripts/data.py (step renders).
The released data: --views 8 --seed 0 --res 768 (data/renders for train, data/renders_test for test).
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
import zlib

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vrm_scene as S  # noqa: E402
from springbone import SpringSim  # noqa: E402

X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)


def smooth(a):
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)


def rest_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = Quaternion()
        pb.location = Vector()
        pb.scale = Vector((1, 1, 1))
    bpy.context.view_layer.update()


def compose(arm, bone, base, *axis_angles):
    """`base` basis rotation followed by extra rotations about rest-frame world axes (degrees)."""
    return S.axis_rot(arm, bone, *axis_angles) @ base


# ----------------------------------------------------------------------------- sampling
def head_rotation(pitch, yaw, roll):
    """The rotation S.axis_rot builds from (Z, yaw), (X, pitch), (Y, roll), as one quaternion."""
    return Quaternion(Vector(Y), math.radians(roll)) @ Quaternion(Vector(X), math.radians(pitch)) @ Quaternion(Vector(Z), math.radians(yaw))


def sample_body(rng, arm, hb, s, boxes):
    group = rng.choice(['natural', 'near_head', 'TA'], p=[0.70, 0.25, 0.05])
    name = str(rng.choice(S.POSE_GROUPS[group]))
    mirror = bool(rng.random() < 0.5)
    rot = dict(spine=(float(np.clip(rng.normal(0, 4), -10, 10)), float(np.clip(rng.normal(0, 5), -12, 12)), float(np.clip(rng.normal(0, 3), -8, 8))),
               head=(float(np.clip(rng.normal(0, 9), -25, 25)), float(np.clip(rng.normal(0, 15), -40, 40)), float(np.clip(rng.normal(0, 7), -20, 20))))
    pose = S.pose_preset(arm, hb, name, rng=rng, jitter=0.02 if name not in ('T',) else 0.0, mirror=mirror,
                         face_box=boxes['face'], head_box=boxes['head'], head_q=head_rotation(*rot['head']),
                         head_pivot=S.bone_head(arm, hb['neck']))
    pitch, yaw, roll = rot['spine']
    for b, k in (('spine', 0.6), ('chest', 0.4)):
        if b in hb:
            pose[hb[b]] = compose(arm, hb[b], pose.get(hb[b], Quaternion()), (Z, k * yaw), (X, k * pitch), (Y, k * roll))
    pitch, yaw, roll = rot['head']  # pitch > 0 looks down
    for b, k in (('neck', 0.45), ('head', 0.55)):
        pose[hb[b]] = compose(arm, hb[b], pose.get(hb[b], Quaternion()), (Z, k * yaw), (X, k * pitch), (Y, k * roll))
    # fingers
    curl = {}
    for side in ('left', 'right'):
        base = rng.uniform(8, 35)
        for fin in S.FINGERS:
            curl[(side, fin)] = float(np.clip(base + rng.normal(0, 6), 0, 60))
    if name == 'v_near_face':
        vs = 'left' if mirror else 'right'
        curl.update({(vs, 'Index'): 0.0, (vs, 'Middle'): 0.0, (vs, 'Ring'): 75.0, (vs, 'Little'): 75.0, (vs, 'Thumb'): 70.0})
    pose.update(S.finger_pose(arm, hb, curl))
    return pose, dict(group=str(group), name=name, mirror=mirror, spine_pyr=rot['spine'], head_pyr=rot['head'],
                      finger_base={sd: round(curl[(sd, 'Index')], 1) for sd in ('left', 'right')})


def sample_motion(rng):
    if rng.random() < 0.35:
        return dict(kind='settled')
    kind = str(rng.choice(['head_turn', 'head_nod', 'body_spin', 'jump', 'sway'], p=[0.3, 0.15, 0.25, 0.15, 0.15]))
    T = int(rng.integers(12, 37))
    m = dict(kind=kind, T=T, frame=int(rng.integers(int(0.4 * T), T + 41)), sign=float(rng.choice([-1, 1])))
    m['amp'] = float(dict(head_turn=rng.uniform(20, 50), head_nod=rng.uniform(10, 25), body_spin=rng.uniform(45, 150),
                          jump=rng.uniform(0.1, 0.3), sway=rng.uniform(5, 15))[kind])
    return m


def sample_view(rng):
    u = rng.random()
    if u < 0.8:
        az, bucket = float(np.clip(rng.normal(0, 22), -55, 55)), 'front'
    elif u < 0.9:
        az, bucket = float(rng.choice([-1, 1]) * rng.uniform(60, 110)), 'side'
    else:
        az, bucket = float(rng.choice([-1, 1]) * rng.uniform(120, 180)), 'back'
    return dict(az_rel=az, bucket=bucket, el=float(np.clip(rng.normal(5, 12), -25, 40)), dist=float(rng.uniform(1.6, 3.2)))


def sample_look(rng):
    u = rng.random()
    if u < 0.70:
        light = dict(kind='preset', name=str(rng.choice(['key_left', 'key_right', 'back_top', 'flat'], p=[0.4, 0.4, 0.05, 0.15])))
    else:
        light = dict(kind='random', rot=[float(rng.uniform(0.2, 1.1)), 0.0, float(rng.uniform(-1.5, 1.5))],
                     energy=float(rng.uniform(1.5, 4.5)), world=float(rng.uniform(0.3, 1.0)))
    return dict(light=light, outline=bool(rng.random() < 0.8), gravity=float(rng.uniform(0.1, 0.6)) if rng.random() < 0.3 else None)


def set_light(light):
    if light['kind'] == 'preset':
        S.set_light(light['name'])
    else:
        sun = bpy.data.objects['Sun']
        sun.rotation_euler, sun.data.energy = light['rot'], light['energy']
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (1, 1, 1, 1)
        bg.inputs[1].default_value = light['world']


# ----------------------------------------------------------------------------- one view
def run_sim(arm, hb, s, pose, motion, gravity):
    """Settle into `pose`, then play `motion` up to its capture frame. Returns the sim (already applied)."""
    rest_pose(arm)  # collider empties must be at rest when the sim reads them
    sim = SpringSim(arm, hair_gravity=gravity)
    for f in range(120):  # blend from rest into the pose, then let the springs settle
        a = smooth((f + 1) / 30)
        sim.basis.update({n: Quaternion().slerp(q, a) for n, q in pose.items()})
        sim.step()
    if motion['kind'] != 'settled':
        root = arm.data.bones[hb['hips']]
        while root.parent is not None:  # translate the top bone: hips may be connected, which ignores pose location
            root = root.parent
        rB = root.matrix_local.to_quaternion()
        k, T, A, sg = motion['kind'], motion['T'], motion['amp'], motion['sign']
        for f in range(1, motion['frame'] + 1):
            a = smooth(f / T)
            p = dict(pose)
            if k == 'head_turn':
                for b, w in (('neck', 0.4), ('head', 0.4), ('spine', 0.1), ('chest', 0.1)):
                    if b in hb:
                        p[hb[b]] = compose(arm, hb[b], pose.get(hb[b], Quaternion()), (Z, sg * w * A * a))
            elif k == 'head_nod':
                for b, w in (('neck', 0.5), ('head', 0.5)):
                    p[hb[b]] = compose(arm, hb[b], pose.get(hb[b], Quaternion()), (X, sg * w * A * a))
            elif k == 'body_spin':
                p[hb['hips']] = compose(arm, hb['hips'], pose.get(hb['hips'], Quaternion()), (Z, sg * A * a))
            elif k == 'sway':  # one side-to-side swing of the upper body, back to the start at frame T
                for b, w in (('spine', 0.5), ('chest', 0.5)):
                    if b in hb:
                        p[hb[b]] = compose(arm, hb[b], pose.get(hb[b], Quaternion()), (Y, sg * w * A * math.sin(math.pi * min(f / T, 1.0))))
            else:  # ballistic jump of the whole body, T frames in the air
                u = min(f / T, 1.0)
                sim.loc[root.name] = rB.inverted() @ Vector((0, 0, 4 * A * s * u * (1 - u)))
            sim.basis.update(p)
            sim.step()
    sim.apply()
    bpy.context.view_layer.update()
    return sim


def head_facing(arm, hb):
    """Posed head centre (world) and the yaw (deg, same convention as S.set_camera's az) of the head's facing."""
    pb = arm.pose.bones[hb['head']]
    R = (arm.matrix_world.to_3x3() @ pb.matrix.to_3x3()) @ (arm.matrix_world.to_3x3() @ pb.bone.matrix_local.to_3x3()).inverted()
    fwd = R @ Vector((0, -1, 0))
    up = R @ Vector((0, 0, 1))
    return math.degrees(math.atan2(fwd.x, -fwd.y)), fwd, up


def mat_list(M):
    return [list(map(float, r)) for r in M]


# ----------------------------------------------------------------------------- one character
def render_character(key, vrm, out, n_views, seed, res):
    t0 = time.time()
    arm = S.import_vrm(vrm)
    t_import = time.time() - t0
    hb = S.human_bones(arm)
    sc = bpy.context.scene
    S.setup_render(sc, res=res)
    fr = S.body_frame(arm, hb)
    s = fr['s']
    hair, face = S.hair_objects(), S.head_objects()
    if not hair:
        raise RuntimeError('no hair objects')
    n_mt, n_on = S.outline_state()
    forced = []
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    if n_on == 0:
        forced = S.force_outline(float(rng.uniform(0.0008, 0.002)))
    rest_pts = S.eval_points(hair)
    boxes = dict(face=None, head=None)  # rest-pose head geometry for hand placement (pose_preset)
    if face:
        fp = S.eval_points(face)
        flo, fhi = fp.min(0), fp.max(0)
        hp_ = np.concatenate([fp, rest_pts[rest_pts[:, 2] > flo[2]]])
        boxes = dict(face=(Vector(flo), Vector(fhi)), head=(Vector(hp_.min(0)), Vector(hp_.max(0))))
    meta = dict(key=key, vrm=vrm, seed=seed, res=res, scale=s, n_hair_objects=len(hair), face_objects=[o.name for o in face],
                mtoon_materials=n_mt, outline_materials=n_on, outline_forced=forced,
                hair_rest_bbox=[rest_pts.min(0).tolist(), rest_pts.max(0).tolist()],
                face_rest_bbox=[list(boxes['face'][0]), list(boxes['face'][1])] if face else None,
                head_rest_matrix=mat_list(arm.matrix_world @ arm.data.bones[hb['head']].matrix_local), views=[])
    d = os.path.join(out, key.replace('/', '_'))
    os.makedirs(d, exist_ok=True)
    for i in range(n_views):
        tv = time.time()
        pose, body = sample_body(rng, arm, hb, s, boxes)
        motion, view, look = sample_motion(rng), sample_view(rng), sample_look(rng)
        run_sim(arm, hb, s, pose, motion, look['gravity'])
        t_sim = time.time() - tv
        yaw, fwd, up = head_facing(arm, hb)
        hp = S.posed_head(arm, hb)[0]
        target = hp + up * 0.09 * s
        pts = S.eval_points(hair + face)
        if not face:
            pts = np.concatenate([pts, np.array(target) + 0.11 * s * np.concatenate([np.eye(3), -np.eye(3)])])
        az = yaw + view['az_rel']
        lens = S.frame_points(pts, target, view['dist'] * s, az, view['el'], margin=0.15)
        set_light(look['light'])
        S.set_outline(look['outline'])
        sc.render.image_settings.color_mode = 'RGBA'
        S.render(f'{d}/v{i}.png')
        with S.LabelPass(hair, face):
            S.render(f'{d}/v{i}_lab.png')
        cam = bpy.data.objects['Cam']
        meta['views'].append(dict(body=body, motion=motion, view=dict(view, az_world=az, head_yaw=yaw), look=look,
                                  lens=lens, shift=[cam.data.shift_x, cam.data.shift_y], sensor=cam.data.sensor_width,
                                  cam_matrix=mat_list(cam.matrix_world),
                                  head_matrix=mat_list(arm.matrix_world @ arm.pose.bones[hb['head']].matrix),
                                  t_sim=round(t_sim, 2), t_total=round(time.time() - tv, 2)))
    meta['t_import'] = round(t_import, 1)
    meta['t_total'] = round(time.time() - t0, 1)
    json.dump(meta, open(f'{d}/meta.json', 'w'))
    return meta


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('list')
    ap.add_argument('--views', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--res', type=int, default=768)
    ap.add_argument('--max_chars', type=int, default=0, help='exit after this many rendered characters (memory hygiene)')
    a = ap.parse_args(argv)
    for sub in ('_failed', '_inprogress'):
        os.makedirs(f'{a.out}/{sub}', exist_ok=True)
    items = [l.rstrip('\n').split('\t')[:2] for l in open(a.list) if l.strip()]
    done = 0
    for key, vrm in items:
        k_ = key.replace('/', '_')
        if os.path.exists(os.path.join(a.out, k_, 'meta.json')) or os.path.exists(f'{a.out}/_failed/{k_}.txt'):
            continue
        marker = f'{a.out}/_inprogress/{k_}'
        if os.path.exists(marker):  # a previous Blender process died on this character: do not retry forever
            open(f'{a.out}/_failed/{k_}.txt', 'w').write('Blender process died while rendering this character\n')
            os.remove(marker)
            print('FAIL', key, 'crashed earlier', flush=True)
            continue
        open(marker, 'w').close()
        try:
            m = render_character(key, vrm, a.out, a.views, a.seed, a.res)
            print('CHAR', key, 'import %.1fs total %.1fs' % (m['t_import'], m['t_total']),
                  'forced outline' if m['outline_forced'] else '', flush=True)
        except Exception:  # noqa: BLE001  (keep the worker alive; log and move on)
            open(f'{a.out}/_failed/{k_}.txt', 'w').write(traceback.format_exc())
            print('FAIL', key, traceback.format_exc().strip().splitlines()[-1], flush=True)
        os.remove(marker)
        done += 1
        if a.max_chars and done >= a.max_chars:
            break
    print('WORKER_EXIT', done, flush=True)


if __name__ == '__main__':
    main()
