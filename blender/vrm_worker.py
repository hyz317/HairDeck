"""Resident Blender worker of the VRM export (Blender 4.2 LTS with the VRM Add-on for Blender, MIT), so an export does
not pay Blender's start-up and the ~25 s VRM import of the body. Same steps as vrm_export.py, split in two:
  once     import the body's VRM, export it unchanged (the orig file in CACHE_DIR: the add-on's re-export, the
           reference of hairdeck.vrm.check), vrm_hair.remove_original_hair, save the result as a .blend plus a JSON with
           what that step returns and records (collider groups and spring center for the new groups, removed groups);
  per job  open that .blend (Blender's own file loader, ~1 s), vrm_hair.add_generated_hair with the job's prep npz, the
           body's meta fields, vrm_hair.export.
Jobs are files: JOB_DIR/<id>.json = {"prep": PREP_DIR, "out": OUT_DIR}; the worker renames it to <id>.running and writes
<id>.done.json ({"ok", "info" | "error"}); OUT_DIR gets the gen VRM and the export json, as from vrm_export.py
(names: hairdeck.vrm.body.file_name). JOB_DIR/ready holds {"pid", "body"} while the worker runs; it stops when JOB_DIR/stop exists or its parent
process is gone.

blender -b --python vrm_worker.py -- BODY_JSON JOB_DIR CACHE_DIR PARENT_PID
"""
import glob
import json
import os
import sys
import time
import traceback

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import vrm_hair as E  # noqa: E402
import vrm_scene as V  # noqa: E402
from hairdeck.vrm import body as B  # noqa: E402


def log(*a):
    print(time.strftime('%F %T'), 'VRM_WORKER', *a, flush=True)


def build_cache(bd, cache):
    h, key, vrm = bd['h'], bd['key'], bd['vrm']
    t0 = time.time()
    arm = V.import_vrm(vrm)
    info = dict(h=h, key=key, vrm=vrm, armature_matrix_world_is_identity=bool(np.allclose(np.array(arm.matrix_world), np.eye(4))),
                head_bone=V.human_bones(arm)['head'])
    ref, nohair = f"{cache}/{B.file_name('orig', bd)}", f"{cache}/{B.file_name('nohair', bd)}"
    if not os.path.exists(ref):  # the add-on's unchanged re-export of the body, for hairdeck.vrm.check
        os.makedirs(cache, exist_ok=True)
        info['export_orig'] = E.export(arm, ref)
    col_uuids, center = E.remove_original_hair(arm, info)
    E.reset_pose(arm)
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    os.makedirs(cache, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=f'{nohair}.blend', compress=False)
    json.dump(dict(info=info, col_uuids=list(col_uuids), center=center, built=time.strftime('%F %T'),
                   build_sec=round(time.time() - t0, 1)), open(f'{nohair}.json', 'w'), indent=1)
    log('cache built', f'{nohair}.blend', round(time.time() - t0, 1), 's')


def job(bd, cache, prep_dir, out):
    nohair = f"{cache}/{B.file_name('nohair', bd)}"
    t0 = time.time()
    c = json.load(open(f'{nohair}.json'))
    bpy.ops.wm.open_mainfile(filepath=f'{nohair}.blend')
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    t1 = time.time()
    P = dict(np.load(f"{prep_dir}/{B.file_name('prep', bd)}.npz"))
    info = dict(c['info'], skin_ASSIGN=float(P['ASSIGN']), skin_RAMP=float(P['RAMP']), skin_BLEND=bool(P.get('BLEND', False)),
                colour_source='given (prep npz)', source='blender/vrm_worker.py (hair removal cached)')
    colour = [float(x) for x in P['colour']]
    E.add_generated_hair(arm, P, c['col_uuids'], c['center'], colour, info)
    t2 = time.time()
    os.makedirs(out, exist_ok=True)
    info['vrm_meta_set'] = B.apply_vrm_meta(arm, bd)
    info['export_gen'] = E.export(arm, f"{out}/{B.file_name('gen', bd)}")
    info['blender_sec'] = dict(open_blend=round(t1 - t0, 2), edit=round(t2 - t1, 2), export=round(time.time() - t2, 2))
    json.dump(info, open(f"{out}/{B.file_name('export', bd)}", 'w'), indent=1)
    return info


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    body_json, jobs, cache, parent = B.body_path(argv[0]), argv[1], argv[2], int(argv[3])
    bd = B.body(body_json)
    nohair = f"{cache}/{B.file_name('nohair', bd)}"
    os.makedirs(jobs, exist_ok=True)
    if not os.path.exists(f'{nohair}.blend') or not os.path.exists(f'{nohair}.json'):
        build_cache(bd, cache)
    json.dump(dict(pid=os.getpid(), body=body_json), open(f'{jobs}/ready.tmp', 'w'))
    os.replace(f'{jobs}/ready.tmp', f'{jobs}/ready')
    log('ready', os.getpid(), body_json)
    while True:
        if os.path.exists(f'{jobs}/stop'):
            break
        try:
            os.kill(parent, 0)
        except OSError:
            log('parent', parent, 'gone')
            break
        todo = sorted(glob.glob(f'{jobs}/*.json'))
        todo = [f for f in todo if not f.endswith('.done.json')]
        if not todo:
            time.sleep(0.2)
            continue
        f = todo[0]
        jid = os.path.basename(f)[:-5]
        run = f'{jobs}/{jid}.running'
        try:
            os.rename(f, run)
        except OSError:
            continue
        try:
            spec = json.load(open(run))
            info = job(bd, cache, spec['prep'], spec['out'])
            res = dict(ok=True, info=info)
        except Exception as e:  # noqa: BLE001
            res = dict(ok=False, error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-3000:])
        json.dump(res, open(f'{jobs}/{jid}.done.json.tmp', 'w'), default=str)
        os.replace(f'{jobs}/{jid}.done.json.tmp', f'{jobs}/{jid}.done.json')
        os.remove(run)
        log('job', jid, 'ok' if res['ok'] else res['error'])
    if os.path.exists(f'{jobs}/ready'):
        os.remove(f'{jobs}/ready')
    log('stopped')


if __name__ == '__main__':
    main()
