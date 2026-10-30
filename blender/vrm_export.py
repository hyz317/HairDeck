"""Blender step of the VRM export, one Blender run per export (Blender 4.2 LTS with the VRM Add-on for Blender, MIT):
import the body's VRM 0.x, remove its hair, add the generated hair skinned to the head bone and the generated chain
bones, one VRM 0.x spring-bone group per chain, the body's meta fields, export (vrm_hair.py). hairdeck.vrm.export runs
it when no resident worker (vrm_worker.py) serves the body.

blender -b --python vrm_export.py -- BODY_JSON PREP_DIR OUT_DIR
Reads the prep npz in PREP_DIR (hairdeck.vrm.prep), writes the gen VRM and the export json to OUT_DIR (names:
hairdeck.vrm.body.file_name). The release root goes on sys.path for hairdeck.vrm.body (paths through HAIRDECK_ROOT or
configs/paths.json, as outside Blender).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import vrm_hair as E  # noqa: E402
import vrm_scene as V  # noqa: E402
from hairdeck.vrm import body as B  # noqa: E402


def main():
    t0 = time.time()
    argv = sys.argv[sys.argv.index('--') + 1:]
    bd, prep_dir, out = B.body(argv[0]), argv[1], argv[2]
    h, key, vrm = bd['h'], bd['key'], bd['vrm']
    os.makedirs(out, exist_ok=True)
    P = dict(np.load(f"{prep_dir}/{B.file_name('prep', bd)}.npz"))
    info = dict(h=h, key=key, vrm=vrm, skin_ASSIGN=float(P['ASSIGN']), skin_RAMP=float(P['RAMP']),
                skin_BLEND=bool(P.get('BLEND', False)), source='blender/vrm_export.py')
    arm = V.import_vrm(vrm)
    t1 = time.time()
    info['armature_matrix_world_is_identity'] = bool(np.allclose(np.array(arm.matrix_world), np.eye(4)))
    info['head_bone'] = V.human_bones(arm)['head']
    colour = [float(x) for x in P['colour']]
    info['colour_source'] = 'given (prep npz)'
    col_uuids, center = E.remove_original_hair(arm, info)
    E.add_generated_hair(arm, P, col_uuids, center, colour, info)
    t2 = time.time()
    info['vrm_meta_set'] = B.apply_vrm_meta(arm, bd)
    info['export_gen'] = E.export(arm, f"{out}/{B.file_name('gen', bd)}")
    info['blender_sec'] = dict(import_vrm=round(t1 - t0, 2), edit=round(t2 - t1, 2), export=round(time.time() - t2, 2))
    json.dump(info, open(f"{out}/{B.file_name('export', bd)}", 'w'), indent=1)
    print('INFO', json.dumps(info)[:2000])


if __name__ == '__main__':
    main()
