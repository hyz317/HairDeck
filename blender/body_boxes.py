"""Rest-pose boxes of an avatar body for hairdeck.vrm.body build (Blender 4.2 LTS with the VRM Add-on for Blender, MIT):
import a VRM (vrm_scene.import_vrm) and write the rest-pose fields that gen_data.py stores in meta.json and
hairdeck.data.head_frame.fit_one reads: hair_rest_bbox, face_rest_bbox (hair and face meshes of vrm_scene, evaluated
vertices) and head_rest_matrix (rest matrix of the humanoid head bone in the Blender world).

blender -b --python body_boxes.py -- VRM OUT_JSON
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vrm_scene as S  # noqa: E402


def main():
    vrm, out = sys.argv[sys.argv.index('--') + 1:][:2]
    arm = S.import_vrm(vrm)
    hb = S.human_bones(arm)
    hair, face = S.hair_objects(), S.head_objects()
    if not hair or not face:
        raise RuntimeError(f'hair objects {len(hair)}, face objects {len(face)}')
    rest_pts, fp = S.eval_points(hair), S.eval_points(face)
    M = arm.matrix_world @ arm.data.bones[hb['head']].matrix_local
    meta = dict(vrm=vrm, n_hair_objects=len(hair), hair_objects=[o.name for o in hair], face_objects=[o.name for o in face],
                hair_rest_bbox=[rest_pts.min(0).tolist(), rest_pts.max(0).tolist()],
                face_rest_bbox=[fp.min(0).tolist(), fp.max(0).tolist()],
                head_rest_matrix=[list(map(float, r)) for r in M], head_bone=hb['head'])
    json.dump(meta, open(out, 'w'), indent=1)
    print('BODY_META', json.dumps(meta))


main()
