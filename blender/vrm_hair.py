"""Generated hair and spring-bone chains on a VRM 0.x body inside Blender (Blender 4.2 LTS with the VRM Add-on for
Blender, MIT). Inputs come from hairdeck.vrm.prep, already in the Blender world of the imported body. Used by
vrm_export.py (one Blender run per export) and vrm_worker.py (resident worker).

  remove_original_hair: hair meshes = mesh objects whose object or mesh name contains 'hair' (VRoid: Hair001...; the
     HairBack material inside Body is part of the body mesh and stays); hair spring-bone groups = groups whose bone
     subtree skins hair-mesh vertices (VRoid group comments / bone names are not reliable); hair bones = non-humanoid
     bones that skin hair vertices, the bones of the hair groups and everything below them (plus non-humanoid bones with
     'hair' in the name), deleted only if no remaining mesh, spring group, collider group or object uses anything in
     their subtree; every other spring group and all collider groups are kept as they are; blend-shape binds and
     first-person annotations of the removed meshes are removed;
  add_generated_hair: the generated hair mesh (single-colour MToon, double sided, per-card UVs) skinned to the head bone
     (humanoid 'head') and to the new chain bones: chain c gets bones HairGen_cCC_KK from joint k to joint k+1, the
     first parented to the head bone, plus a leaf HairGen_cCC_end at the last joint (VRoid style '_end' node, so the last
     joint is an explicit node and the last segment keeps its length in UniVRM / three-vrm); weights from
     hairdeck.spring.sim.bind (prep): head 1 - w, segment k w * alpha, neighbour segment w * (1 - alpha); then one VRM
     0.x spring-bone group per chain: stiffness = predicted, drag 0.4, gravity 0, hit radius 0.02, center as the
     original hair groups, collider groups = those the original hair groups used (all collider groups if none); root =
     the bone starting at the last joint of the leading run (from the chain root) of joints inside those collider
     spheres at rest (radius + hit radius), the bones before it stay rigid with the head (no start-up push of the scalp
     part); no group if every joint is inside;
  export: rest pose, VRM export of the armature.
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vrm_scene as V  # noqa: E402

DRAG, GRAVITY, HIT = 0.4, 0.0, 0.02


def used_groups(o):
    """Names of the vertex groups of mesh object o that carry a positive weight on some vertex."""
    names = {g.index: g.name for g in o.vertex_groups}
    used = set()
    for v in o.data.vertices:
        for g in v.groups:
            if g.weight > 0 and g.group in names:
                used.add(names[g.group])
    return used


def subtree(bones, name):
    return [name] + [c.name for c in bones[name].children_recursive] if name in bones else []


def reset_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def export(arm, path):
    """Rest pose, object mode, VRM export of the armature to path; returns whether the file was written."""
    reset_pose(arm)
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    r = bpy.ops.export_scene.vrm(filepath=path, armature_object_name=arm.name, ignore_warning=True)
    ok = 'FINISHED' in r and os.path.exists(path)
    print('EXPORT', path, r, '%.1f MB' % (os.path.getsize(path) / 1e6) if ok else 'MISSING')
    return ok


def remove_original_hair(arm, info):
    """Remove the body's hair meshes, hair spring groups and hair bones (module docstring); records what was removed in
    info. Returns (collider group uuids for the new groups, spring center bone name)."""
    bones = arm.data.bones
    sa = arm.data.vrm_addon_extension.vrm0.secondary_animation
    hum = set(V.human_bones(arm).values())
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    hair = [o for o in meshes if 'hair' in o.name.lower() or 'hair' in o.data.name.lower()]
    other = [o for o in meshes if o not in hair]
    hair_used = set().union(*[used_groups(o) for o in hair]) & set(bones.keys()) if hair else set()
    other_used = set().union(*[used_groups(o) for o in other]) & set(bones.keys()) if other else set()
    hair_bones = hair_used - hum
    info['hair_objects'] = [(o.name, o.data.name, len(o.data.vertices)) for o in hair]
    info['hair_skin_bones'] = len(hair_bones)
    # spring-bone groups
    hair_g, mixed_g, col_uuids, centers, params = [], [], [], [], []
    for i, g in enumerate(sa.bone_groups):
        sub = set().union(*[set(subtree(bones, b.bone_name)) for b in g.bones]) if len(g.bones) else set()
        if not sub & hair_bones:
            continue
        if sub & other_used:
            mixed_g.append(i)
            continue
        hair_g.append(i)
        for r in g.collider_groups:
            if r.collider_group_uuid and r.collider_group_uuid not in col_uuids:
                col_uuids.append(r.collider_group_uuid)
        centers.append(g.center.bone_name)
        params.append(dict(roots=[b.bone_name for b in g.bones], stiffness=round(g.stiffiness, 4), gravity=round(g.gravity_power, 4),
                           drag=round(g.drag_force, 4), hit=round(g.hit_radius, 4), comment=g.comment))
    info['removed_groups'] = params
    info['mixed_groups_kept'] = mixed_g
    info['kept_groups'] = [(g.comment, [b.bone_name for b in g.bones][:3]) for i, g in enumerate(sa.bone_groups) if i not in hair_g]
    cand = set(hair_bones)
    for i in hair_g:
        for b in sa.bone_groups[i].bones:
            cand |= set(subtree(bones, b.bone_name))
    cand |= {b.name for b in bones if 'hair' in b.name.lower() and b.name not in hum}
    for n in list(cand):
        cand |= set(subtree(bones, n))
    for i in sorted(hair_g, reverse=True):
        sa.bone_groups.remove(i)
    protect = set(hum) | other_used
    for g in sa.bone_groups:
        for b in g.bones:
            protect |= set(subtree(bones, b.bone_name))
        protect.add(g.center.bone_name)
    for cg in sa.collider_groups:
        protect.add(cg.node.bone_name)
    for o in bpy.data.objects:
        if o.parent == arm and o.parent_bone:
            protect.add(o.parent_bone)
    dele = {n for n in cand if not set(subtree(bones, n)) & protect}
    info['bones_candidates'] = len(cand)
    info['bones_deleted'] = len(dele)
    info['bones_protected_candidates'] = sorted(cand - dele)[:20]
    # meshes, then bones
    for o in hair:
        me = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if me.users == 0:
            bpy.data.meshes.remove(me)
    for o in other:
        for vg in list(o.vertex_groups):
            if vg.name in dele:
                o.vertex_groups.remove(vg)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    for n in dele:
        if n in eb:
            eb.remove(eb[n])
    bpy.ops.object.mode_set(mode='OBJECT')
    head = V.human_bones(arm)['head']
    info['nonhumanoid_bones_left_under_head'] = [c.name for c in arm.data.bones[head].children_recursive if c.name not in hum]
    # dangling references from blend shapes / first person to the removed meshes
    ext = arm.data.vrm_addon_extension.vrm0
    gone = {n for n, _, _ in info['hair_objects']}
    nb = 0
    for bg in ext.blend_shape_master.blend_shape_groups:
        for k in reversed(range(len(bg.binds))):
            if bg.binds[k].mesh.mesh_object_name in gone:
                bg.binds.remove(k)
                nb += 1
    for k in reversed(range(len(ext.first_person.mesh_annotations))):
        if ext.first_person.mesh_annotations[k].mesh.mesh_object_name in gone:
            ext.first_person.mesh_annotations.remove(k)
            nb += 1
    info['dangling_refs_removed'] = nb
    if not col_uuids:  # no original hair colliders: use all collider groups
        col_uuids = [cg.uuid for cg in sa.collider_groups]
    center = max(set(centers), key=centers.count) if centers else ''
    return col_uuids, center


def add_generated_hair(arm, P, col_uuids, center, colour, info):
    """Chain bones, the skinned hair mesh, its MToon material and the spring groups (module docstring). P: the prep
    arrays (hairdeck.vrm.prep), colour: linear RGB; records counts and the spring-root choice in info."""
    sa = arm.data.vrm_addon_extension.vrm0.secondary_animation
    head = V.human_bones(arm)['head']
    inv = np.array(arm.matrix_world.inverted())
    to_arm = lambda X: X @ inv[:3, :3].T + inv[:3, 3]  # noqa: E731
    ptr = P['chain_ptr']
    chains = [to_arm(P['chain_J'][ptr[c]:ptr[c + 1]]) for c in range(len(ptr) - 1)]
    names = []
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    for c, J in enumerate(chains):
        prev, nm = eb[head], []
        for k in range(len(J) - 1):
            b = eb.new(f'HairGen_c{c:02d}_{k:02d}')
            b.head, b.tail = Vector(J[k]), Vector(J[k + 1])
            b.parent, b.use_connect, b.use_deform = prev, k > 0, True
            prev = b
            nm.append(b.name)
        d = Vector(J[-1] - J[-2]).normalized()
        e = eb.new(f'HairGen_c{c:02d}_end')
        e.head, e.tail = Vector(J[-1]), Vector(J[-1]) + d * 0.02
        e.parent, e.use_connect, e.use_deform = prev, True, True
        names.append(nm)
    bpy.ops.object.mode_set(mode='OBJECT')
    missing = [n for nm in names for n in nm if n not in arm.data.bones]
    assert not missing, f'bones lost on leaving edit mode: {missing[:5]}'
    # mesh
    Vb = to_arm(P['V'])
    me = bpy.data.meshes.new('Hair_gen')
    me.from_pydata(Vb.tolist(), [], P['F'].tolist())
    me.update()
    if 'UV' in P:  # per-card UVs (hairdeck.cards.mesh.card_uv: u across the width, v from root to tip)
        uvl = me.uv_layers.new(name='UVMap')
        vi = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', vi)
        uvl.data.foreach_set('uv', np.asarray(P['UV'], np.float32)[vi].reshape(-1))
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    obj = bpy.data.objects.new('Hair_gen', me)
    (arm.users_collection[0] if arm.users_collection else bpy.context.scene.collection).objects.link(obj)
    obj.parent = arm
    mod = obj.modifiers.new('Armature', 'ARMATURE')
    mod.object = arm
    vg_head = obj.vertex_groups.new(name=head)
    vgs = {}

    def add(bn, i, wt):
        if bn not in vgs:
            vgs[bn] = obj.vertex_groups.new(name=bn)
        vgs[bn].add([i], wt, 'REPLACE')
    ch, sg, w = P['vchain'], P['vseg'], P['vw']
    sg2, al = P.get('vseg2', sg), P.get('valpha', np.ones(len(sg)))
    n3 = 0
    for i in range(len(Vb)):  # head 1 - w, segment k w * alpha, neighbour segment w * (1 - alpha) (sim.bind)
        if ch[i] < 0 or w[i] <= 0:
            vg_head.add([i], 1.0, 'REPLACE')
            continue
        two = sg2[i] != sg[i] and al[i] < 1
        add(names[ch[i]][sg[i]], i, float(w[i] * (al[i] if two else 1.0)))
        if two:
            add(names[ch[i]][sg2[i]], i, float(w[i] * (1 - al[i])))
        if w[i] < 1:
            vg_head.add([i], float(1 - w[i]), 'REPLACE')
        n3 += two and w[i] < 1
    # material: single-colour MToon, double sided
    mat = bpy.data.materials.new('Hair_gen_MToon')
    mat.use_nodes = True
    obj.data.materials.append(mat)
    bpy.ops.vrm.convert_material_to_mtoon1(material_name=mat.name)
    mt = mat.vrm_addon_extension.mtoon1
    mt.pbr_metallic_roughness.base_color_factor = (*colour, 1.0)
    mt.extensions.vrmc_materials_mtoon.shade_color_factor = tuple(0.6 * c for c in colour)
    mt.double_sided = True
    # spring-bone groups. Walking from the chain root, the leading run of joints that lie inside a collider sphere of the
    # chain's collider groups at rest (distance < radius + hit radius; typically the scalp part inside the head
    # collider) stays rigid with the head: the bones between those joints keep their skin weights but are not
    # simulated, and the group's root is the bone that starts at the last joint of that run, so its tail (the first
    # joint outside all colliders) and everything below are simulated. Joints further down that rest inside a body
    # collider (tips on the chest / arms) stay simulated, as in the VRoid rigs. A chain whose joints are all inside gets
    # no group; a chain whose root joint is outside is unchanged.
    reset_pose(arm)
    spheres, sphere_bone = [], []
    for cg in sa.collider_groups:
        if cg.uuid not in col_uuids:
            continue
        for col in cg.colliders:
            o = col.bpy_object
            if o is None:
                continue
            sc = float(np.mean([abs(x) for x in o.matrix_world.to_scale()]))
            spheres.append((to_arm(np.array(o.matrix_world.to_translation())[None])[0], o.empty_display_size * sc + HIT))
            sphere_bone.append(cg.node.bone_name)
    roots, rigid_chains, rigid_bones, rigid_alt, cause, later_inside = [], [], 0, 0, {}, 0

    def inside(p):
        return [si for si, (ctr, r) in enumerate(spheres) if np.linalg.norm(p - ctr) < r]
    for c, J in enumerate(chains):
        f = 0  # first joint outside all colliders
        while f < len(J) and inside(J[f]):
            f += 1
        if f > 0:
            for si in inside(J[f - 1]):
                cause[sphere_bone[si]] = cause.get(sphere_bone[si], 0) + 1
        later_inside += sum(1 for j in range(f + 1, len(J)) if inside(J[j]))
        L = max(f - 1, 0)  # root bone index
        rigid_alt += min(f, len(J) - 1)  # if the root were the bone starting at the first outside joint instead
        if len(J) - 1 - L < 1:
            rigid_chains.append(c)
            rigid_bones += len(J) - 1
            roots.append(None)
            continue
        rigid_bones += L
        roots.append(L)
    ref = None
    for c, nm in enumerate(names):
        if roots[c] is None:
            continue
        g = sa.bone_groups.add()
        g.comment = f'HairGen chain {c}'
        g.stiffiness = float(P['stiffness'][c])
        g.drag_force, g.gravity_power, g.hit_radius = DRAG, GRAVITY, HIT
        g.gravity_dir = (0.0, 0.0, -1.0)
        if center:
            g.center.bone_name = center
        b = g.bones.add()
        b.bone_name = nm[roots[c]]
        for u in col_uuids:
            r = g.collider_groups.add()
            r.collider_group_uuid = u
        ref = g
    bad = [g.comment for g in sa.bone_groups if g.comment.startswith('HairGen') and (len(g.bones) != 1 or not g.bones[0].bone_name)]
    assert not bad, f'spring group root not set: {bad[:3]}'
    cols = {cg.uuid: cg.node.bone_name for cg in sa.collider_groups}
    info.update(n_new_chains=len(names), n_new_bones=sum(len(n) + 1 for n in names), n_spring_groups=len(names) - len(rigid_chains),
                rigid_chains=rigid_chains, n_rigid_prefix_bones=int(rigid_bones), n_bones_total=int(sum(len(n) for n in names)),
                spring_root_index=roots, leading_run_end_collider=cause, simulated_joints_inside_at_rest=int(later_inside),
                n_rigid_bones_if_root_at_first_outside_joint=int(rigid_alt), verts_three_influences=int(n3),
                hair_verts=len(Vb), hair_faces=len(P['F']),
                verts_head_only=int(((ch < 0) | (w <= 0)).sum()), colliders_used=[cols.get(u) for u in col_uuids],
                collider_refs_per_group=len(ref.collider_groups) if ref else 0, center=center, colour=[round(x, 4) for x in colour])
