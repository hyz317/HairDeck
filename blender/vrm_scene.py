"""Blender scene helpers for rendering VRoid VRMs (Blender 4.2 LTS with the VRM Add-on for Blender, MIT, extension
4.10.2): import, humanoid lookup, arm poses (two-bone IK), cameras, lighting, MToon outline toggle and the label pass.
Used by gen_data.py."""
import math

import bpy
from mathutils import Quaternion, Vector

IMPORT_KW = dict(use_addon_preferences=False, extract_textures_into_folder=False, make_new_texture_folder=False,
                 enable_mtoon_outline_preview=True)


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.cameras,
                 bpy.data.lights, bpy.data.node_groups):
        for b in list(coll):
            coll.remove(b)


def import_vrm(path):
    clear_scene()
    bpy.ops.import_scene.vrm(filepath=path, **IMPORT_KW)
    return next(o for o in bpy.data.objects if o.type == 'ARMATURE')


def human_bones(arm):
    return {hb.bone: hb.node.bone_name for hb in arm.data.vrm_addon_extension.vrm0.humanoid.human_bones if hb.node.bone_name}


def hair_objects():
    return [o for o in bpy.data.objects if o.type == 'MESH' and 'hair' in o.name.lower()]


def bone_head(arm, name):
    return arm.data.bones[name].head_local.copy()


def body_frame(arm, hb):
    """Rest-pose landmarks and a size scale (1.0 ~ hips-to-head distance of 0.6 m)."""
    head, hips = bone_head(arm, hb['head']), bone_head(arm, hb['hips'])
    chest = bone_head(arm, hb.get('upperChest', hb.get('chest', hb['spine'])))
    s = (head - hips).length / 0.6
    return dict(head=head, head_c=head + Vector((0, 0, 0.09 * s)), chest=chest, hips=hips, s=s)


def _world_to_basis(arm, bone, q_world, q_parent_world=Quaternion()):
    """Pose-bone basis whose armature-space rotation (relative to rest) is q_world, given the parent's q_parent_world."""
    B = arm.data.bones[bone].matrix_local.to_quaternion()
    return B.inverted() @ q_parent_world.inverted() @ q_world @ B


def axis_rot(arm, bone, *axis_angles):
    """Basis rotation = successive rotations about rest-frame world axes (degrees)."""
    Q = Quaternion()
    for axis, deg in axis_angles:
        Q = Quaternion(Vector(axis), math.radians(deg)) @ Q
    return _world_to_basis(arm, bone, Q)


def arm_ik(arm, hb, side, target, pole):
    """Two-bone IK in the rest frame: basis rotations for upper/lower arm so the hand reaches `target`."""
    up, lo, hand = hb[f'{side}UpperArm'], hb[f'{side}LowerArm'], hb[f'{side}Hand']
    S, E0, H0 = bone_head(arm, up), bone_head(arm, lo), bone_head(arm, hand)
    l1, l2 = (E0 - S).length, (H0 - E0).length
    d_vec = target - S
    d = min(max(d_vec.length, 1e-4), (l1 + l2) * 0.999)
    dirv = d_vec.normalized()
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    p = Vector(pole) - Vector(pole).dot(dirv) * dirv
    p = p.normalized() if p.length > 1e-6 else Vector((0, 0, -1))
    E = S + a * dirv + h * p
    T = S + d * dirv
    q1 = (E0 - S).normalized().rotation_difference((E - S).normalized())
    q2 = (H0 - E0).normalized().rotation_difference((T - E).normalized())
    return {up: _world_to_basis(arm, up, q1), lo: _world_to_basis(arm, lo, q2, q1)}


def arm_dir(arm, hb, side, d_upper, d_fore):
    """Arm pose from upper-arm and forearm directions (rest frame)."""
    up, lo, hand = hb[f'{side}UpperArm'], hb[f'{side}LowerArm'], hb[f'{side}Hand']
    S, E0, H0 = bone_head(arm, up), bone_head(arm, lo), bone_head(arm, hand)
    q1 = (E0 - S).normalized().rotation_difference(Vector(d_upper).normalized())
    q2 = (H0 - E0).normalized().rotation_difference(Vector(d_fore).normalized())
    return {up: _world_to_basis(arm, up, q1), lo: _world_to_basis(arm, lo, q2, q1)}


def pose_preset(arm, hb, name, rng=None, jitter=0.0, mirror=False, face_box=None, head_box=None, head_q=None, head_pivot=None):
    """Basis rotations for the arms. +X is the character's left, -Y is front, +Z is up.
    rng + jitter: Gaussian noise (std jitter * body scale) on every IK target and on the A-pose angle;
    mirror: swap the roles of the left and right arm (for the one-handed poses);
    face_box / head_box: rest-pose (min, max) of the face mesh / of face + hair above the chin, so hands near the head
    touch its actual surface instead of a bone-based guess (big-headed characters otherwise get hands inside the face);
    head_q, head_pivot: the neck + head rotation that will be applied, so those targets follow the turned head."""
    f = body_frame(arm, hb)
    Hc, C, P, s = f['head_c'], f['chest'], f['hips'], f['s']
    if face_box is None:  # bone-based guesses
        face_box = (Hc - s * Vector((0.085, 0.10, 0.12)), Hc + s * Vector((0.085, 0.10, 0.10)))
    if head_box is None:
        head_box = face_box
    flo, fhi = face_box
    hlo, hhi = head_box
    fc, fw = (flo + fhi) / 2, (fhi - flo) / 2

    def hx(p):  # a point attached to the head
        return p if head_q is None else head_pivot + head_q @ (p - head_pivot)

    def hd(v):  # a direction attached to the head
        return Vector(v) if head_q is None else head_q @ Vector(v)
    sh = {k: bone_head(arm, hb[f'{k}UpperArm']) for k in ('left', 'right')}
    sg = {'left': 1, 'right': -1}
    specs = []  # ('ik', side, target, pole) or ('dir', side, upper_dir, fore_dir)

    def ik(side, target, pole, head=False):
        specs.append(('ik', side, Vector(target), Vector(pole), head))

    def relaxed(side):
        g = sg[side]
        ik(side, sh[side] + s * Vector((g * 0.07, -0.03, -0.52)), (0, 1, 0))

    def hip(side):
        g = sg[side]
        ik(side, P + s * Vector((g * 0.15, -0.01, 0.06)), (g * 1, 0.4, 0))

    if name == 'T':
        pass
    elif name == 'A':
        ang = 40 + (rng.normal(0, 6) if rng is not None and jitter > 0 else 0)
        for k, g in sg.items():
            v = (g * math.cos(math.radians(ang)), 0, -math.sin(math.radians(ang)))
            specs.append(('dir', k, Vector(v), Vector(v), False))
    elif name == 'relaxed':
        relaxed('left'), relaxed('right')
    elif name == 'hand_in_hair':  # left hand at the side/back of the head, in the hair
        ik('left', Vector((fhi.x + 0.02 * s, fc.y + 0.4 * fw.y, fc.z + 0.1 * fw.z)), (1, 0.2, -0.4), True), relaxed('right')
    elif name == 'v_near_face':  # right hand a little in front of the right cheek
        ik('right', Vector((fc.x - 0.45 * fw.x, flo.y - 0.06 * s, fc.z - 0.15 * fw.z)), (-0.4, 0, -1), True), relaxed('left')
    elif name == 'hand_on_cheek':  # right palm against the right side of the jaw
        ik('right', Vector((flo.x - 0.015 * s, fc.y - 0.5 * fw.y, fc.z - 0.55 * fw.z)), (-0.3, 0, -1), True), relaxed('left')
    elif name == 'hand_on_head':  # left hand on top of the head (hair surface)
        # IK places the wrist: put it at the upper side of the head so the hand lies over the top; cap the top at the
        # face top + 0.05 s so tall accessories in the hair mesh (crowns) do not lift it
        ik('left', Vector((fc.x + 0.85 * fw.x, fc.y, min(hhi.z, fhi.z + 0.05 * s) - 0.03 * s)), (1, 0, 0), True), relaxed('right')
    elif name == 'arms_behind_head':  # hands behind the head (hair surface)
        for k, g in sg.items():
            ik(k, Vector((fc.x + g * 0.4 * fw.x, hhi.y + 0.02 * s, fc.z + 0.2 * fw.z)), (g * 1, -0.2, 0.3), True)
    elif name == 'crossed_arms':
        ik('left', C + s * Vector((-0.10, -0.13, -0.06)), (0.6, 0, -1))
        ik('right', C + s * Vector((0.10, -0.15, -0.03)), (-0.6, 0, -1))
    elif name == 'hands_on_hips':
        hip('left'), hip('right')
    elif name == 'one_hand_on_hip':
        hip('left'), relaxed('right')
    elif name == 'hand_on_chest':
        ik('right', C + s * Vector((0.02, -0.13, 0.02)), (-0.6, 0, -1)), relaxed('left')
    elif name == 'arms_behind_back':
        for k, g in sg.items():
            ik(k, P + s * Vector((g * 0.07, 0.12, 0.04)), (g * 1, 0.5, 0))
    elif name == 'hands_together_front':
        for k, g in sg.items():
            ik(k, P + s * Vector((g * 0.03, -0.15, 0.06)), (g * 0.6, 0.2, -1))
    else:
        raise ValueError(name)
    out = {}
    cx = P.x
    for kind, side, a, b, head in specs:
        if mirror:
            side = 'right' if side == 'left' else 'left'
            a = Vector((2 * cx - a.x, a.y, a.z)) if kind == 'ik' else Vector((-a.x, a.y, a.z))
            b = Vector((-b.x, b.y, b.z))
        if head:  # after mirroring, so the hand follows the actual (unmirrored) head rotation
            a, b = hx(a), hd(b)
        if kind == 'ik':
            if rng is not None and jitter > 0:
                a = a + s * jitter * Vector(rng.normal(0, 1, 3))
            out.update(arm_ik(arm, hb, side, a, b))
        else:
            out.update(arm_dir(arm, hb, side, a, b))
    return out


POSE_GROUPS = {  # sampling mix of the data generator (informed by Danbooru pose tags)
    'natural': ['relaxed', 'hands_on_hips', 'one_hand_on_hip', 'crossed_arms', 'hand_on_chest', 'arms_behind_back', 'hands_together_front'],
    'near_head': ['hand_in_hair', 'v_near_face', 'hand_on_cheek', 'hand_on_head', 'arms_behind_head'],
    'TA': ['T', 'A'],
}
FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']


def finger_pose(arm, hb, curl):
    """Basis rotations curling the fingers. curl: {(side, finger): degrees per joint}. Fingers bend about the rest-frame
    Y axis (palm faces down in the T pose), the thumb about Z towards the palm, at half the angle."""
    out = {}
    for (side, fin), deg in curl.items():
        g = 1 if side == 'left' else -1
        axis, d = ((0, 0, 1), 0.5 * deg) if fin == 'Thumb' else ((0, 1, 0), deg)
        for seg in ('Proximal', 'Intermediate', 'Distal'):
            b = hb.get(f'{side}{fin}{seg}')
            if b:
                out[b] = axis_rot(arm, b, (axis, g * d))
    return out


# ----------------------------------------------------------------------------- render setup
def setup_render(scene, res=512, samples=16):
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    scene.eevee.taa_render_samples = samples
    if scene.world is None:
        scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True
    if 'Sun' not in bpy.data.objects:
        sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN'))
        scene.collection.objects.link(sun)
    if 'Cam' not in bpy.data.objects:
        cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam'))
        scene.collection.objects.link(cam)
        scene.camera = cam
    set_light('key_left')


LIGHTS = {  # sun euler, sun energy, world strength
    'key_left': ((0.6, 0.0, 0.4), 3.0, 0.5),
    'key_right': ((0.6, 0.0, -0.7), 3.0, 0.5),
    'back_top': ((-0.6, 0.0, 0.2), 4.0, 0.4),
    'flat': ((0.0, 0.0, 0.0), 0.3, 1.2),
}


def set_light(name):
    rot, energy, world = LIGHTS[name]
    sun = bpy.data.objects['Sun']
    sun.rotation_euler, sun.data.energy = rot, energy
    bg = bpy.context.scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = world


def set_camera(target, dist, az=0.0, el=0.0, lens=50):
    """Orbit camera: az=0 is the front (-Y), az=90 the character's left side, el>0 looks down."""
    cam = bpy.data.objects['Cam']
    cam.data.lens = lens
    a, e = math.radians(az), math.radians(el)
    off = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist
    cam.location = target + off
    cam.rotation_euler = (-off).to_track_quat('-Z', 'Y').to_euler()
    cam.data.shift_x = cam.data.shift_y = 0.0


def eval_points(objs):
    """World-space vertices of the evaluated (posed, spring-deformed) meshes, as an (N, 3) numpy array."""
    import numpy as np
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get('co', co)
        M = np.array(oe.matrix_world)
        out.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        oe.to_mesh_clear()
    return np.concatenate(out) if out else np.zeros((0, 3))


def posed_head(arm, hb):
    """World position of the posed head bone's head and the posed head-up direction."""
    bpy.context.view_layer.update()
    pb = arm.pose.bones[hb['head']]
    M = arm.matrix_world @ pb.matrix
    return M.to_translation(), (M.to_3x3() @ Vector((0, 1, 0))).normalized()


def frame_points(pts, target, dist, az=0.0, el=0.0, margin=0.08):
    """Orbit camera around `target`, then set lens and shift so the projected bbox of `pts` fills the square frame
    with `margin` on each side. Lens + shift is exactly a 2D crop of a wider shot, the same operation as cropping
    an input image around its hair."""
    import numpy as np
    set_camera(target, dist, az, el)
    cam = bpy.data.objects['Cam']
    bpy.context.view_layer.update()
    R = np.array(cam.matrix_world.to_3x3())
    q = (pts - np.array(cam.location)) @ R  # camera coordinates, looking down -Z
    t = q[:, :2] / -q[:, 2:3]
    lo, hi = t.min(0), t.max(0)
    c, h = (lo + hi) / 2, (hi - lo).max() / 2 * (1 + 2 * margin)
    sw = cam.data.sensor_width
    cam.data.sensor_fit = 'AUTO'
    cam.data.lens = 0.5 * sw / h
    cam.data.shift_x, cam.data.shift_y = c * (cam.data.lens / sw)
    return cam.data.lens


def head_objects():
    """Meshes that show the face (VRoid exports a separate Face mesh)."""
    return [o for o in bpy.data.objects if o.type == 'MESH' and 'face' in o.name.lower()]


def set_outline(on):
    for o in bpy.data.objects:
        for m in getattr(o, 'modifiers', []):
            if 'outline' in m.name.lower():
                m.show_render = on


# ----------------------------------------------------------------------------- label pass
def _emission(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission')
    e.inputs[0].default_value = (*rgb, 1)
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


class LabelPass:
    """Context manager: exact 8-bit label image (0 background, 1 hair mesh, 2 face mesh, 3 everything else), single
    channel, Raw view, no filtering / dither, one sample."""

    def __init__(self, hair_objs, face_objs):
        self.hair, self.face = {o.name for o in hair_objs}, {o.name for o in face_objs}

    def __enter__(self):
        sc = bpy.context.scene
        bg = sc.world.node_tree.nodes['Background']
        self.saved = dict(transp=sc.render.film_transparent, samples=sc.eevee.taa_render_samples, filt=sc.render.filter_size,
                          dither=sc.render.dither_intensity, view=sc.view_settings.view_transform, mode=sc.render.image_settings.color_mode,
                          world=bg.inputs[1].default_value, sun=bpy.data.objects['Sun'].data.energy,
                          mats={(o.name, i): sl.material for o in bpy.data.objects if o.type == 'MESH' for i, sl in enumerate(o.material_slots)},
                          outline={(o.name, m.name): m.show_render for o in bpy.data.objects for m in getattr(o, 'modifiers', [])})
        lab = {k: _emission(f'_label_{k}', (k / 255,) * 3) for k in (1, 2, 3)}
        for o in bpy.data.objects:
            if o.type == 'MESH':
                k = 1 if o.name in self.hair else 2 if o.name in self.face else 3
                for sl in o.material_slots:
                    sl.material = lab[k]
        set_outline(False)
        sc.render.film_transparent = False
        sc.eevee.taa_render_samples = 1
        sc.render.filter_size = 0.0
        sc.render.dither_intensity = 0.0
        sc.view_settings.view_transform = 'Raw'
        sc.render.image_settings.color_mode = 'BW'
        bg.inputs[1].default_value = 0.0
        bpy.data.objects['Sun'].data.energy = 0.0
        return self

    def __exit__(self, *a):
        sc = bpy.context.scene
        for (on, i), m in self.saved['mats'].items():
            bpy.data.objects[on].material_slots[i].material = m
        for (on, mn), v in self.saved['outline'].items():
            bpy.data.objects[on].modifiers[mn].show_render = v
        sc.render.film_transparent = self.saved['transp']
        sc.eevee.taa_render_samples = self.saved['samples']
        sc.render.filter_size = self.saved['filt']
        sc.render.dither_intensity = self.saved['dither']
        sc.view_settings.view_transform = self.saved['view']
        sc.render.image_settings.color_mode = self.saved['mode']
        sc.world.node_tree.nodes['Background'].inputs[1].default_value = self.saved['world']
        bpy.data.objects['Sun'].data.energy = self.saved['sun']


OUTLINE_MATERIAL_KEYS = ('hair', 'skin', 'cloth', 'tops', 'bottoms', 'shoes', 'onepiece', 'accessory')


def outline_state():
    """(number of MToon materials, number with an outline) over the non-outline MToon materials."""
    n = on = 0
    for m in bpy.data.materials:
        ext = m.vrm_addon_extension.mtoon1
        if not ext.enabled or ext.is_outline_material:
            continue
        mt = ext.extensions.vrmc_materials_mtoon
        n += 1
        on += mt.outline_width_mode != mt.OUTLINE_WIDTH_MODE_NONE.identifier and mt.outline_width_factor > 0
    return n, on


def force_outline(width):
    """Give hair / skin / clothing MToon materials a world-space outline of `width` metres (for models exported without
    outlines). The add-on rebuilds the outline modifiers when these properties are set. Returns the materials changed."""
    changed = []
    for m in list(bpy.data.materials):
        ext = m.vrm_addon_extension.mtoon1
        if not ext.enabled or ext.is_outline_material or not any(k in m.name.lower() for k in OUTLINE_MATERIAL_KEYS):
            continue
        mt = ext.extensions.vrmc_materials_mtoon
        mt.outline_width_mode = mt.OUTLINE_WIDTH_MODE_WORLD_COORDINATES.identifier
        mt.outline_width_factor = width
        changed.append(m.name)
    return changed


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
