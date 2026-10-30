"""VRM 0.x spring-bone simulation inside Blender, a port of UniVRM's VRMSpringBone / VRMSpringBoneLogic
(https://github.com/vrm-c/UniVRM, MIT licence, notice below).

The VRM add-on only simulates VRM 1.0 spring bones; the VRoid models are VRM 0.x, so this re-implements the 0.x update
on top of the data the add-on imports (bone groups, colliders as bone-parented sphere empties). Everything is computed
in armature space with our own forward kinematics, so a whole simulation runs without depsgraph updates; apply() writes
the resulting rotations to the pose bones for rendering. hairdeck/spring/sim.py is the numpy version of the same
update (without Blender).

  UniVRM licence (LICENSE.txt of the repository):
  MIT License

  Copyright (c) 2020 VRM Consortium
  Copyright (c) 2018 Masataka SUMI for MToon

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
  Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from mathutils import Matrix, Quaternion, Vector


class SpringSim:
    def __init__(self, arm, only_hair=False, hair_gravity=None):
        self.arm = arm
        bones = arm.data.bones
        self.order = []  # parent-before-child traversal of all bones
        stack = [b for b in bones if b.parent is None][::-1]
        while stack:
            b = stack.pop()
            self.order.append(b.name)
            stack += list(b.children)[::-1]
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
        self.rel = {b.name: (b.parent.matrix_local.inverted() @ b.matrix_local if b.parent else b.matrix_local.copy())
                    for b in bones}
        self.basis = {n: Quaternion() for n in self.order}  # user pose (humanoid bones)
        self.loc = {}  # optional pose-bone translations (bone-local), e.g. hips for jumps
        sa = arm.data.vrm_addon_extension.vrm0.secondary_animation
        M = self.fk()
        inv_world = arm.matrix_world.inverted()
        cgroups = {}
        for cg in sa.collider_groups:
            bn = cg.node.bone_name
            if bn not in bones:
                continue
            cols = []
            for c in cg.colliders:
                o = c.bpy_object
                if o is None:
                    continue
                p = inv_world @ o.matrix_world.to_translation()
                cols.append((bn, M[bn].inverted() @ p, o.empty_display_size))
            cgroups[cg.uuid] = cols
        self.joints = []  # dicts in parent-before-child order
        jset = {}
        for g in sa.bone_groups:
            roots = [b.bone_name for b in g.bones if b.bone_name in bones]
            is_hair = any('hair' in r.lower() for r in roots)
            if only_hair and not is_hair:
                continue
            gdir = Vector(g.gravity_dir)
            gdir = gdir.normalized() if gdir.length > 1e-6 else Vector((0, 0, -1))
            gp = hair_gravity if (is_hair and hair_gravity is not None) else g.gravity_power  # augmentation: override hair gravity only
            cols = [c for ref in g.collider_groups for c in cgroups.get(ref.collider_group_uuid, [])]
            prm = dict(stiff=g.stiffiness, drag=g.drag_force, grav=gdir * gp, hit=g.hit_radius, cols=cols)
            for r in roots:
                stack = [bones[r]]
                while stack:
                    b = stack.pop()
                    stack += list(b.children)
                    if b.name in jset:
                        continue
                    if b.children:
                        tail_arm = b.children[0].head_local
                    else:  # leaf: extend 7 cm along parent->bone (UniVRM)
                        d = b.head_local - (b.parent.head_local if b.parent else b.head_local - Vector((0, 0, 1)))
                        tail_arm = b.head_local + d.normalized() * 0.07
                    tail_loc = b.matrix_local.inverted() @ tail_arm
                    j = dict(name=b.name, tail_loc=tail_loc, L=tail_loc.length, **prm)
                    jset[b.name] = j
        rank = {n: i for i, n in enumerate(self.order)}
        self.joints = sorted(jset.values(), key=lambda j: rank[j['name']])
        self.jmap = {j['name']: j for j in self.joints}
        for j in self.joints:  # the non-simulated bone each chain hangs from (head for hair, hips for skirts)
            a = self.parent[j['name']]
            while a in self.jmap:
                a = self.parent[a]
            j['anchor'] = a
        self.reset()

    def fk(self, spring_rot=None):
        """Armature-space matrices of all bones; spring_rot overrides the basis of simulated bones."""
        M = {}
        for n in self.order:
            p = self.parent[n]
            q = spring_rot.get(n, self.basis[n]) if spring_rot else self.basis[n]
            local = self.rel[n] @ q.to_matrix().to_4x4()
            if n in self.loc:
                local = self.rel[n] @ Matrix.Translation(self.loc[n]) @ q.to_matrix().to_4x4()
            M[n] = M[p] @ local if p else local
        return M

    def reset(self):
        """Put every joint tail at its current (posed, unsimulated) position with zero velocity."""
        self.rot = {}
        M = self.fk()
        for j in self.joints:
            t = M[j['name']] @ j['tail_loc']
            j['cur'], j['prev'] = t.copy(), t.copy()

    def step(self, dt=1.0 / 60):
        M = self.fk(self.rot)  # body pose (colliders) + last spring rotations
        for j in self.joints:  # parent-before-child, so M[parent] is already this step's
            n, p = j['name'], self.parent[j['name']]
            R0 = (M[p] @ self.rel[n]) if p else self.rel[n].copy()
            pos = R0.to_translation()
            rest_dir = (R0.to_3x3() @ j['tail_loc']).normalized()
            cur, prev = j['cur'], j['prev']
            nxt = cur + (cur - prev) * (1.0 - j['drag']) + rest_dir * (j['stiff'] * dt) + j['grav'] * dt
            nxt = pos + (nxt - pos).normalized() * j['L']
            for bn, off, rad in j['cols']:
                c = M[bn] @ off
                r = rad + j['hit']
                d = nxt - c
                if d.length_squared <= r * r:
                    nxt = pos + ((c + d.normalized() * r) - pos).normalized() * j['L']
            j['prev'], j['cur'] = cur, nxt
            q = rest_dir.rotation_difference((nxt - pos).normalized())
            world_rot = q @ R0.to_quaternion()
            self.rot[n] = R0.to_quaternion().inverted() @ world_rot  # basis rotation relative to rest frame
            M[n] = R0 @ self.rot[n].to_matrix().to_4x4()
        return M

    def apply(self):
        pb = self.arm.pose.bones
        for n, q in self.basis.items():
            if n in self.jmap:
                continue
            pb[n].rotation_mode = 'QUATERNION'
            pb[n].rotation_quaternion = q
        for n, q in self.rot.items():
            pb[n].rotation_mode = 'QUATERNION'
            pb[n].rotation_quaternion = q
        for n in self.order:
            pb[n].location = self.loc.get(n, Vector())

    def tails(self):
        return {j['name']: j['cur'].copy() for j in self.joints}

    def local_tails(self, M):
        """Joint tails in their anchor bone's frame (for measuring deformation independent of the body pose)."""
        return {j['name']: M[j['anchor']].inverted() @ j['cur'] for j in self.joints}
