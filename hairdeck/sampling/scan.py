"""CHARM-protocol surface sampling without CHARM: a fast, bitwise-identical replacement for the mesh_to_sdf 'scan'
surface point cloud (get_surface_point_cloud(mesh, 'scan', None, 100, 400, ...), which CHARM's sample.py uses for the
point clouds of the CHARM evaluation protocol).

The original renders each of the 100 views with a new pyrender Scene and a new EGL OffscreenRenderer (re-uploading the
mesh, re-initialising and terminating EGL every view), then back-projects every view with numpy. Here, per call (or per
Scanner, reusable across calls):
  * one OffscreenRenderer, Scene and uploaded mesh with mesh_to_sdf's shader (shaders/mesh.{vert,frag}), its
    multisampling suppression and render flags; view 0 goes through pyrender's Renderer.render, views 1..99 only change
    the camera (same V uniform through pyrender's own code, clear, same draw call);
  * readback: pyrender's two blits into its main framebuffer, then an exact blit (GL_NEAREST, same formats) of the pixel
    rectangle that can contain the mesh (projected bbox + 2 px) into an atlas framebuffer, one glReadPixels per atlas fill;
  * pyrender's depth linearisation and mesh_to_sdf's back-projection / normal flipping run once over the hit pixels of
    all views, with the same numpy operations on the same dtypes and the hits in the same order, so the candidate points,
    normals and the random pick are bitwise equal to the original scan.

    fast_surface_point_cloud(mesh) -> (points (N,3) float64, normals (N,3) float64)  == get_surface_point_cloud(...).points/.normals
    fast_scan_sample(mesh, n, seed) -> (points (n,3), normals (n,3))   n points picked with np.random.default_rng(seed)
    with Scanner() as s: fast_scan_sample(m, scanner=s)                one EGL context for many meshes in one process
    scan_runs([run])   RUN/JsonResults/<name>.glb -> RUN/PointClouds/<name>.ply, as CHARM's sample.py writes them

CHARM's sample.py picks the points of a mesh with an unseeded generator; scan_run picks those of the i-th mesh of a run
(sorted names) with np.random.default_rng(seed + i), so its point clouds are reproducible (every point is one of the
candidate points of CHARM's own scan). Existing ply files are kept.

GPU: the EGL device is egl_device (Scanner / fast_* argument) or else EGL_DEVICE_ID (pyrender's default 0), not
CUDA_VISIBLE_DEVICES. pyrender is imported on the first scan; the multisampling suppression is installed on pyrender's
renderer module then, so the import order of pyrender, OpenGL and mesh_to_sdf (if used too) does not matter.
Worker processes: create Scanners inside the workers; 'fork' only from a parent that never rendered with pyrender (the
children would hang: Scanner() raises there instead), 'spawn' always works. Other pyrender users in the same process are
fine: if one terminates the EGL display, the Scanner rebuilds its context.

Derived from mesh_to_sdf (https://github.com/marian42/mesh_to_sdf, 0.0.15) and pyrender (https://github.com/mmatl/pyrender,
0.1.45, used as a library). Copied from mesh_to_sdf 0.0.15: the shaders shaders/mesh.vert and shaders/mesh.frag, the
multisampling suppression of pyrender_wrapper.py (in _gl), get_rotation_matrix and get_camera_transform_looking_at_origin
of scan.py and get_equidistant_camera_angles of surface_point_cloud.py (camera_poses). Their licences:
  mesh_to_sdf: MIT License, Copyright (c) 2020 Marian Kleineberg.
  pyrender:    MIT License, Copyright (c) 2019 Matthew Matl.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright
  notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import ctypes
import math
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from .. import ply  # noqa: E402

FOV = 1.0472  # surface_point_cloud.create_from_scans

# fork safety: a process forked from a parent that had already initialised EGL through pyrender (this module or the original
# mesh_to_sdf; pyrender imports pyrender.platforms.egl only when it creates an EGL OffscreenRenderer) hangs when it renders
# on NVIDIA EGL. Flag such children at fork time so Scanner() raises instead of hanging. HAIRDECK_ALLOW_FORKED_EGL=1 skips
# the check.
_FORKED_AFTER_EGL = [False]


def _after_fork_in_child():
    if 'pyrender.platforms.egl' in sys.modules:
        _FORKED_AFTER_EGL[0] = True


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork_in_child)


SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shaders')  # mesh_to_sdf's normal shader


class _GLState:
    suppress_multisampling = False  # mesh_to_sdf.pyrender_wrapper's flag


_STATE = _GLState()
_PYRENDER = []


def _gl():
    """(state, pyrender); on the first call, mesh_to_sdf's multisampling suppression is installed on the two names that
    pyrender's renderer module calls them by (mesh_to_sdf patches OpenGL.GL before pyrender imports them; pyrender.renderer
    is the only module that calls them): while state.suppress_multisampling is set, glEnable(GL_MULTISAMPLE) disables
    multisampling and glRenderbufferStorageMultisample allocates a renderbuffer without samples."""
    if not _PYRENDER:
        from OpenGL import GL
        import pyrender
        from pyrender import renderer
        enable, storage = renderer.glEnable, renderer.glRenderbufferStorageMultisample

        def gl_enable(value):
            if _STATE.suppress_multisampling and value == GL.GL_MULTISAMPLE:
                GL.glDisable(value)
            else:
                enable(value)

        def gl_renderbuffer_storage_multisample(target, samples, internalformat, width, height):
            if _STATE.suppress_multisampling:
                GL.glRenderbufferStorage(target, internalformat, width, height)
            else:
                storage(target, samples, internalformat, width, height)

        renderer.glEnable, renderer.glRenderbufferStorageMultisample = gl_enable, gl_renderbuffer_storage_multisample
        _PYRENDER.append(pyrender)
    return _STATE, _PYRENDER[0]


class _ShaderCache:
    """Same program as mesh_to_sdf.pyrender_wrapper.CustomShaderCache (SHADERS/mesh.vert + mesh.frag, pyrender's defines), keyed
    by the defines so one renderer can serve meshes with different vertex layouts, and with clear() so Renderer.delete() works."""

    def __init__(self):
        self.programs = {}

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        _, pyrender = _gl()
        key = tuple(sorted((defines or {}).items()))
        if key not in self.programs:
            self.programs[key] = pyrender.shader_program.ShaderProgram(os.path.join(SHADERS, 'mesh.vert'),
                                                                       os.path.join(SHADERS, 'mesh.frag'), defines=defines)
        return self.programs[key]

    def clear(self):
        for p in self.programs.values():
            p.delete()
        self.programs = {}


def _empty_hits():
    return (0, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros((0, 3), np.uint8))


_TERMINATIONS = [0]  # how many times pyrender has eglTerminate'd an EGL display in this process


def _count_terminations():
    """pyrender's Platform.__del__ -> EGLPlatform.delete_context() destroys its context AND eglTerminate's the EGL display, which is
    shared by every context on that device in the process (the original mesh_to_sdf does this once per view). That invalidates a
    Scanner kept open across calls. Wrap delete_context (in memory, behaviour unchanged) to count terminations, so a Scanner can
    notice and rebuild its context instead of using a dead (or, worse, re-used) context handle."""
    from pyrender.platforms import egl as _egl
    cls = _egl.EGLPlatform
    if getattr(cls.delete_context, '_hairdeck_counted', False):
        return
    orig = cls.delete_context

    def delete_context(self):
        if getattr(self, '_egl_display', None) is not None:
            _TERMINATIONS[0] += 1
        return orig(self)

    delete_context._hairdeck_counted = True
    cls.delete_context = delete_context


class Scanner:
    """One EGL OffscreenRenderer with mesh_to_sdf's normal shader, reusable for many meshes (scan()).

    readback='atlas' (default): crops blitted into an atlas framebuffer (atlas_rows x resolution, RGBA + DEPTH_COMPONENT24, 26 MB
        for 8192 x 400), read back once per fill. crop=False reads full images; lean=False renders every view through pyrender's
        Renderer.render (only the readback differs).
    readback='pyrender': every view through pyrender's OffscreenRenderer.render() incl. its readback (reference path).
    close() frees the GL objects and destroys the context, but does not eglTerminate the process-wide EGL display, so other live
    pyrender contexts in the process stay valid. If anything else terminates the display while the Scanner is open (e.g. a call to
    the original mesh_to_sdf sampler), the next scan() transparently builds a new context; if it happens during a scan, that scan
    is redone in a new context. Terminations are counted through pyrender's EGLPlatform.delete_context, so this covers every
    pyrender user in the process; a raw eglTerminate by non-pyrender code is only caught if eglMakeCurrent then fails."""

    def __init__(self, resolution=400, readback='atlas', crop=True, lean=True, atlas_rows=8192, egl_device=None):
        if readback not in ('atlas', 'pyrender'):
            raise ValueError(readback)
        if egl_device is not None:  # pyrender reads EGL_DEVICE_ID when it creates the context
            os.environ['EGL_DEVICE_ID'] = str(int(egl_device))
        self.resolution, self.readback, self.crop, self.lean = int(resolution), readback, crop, lean
        self.atlas_rows = int(atlas_rows)
        self.pid = os.getpid()
        self.rebuilds = 0
        self.r = None
        self._create()

    def _create(self):
        if _FORKED_AFTER_EGL[0] and os.environ.get('HAIRDECK_ALLOW_FORKED_EGL') != '1':
            raise RuntimeError('hairdeck.sampling.scan: this process was forked from a parent that had already initialised EGL (pyrender); '
                               'rendering here hangs on NVIDIA EGL. Use multiprocessing "spawn", or fork before any rendering.')
        gl_state, pyrender = _gl()
        from pyrender.platforms import egl as _egl  # runs pyrender's _ensure_egl_loaded() once (module import)
        _count_terminations()
        prev, ensure = gl_state.suppress_multisampling, _egl._ensure_egl_loaded
        gl_state.suppress_multisampling = True
        _egl._ensure_egl_loaded = lambda: None  # already installed at import; re-running it spawns `ldconfig -p` (~50 ms)
        try:
            self.r = pyrender.OffscreenRenderer(self.resolution, self.resolution)
        finally:
            gl_state.suppress_multisampling = prev
            _egl._ensure_egl_loaded = ensure
        self.generation = _TERMINATIONS[0]
        self.R = self.r._renderer
        self.R._program_cache = _ShaderCache()
        self.atlas_fb = None
        self._target = None
        if self.readback == 'atlas':
            self.R._read_main_framebuffer = self._read_into_atlas  # called at the end of Renderer._forward_pass

    def _abandon(self):
        """Drop a renderer whose context is gone without issuing any GL / EGL call: its GL names would otherwise be deleted later
        (Renderer.__del__) in whatever context is current, and Platform.__del__ would eglTerminate the display again."""
        r, self.r = self.r, None
        if r is None:
            return
        if r._renderer is not None:
            r._renderer.delete = lambda: None
        if r._platform is not None:
            r._platform._egl_context = r._platform._egl_display = None
        r._renderer = r._platform = None
        self.R = None
        self.atlas_fb = None

    def _ensure_context(self):
        if self.r is None:
            raise RuntimeError('Scanner is closed')
        if os.getpid() != self.pid:
            raise RuntimeError('Scanner was created in another process (fork); create one per worker process')
        if _TERMINATIONS[0] == self.generation:
            try:
                self.r._platform.make_current()
                return
            except Exception:  # context destroyed behind pyrender's back
                pass
        self._abandon()
        self._create()
        self.rebuilds += 1
        self.r._platform.make_current()

    # ------------------------------------------------------------------ GL
    def _alloc_atlas(self):
        from OpenGL import GL
        res = self.resolution
        self.atlas_rows = max(res, min(self.atlas_rows, int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE))))
        self.atlas_rb = [int(b) for b in GL.glGenRenderbuffers(2)]
        for b, fmt in zip(self.atlas_rb, (GL.GL_RGBA, GL.GL_DEPTH_COMPONENT24)):  # = pyrender's main framebuffer formats
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, b)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, fmt, res, self.atlas_rows)
        self.atlas_fb = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.atlas_fb)
        GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.atlas_rb[0])
        GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.atlas_rb[1])
        status = GL.glCheckFramebufferStatus(GL.GL_DRAW_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f'atlas framebuffer incomplete: {status}')
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

    def _read_into_atlas(self, scene, flags):
        """Renderer._read_main_framebuffer up to its glReadPixels (same two blits into the main framebuffer), then an exact blit of
        the image rectangle box = (r0, r1, c0, c1) (top-down rows) to atlas rows [y, y + r1 - r0), columns [0, c1 - c0)."""
        from OpenGL import GL
        R = self.R
        width, height = R._main_fb_dims[0], R._main_fb_dims[1]
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, R._main_fb_ms)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, R._main_fb)
        GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL.GL_DEPTH_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, R._main_fb)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.atlas_fb)
        (r0, r1, c0, c1), y = self._target
        h, w = r1 - r0, c1 - c0
        GL.glBlitFramebuffer(c0, height - r1, c1, height - r0, 0, y, w, y + h, GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT,
                             GL.GL_NEAREST)
        return None

    def _draw_lean(self, scene, draws, flags):
        """Renderer._forward_pass for a scene whose only change since the previous full pass is the camera pose: same V (pyrender's
        _get_camera_matrices + ShaderProgram.set_uniform), clear, same draw call; every other GL state / uniform persists."""
        from OpenGL import GL
        R = self.R
        V, _ = R._get_camera_matrices(scene)
        self.r._platform.make_current()  # as OffscreenRenderer.render does before every pass
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, R._main_fb_ms)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for primitive, program in draws:
            program._bind()
            program.set_uniform('V', V)
            primitive._bind()
            n_instances = 1 if primitive.poses is None else len(primitive.poses)
            if primitive.indices is not None:
                GL.glDrawElementsInstanced(primitive.mode, primitive.indices.size, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0), n_instances)
            else:
                GL.glDrawArraysInstanced(primitive.mode, 0, len(primitive.positions), n_instances)
            primitive._unbind()
            program._unbind()
        GL.glFlush()
        self._read_into_atlas(scene, flags)

    def _flush(self, pending, out):
        """One glReadPixels (depth GL_FLOAT, colour GL_RGB/GL_UNSIGNED_BYTE, as pyrender) of the filled atlas rows; per view: flip to
        top-down rows (pyrender's np.flip), hits = depth != 1.0 (pyrender: depth == 1.0 -> 0; Scan: argwhere(depth != 0))."""
        if not pending:
            return
        from OpenGL import GL
        from OpenGL.raw.GL.VERSION.GL_1_0 import glReadPixels as raw_read
        rows = pending[-1][2] + pending[-1][1][1] - pending[-1][1][0]
        width = max(b[3] - b[2] for _, b, _ in pending)
        depth = np.empty((rows, width), np.float32)
        color = np.empty((rows, width, 3), np.uint8)
        self.r._platform.make_current()
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.atlas_fb)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)  # as PyOpenGL's glReadPixels wrapper
        raw_read(0, 0, width, rows, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, ctypes.c_void_p(depth.ctypes.data))
        raw_read(0, 0, width, rows, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(color.ctypes.data))
        for i, (r0, r1, c0, c1), y in pending:
            D = depth[y:y + r1 - r0, :c1 - c0][::-1]
            C = color[y:y + r1 - r0, :c1 - c0][::-1]
            r, c = np.nonzero(D != 1.0)
            out[i] = (len(r), r + r0, c + c0, D[r, c], C[r, c])
        pending.clear()

    def _render(self, scene):
        gl_state, pyrender = _gl()
        prev = gl_state.suppress_multisampling
        gl_state.suppress_multisampling = True
        try:
            return self.r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        finally:
            gl_state.suppress_multisampling = prev

    # ------------------------------------------------------------------ scan
    def scan(self, mesh, poses, z_near, z_far, max_attempts=3):
        """Hit pixels of all views, in the order mesh_to_sdf visits them (view, then np.argwhere row-major order of the flipped
        image): per-view counts, rows, cols, pyrender's linear depth (float32) and the normal-buffer RGB (uint8).
        If a pyrender context anywhere in the process eglTerminate's the display DURING the scan (e.g. another user's renderer
        collected by the cyclic GC), the result is discarded, the context rebuilt and the scan redone."""
        for _ in range(max_attempts):
            self._ensure_context()
            gen = self.generation
            try:
                out = self._scan_once(mesh, poses, z_near, z_far)
            except Exception:
                if _TERMINATIONS[0] == gen:
                    raise
                continue
            if _TERMINATIONS[0] == gen:
                return out
        raise RuntimeError(f'EGL display was terminated during each of {max_attempts} scan attempts')

    def _scan_once(self, mesh, poses, z_near, z_far):
        gl_state, pyrender = _gl()
        camera = pyrender.PerspectiveCamera(yfov=FOV, aspectRatio=1.0, znear=z_near, zfar=z_far)
        scene = pyrender.Scene()
        mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        cam_node = scene.add(camera, pose=poses[0])
        zn, zf = camera.znear, camera.zfar  # python floats, as Renderer._read_main_framebuffer reads them
        res = self.resolution
        out = [None] * len(poses)
        if self.readback == 'pyrender':
            for i, pose in enumerate(poses):
                scene.set_pose(cam_node, pose)
                color, depth = self._render(scene)
                r, c = np.nonzero(depth)
                out[i] = (len(r), r, c, depth[r, c], color[r, c])
        else:
            from pyrender.constants import ProgramFlags, RenderFlags
            flags = RenderFlags.SKIP_CULL_FACES | RenderFlags.OFFSCREEN
            boxes = crop_boxes(mesh, poses, camera, res) if self.crop else [(0, res, 0, res)] * len(poses)
            if self.atlas_fb is None:
                self._alloc_atlas()
            pending, y, draws = [], 0, None
            prev = gl_state.suppress_multisampling
            gl_state.suppress_multisampling = True
            try:
                for i, pose in enumerate(poses):
                    box = boxes[i]
                    if box[1] <= box[0] or box[3] <= box[2]:
                        out[i] = _empty_hits()
                        continue
                    if y + box[1] - box[0] > self.atlas_rows:
                        self._flush(pending, out)
                        y = 0
                    scene.set_pose(cam_node, pose)
                    self._target = (box, y)
                    if draws is None or not self.lean:
                        self.r.render(scene, flags=RenderFlags.SKIP_CULL_FACES)  # full pyrender pass (+ _read_into_atlas)
                        draws = [(p, self.R._get_primitive_program(p, flags, ProgramFlags.USE_MATERIAL))
                                 for p in mesh_node.mesh.primitives]
                    else:
                        self._draw_lean(scene, draws, flags)
                    pending.append((i, box, y))
                    y += box[1] - box[0]
                self._flush(pending, out)
            finally:
                gl_state.suppress_multisampling = prev
        counts = np.array([o[0] for o in out], dtype=np.int64)
        rows, cols, depth, rgb = (np.concatenate([o[k] for o in out]) for k in range(1, 5))
        if self.readback != 'pyrender':  # Renderer._read_main_framebuffer's linearisation, same expression on the same float32 values
            depth = 2.0 * depth - 1.0
            depth = ((2.0 * zn * zf) / (zf + zn - depth * (zf - zn)))
        return counts, rows, cols, depth, rgb

    def close(self):
        if self.r is None:
            return
        if os.getpid() != self.pid or _TERMINATIONS[0] != self.generation:  # inherited through fork / context already dead
            self._abandon()
            return
        r, self.r = self.r, None
        plat = r._platform
        try:
            plat.make_current()
            if self.atlas_fb is not None:
                from OpenGL import GL
                GL.glDeleteFramebuffers(1, [self.atlas_fb])
                GL.glDeleteRenderbuffers(2, self.atlas_rb)
                self.atlas_fb = None
            r._renderer.delete()
        finally:
            if getattr(plat, '_egl_context', None) is not None:
                from OpenGL import EGL
                EGL.eglMakeCurrent(plat._egl_display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
                EGL.eglDestroyContext(plat._egl_display, plat._egl_context)
                plat._egl_context = plat._egl_display = None  # Platform.__del__ must not eglTerminate the shared display
            r._renderer = r._platform = None  # makes OffscreenRenderer.__del__ a no-op
            self.R = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def as_trimesh(mesh):
    """mesh_to_sdf.get_surface_point_cloud's input handling."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError('The mesh parameter must be a trimesh mesh.')
    return mesh


def crop_boxes(mesh, poses, camera, res, margin=2):
    """Per view, image rectangle (r0, r1, c0, c1) (top-down rows, half-open) containing every pixel the mesh can cover: the
    projection of the 8 corners of the mesh's axis-aligned box (all in front of the camera => the projected box contains the
    projection of every triangle, hence every rasterised pixel centre), widened by `margin` px, clamped to the image. A corner at or
    behind the camera plane => full image. Only decides which pixels are read back and scanned, never changes a value."""
    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    P = camera.get_projection_matrix(width=res, height=res)
    boxes = []
    for pose in poses:
        clip = corners @ (P @ np.linalg.inv(pose)).T
        if not np.all(clip[:, 3] > 1e-6 * np.abs(clip[:, :3]).max()):
            boxes.append((0, res, 0, res))
            continue
        x = (clip[:, 0] / clip[:, 3] + 1) * (res / 2)  # GL window coordinates, y up
        y = (clip[:, 1] / clip[:, 3] + 1) * (res / 2)
        c0, c1 = max(int(np.floor(x.min())) - margin, 0), min(int(np.ceil(x.max())) + margin, res)
        j0, j1 = max(int(np.floor(y.min())) - margin, 0), min(int(np.ceil(y.max())) + margin, res)
        boxes.append((res - j1, res - j0, c0, c1))
    return boxes


def get_rotation_matrix(angle, axis='y'):
    """mesh_to_sdf.scan.get_rotation_matrix (scipy >= 1.4)."""
    from scipy.spatial.transform import Rotation
    matrix = np.identity(4)
    matrix[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    return matrix


def get_camera_transform_looking_at_origin(rotation_y, rotation_x, camera_distance=2):
    """mesh_to_sdf.scan.get_camera_transform_looking_at_origin."""
    camera_transform = np.identity(4)
    camera_transform[2, 3] = camera_distance
    camera_transform = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_transform)
    camera_transform = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_transform)
    return camera_transform


def get_equidistant_camera_angles(count):
    """mesh_to_sdf.surface_point_cloud.get_equidistant_camera_angles: (phi, theta) of count views on a Fibonacci sphere."""
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta


def camera_poses(scan_count, bounding_radius):
    """The camera poses of mesh_to_sdf's surface_point_cloud.create_from_scans."""
    return [get_camera_transform_looking_at_origin(phi, theta, camera_distance=2 * bounding_radius)
            for phi, theta in get_equidistant_camera_angles(scan_count)]


def backproject(counts, rows, cols, depth, rgb, poses, projection_matrix, z_near, z_far, resolution, calculate_normals=True):
    """scan.Scan.__init__ for all views at once (same ops, same dtypes, same order). depth = pyrender's linear depth at the hits."""
    n = int(np.sum(counts))
    # Scan: depth[depth == 0] = inf; self.depth = (z_far + z_near - (2.0 * z_near * z_far) / depth) / (z_far - z_near)
    depth_clip = (z_far + z_near - (2.0 * z_near * z_far) / depth) / (z_far - z_near)
    indices = np.stack([rows, cols], axis=1)
    points = np.ones((n, 4))
    points[:, [1, 0]] = indices.astype(float) / (resolution - 1) * 2 - 1
    points[:, 1] *= -1
    points[:, 2] = depth_clip
    inv_p = np.linalg.inv(projection_matrix)
    out = np.empty((n, 4))
    s = 0
    for pose, k in zip(poses, counts):
        clipping_to_world = np.matmul(pose, inv_p)
        out[s:s + k] = np.matmul(points[s:s + k], clipping_to_world.transpose())
        s += k
    out /= out[:, 3][:, np.newaxis]
    pts = out[:, :3]
    normals = None
    if calculate_normals:
        normals = rgb / 255 * 2 - 1  # elementwise, so one call over all views gives the same values
        s = 0
        for pose, k in zip(poses, counts):  # the einsum is a reduction: run it per view, on (k, 3) arrays as Scan does
            camera_position = np.matmul(pose, np.array([0, 0, 0, 1]))[:3]
            camera_to_points = camera_position - pts[s:s + k]
            normal_orientation = np.einsum('ij,ij->i', camera_to_points, normals[s:s + k])
            normals[s:s + k][normal_orientation < 0] *= -1
            s += k
    return np.ascontiguousarray(pts), normals


def fast_surface_point_cloud(mesh, bounding_radius=None, scan_count=100, scan_resolution=400, calculate_normals=True, scanner=None,
                             **scanner_kw):
    """== mesh_to_sdf.get_surface_point_cloud(mesh, 'scan', bounding_radius, scan_count, scan_resolution, ..., calculate_normals)
    .points / .normals. scanner: an open Scanner to reuse (else one EGL context is created and destroyed in this call, on
    scanner_kw['egl_device'] if given)."""
    _, pyrender = _gl()
    mesh = as_trimesh(mesh)
    if bounding_radius is None:
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1
    z_near, z_far = bounding_radius * 1, bounding_radius * 3
    poses = camera_poses(scan_count, bounding_radius)
    projection_matrix = pyrender.PerspectiveCamera(yfov=FOV, aspectRatio=1.0, znear=z_near, zfar=z_far).get_projection_matrix()
    own = scanner is None
    if own:
        scanner = Scanner(scan_resolution, **scanner_kw)
    elif scanner.resolution != scan_resolution:
        raise ValueError('scanner resolution != scan_resolution')
    try:
        hits = scanner.scan(mesh, poses, z_near, z_far)
    finally:
        if own:
            scanner.close()
    return backproject(*hits, poses, projection_matrix, z_near, z_far, scan_resolution, calculate_normals)


def fast_scan_sample(mesh, n=10000, seed=0, scanner=None, egl_device=None):
    """n points (with replacement, np.random.default_rng(seed)) of the 'scan' surface point cloud (100 views, 400 px)."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    points, normals = fast_surface_point_cloud(mesh, None, 100, 400, calculate_normals=True, scanner=scanner, egl_device=egl_device)
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], n, replace=True)
    return points[idx], normals[idx]


def write_ply(path, points, normals):
    """The layout of CHARM sample.py's point clouds (hairdeck.ply: double positions and normals, grey colours)."""
    ply.write(path, points, normals)


def missing_point_clouds(run):
    """(i, glb, ply) of the meshes of a run without a point cloud; i: the position among all meshes (sorted names),
    which seeds the pick. A 0-byte glb (an empty generation) gets none."""
    import glob
    out = []
    for i, f in enumerate(sorted(glob.glob(f'{run}/JsonResults/*.glb'))):
        ply = f'{run}/PointClouds/{os.path.basename(f)[:-4]}.ply'
        if not os.path.exists(ply) and os.path.getsize(f) > 0:
            out.append((i, f, ply))
    return out


def scan_run(run, scanner, n=10000, seed=0):
    """The missing point clouds of a run with an open Scanner; returns how many were written."""
    os.makedirs(f'{run}/PointClouds', exist_ok=True)
    done = 0
    for i, f, out in missing_point_clouds(run):
        p, nr = fast_scan_sample(trimesh.load(f, force='mesh'), n, seed + i, scanner=scanner)
        assert (np.linalg.norm(nr, axis=-1) > 0.99).all() and (np.linalg.norm(nr, axis=-1) < 1.01).all()
        write_ply(out, p, nr)
        done += 1
    return done


def scan_runs(runs, egl_device=None, n=10000, seed=0):
    """scan_run for every run with one Scanner, which is only created when a point cloud is missing (no GPU
    otherwise): {run: point clouds written}."""
    if not any(missing_point_clouds(r) for r in runs):
        return {r: 0 for r in runs}
    with Scanner(400, egl_device=egl_device) as s:
        return {r: scan_run(r, s, n, seed) for r in runs}
