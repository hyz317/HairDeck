"""ROCm / gfx1151 smoke test: GPU torch, bf16 SDPA, EGL (pyrender) on radeonsi."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import torch
assert torch.version.hip and torch.cuda.is_available(), 'Requires a working ROCm PyTorch GPU'
p = torch.cuda.get_device_properties(0)
print('torch', torch.__version__, 'hip', torch.version.hip, p.name, p.gcnArchName, 'total %.1f GiB' % (p.total_memory / 2**30))
print('mem_get_info GiB', [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()])
q = torch.randn(2, 16, 1024, 64, device='cuda', dtype=torch.bfloat16)
o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
assert torch.isfinite(o).all(), 'BF16 attention produced nonfinite output'
print('sdpa ok', bool(torch.isfinite(o).all()))
import torchvision.ops as tvo
b = torch.rand(16, 4, device='cuda'); b[:, 2:] += b[:, :2]
tvo.nms(b, torch.rand(16, device='cuda'), 0.5)
tvo.roi_align(torch.randn(1, 3, 8, 8, device='cuda'), [b], 2)
x = torch.randn(1, 8, 16, 16, device='cuda')
y = tvo.deform_conv2d(x, torch.zeros(1, 18, 16, 16, device='cuda'), w := torch.randn(8, 8, 3, 3, device='cuda'), padding=1)
assert torch.allclose(y, torch.nn.functional.conv2d(x, w, padding=1), atol=1e-3), 'Deformable convolution mismatch'
print('torchvision GPU ops ok (nms, roi_align, deform_conv2d == conv2d at zero offset:',
      bool(torch.allclose(y, torch.nn.functional.conv2d(x, w, padding=1), atol=1e-3)), ')')
import pyrender, trimesh
import numpy as np
r = pyrender.OffscreenRenderer(128, 128)
s = pyrender.Scene()
s.add(pyrender.Mesh.from_trimesh(trimesh.creation.box()))
cam = np.eye(4); cam[2, 3] = 3
s.add(pyrender.PerspectiveCamera(yfov=1.0), pose=cam)
s.add(pyrender.DirectionalLight(intensity=3), pose=cam)
c, d = r.render(s)
import OpenGL.GL as g
print('GL', g.glGetString(g.GL_RENDERER).decode(), '|', g.glGetString(g.GL_VERSION).decode(), '| depth px', int((d > 0).sum()))
assert (d > 0).any(), 'EGL render produced no geometry'
assert 'llvmpipe' not in g.glGetString(g.GL_RENDERER).decode().lower(), 'Unexpected software rendering'
r.delete()
