#!/usr/bin/env bash
# Rebuilds the image's torchvision (same version and commit) with its HIP kernels for the target GPU. The base image's
# build has CPU-only ops: torchvision::deform_conv2d (BiRefNet's decoder), nms and roi_align (SAM 3) fail on the GPU
# with "Could not run 'torchvision::...' with arguments from the 'CUDA' backend".
set -euo pipefail
ARCH=${PYTORCH_ROCM_ARCH:-gfx1151}
REV=$(python3 -c "import torchvision; print(torchvision.version.git_version)")
VER=$(python3 -c "import torchvision; print(torchvision.__version__.split('+')[0])")
work=$(mktemp -d)
git clone -q https://github.com/pytorch/vision.git "$work/vision"
cd "$work/vision"
git checkout -q "$REV"
export FORCE_CUDA=1 PYTORCH_ROCM_ARCH="$ARCH" MAX_JOBS=${MAX_JOBS:-$(nproc)} BUILD_VERSION="$VER+${REV:0:7}"
export TORCHVISION_USE_VIDEO_CODEC=0 TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_FFMPEG=0
python3 setup.py bdist_wheel -q
python3 -m pip install --no-deps --force-reinstall --break-system-packages dist/torchvision-*.whl
cd /
rm -rf "$work"
python3 -c "import torchvision; print('torchvision', torchvision.__version__)"
[ "${SKIP_GPU_CHECK:-0}" = 1 ] && exit 0  # docker build: no GPU
python3 - <<'PY'
import torch, torchvision.ops as o
b = torch.rand(10, 4, device='cuda'); b[:, 2:] += b[:, :2]
o.nms(b, torch.rand(10, device='cuda'), 0.5)
x = torch.randn(1, 8, 16, 16, device='cuda')
o.deform_conv2d(x, torch.zeros(1, 18, 16, 16, device='cuda'), torch.randn(8, 8, 3, 3, device='cuda'), padding=1)
print('torchvision GPU ops ok:', torch.__version__, __import__('torchvision').__version__)
PY
