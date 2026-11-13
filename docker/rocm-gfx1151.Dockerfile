# HairDeck on AMD ROCm, tested on the Radeon 8060S (gfx1151, Strix Halo).
# Base: the vLLM ROCm image, for its gfx1151-capable PyTorch 2.13 / ROCm 7.2.3 / Triton 3.8 builds.
# Build:  docker build -f docker/rocm-gfx1151.Dockerfile -t hairdeck-rocm .
# Run:    see docs/rocm.md
ARG BASE=vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive

# Headless radeonsi Mesa for pyrender's EGL path: Ubuntu 22.04's Mesa 23.2 does not know RDNA3.5.
COPY docker/build-mesa.sh /tmp/build-mesa.sh
RUN bash /tmp/build-mesa.sh && rm /tmp/build-mesa.sh

# ffmpeg for previews; libglvnd's EGL/GL front ends; X/GL libraries Blender 4.2 loads even in --background.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libegl1 libgl1 libgles2 libopengl0 \
      libglu1-mesa libxi6 libxrender1 libxkbcommon0 libsm6 libxxf86vm1 libxfixes3 libx11-6 libxext6 \
 && rm -rf /var/lib/apt/lists/*
# glvnd: only our Mesa vendor; the distro 23.2 libEGL_mesa (if any) is never picked.
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/opt/mesa/share/glvnd/egl_vendor.d/50_mesa.json

# HairDeck's Python dependencies go into the image's Python next to its ROCm torch / torchvision / triton, which the
# constraints pin (a venv makes uv resolve a CUDA torch from PyPI instead).
COPY requirements-rocm.txt requirements-demo.txt docker/rocm-constraints.txt /tmp/
RUN uv pip install --system --break-system-packages -c /tmp/rocm-constraints.txt \
      -r /tmp/requirements-rocm.txt -r /tmp/requirements-demo.txt \
 && uv pip install --system --break-system-packages --no-deps pyrender==0.1.45 \
 && python3 -c "import torch; assert torch.version.hip, torch.__version__"

# torchvision with its HIP kernels (the base image's build has CPU-only deform_conv2d / nms / roi_align). Built for
# gfx1151 by default; the build needs the GPU only for the final check, so docker build runs it with the GPU optional.
ARG PYTORCH_ROCM_ARCH=gfx1151
COPY docker/build-torchvision.sh /tmp/build-torchvision.sh
RUN SKIP_GPU_CHECK=1 PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} bash /tmp/build-torchvision.sh && rm /tmp/build-torchvision.sh

ENV PYOPENGL_PLATFORM=egl HAIRDECK_ROOT=/workspace/hairdeck-root HF_HOME=/root/.cache/huggingface \
    MIOPEN_USER_DB_PATH=/workspace/hairdeck-root/.cache/miopen \
    TRITON_CACHE_DIR=/workspace/hairdeck-root/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/workspace/hairdeck-root/.cache/inductor
WORKDIR /workspace/HairDeck
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["exec sleep infinity"]
