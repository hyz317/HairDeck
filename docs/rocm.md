# AMD ROCm on Linux

HairDeck's existing PyTorch `cuda` device path also runs on HIP. This setup uses the
original models, sampler and export pipeline; it adds no precision changes, caching,
batched sampling or NPU offload. NVIDIA requirements and defaults remain unchanged.

The tested target is Ryzen AI Max+ 395 / Radeon 8060S, `gfx1151` (Strix Halo), on a
native Arch host with Docker. The container provides ROCm 7.2.3, a gfx1151-capable
PyTorch 2.13 development build and Triton 3.8, from a digest-pinned vLLM base image.
This is a target-specific working environment, not a claim of support for every AMD GPU.

## Build

```bash
docker build -f docker/rocm-gfx1151.Dockerfile -t hairdeck-rocm .
```

Two parts of the supplied base environment need rebuilding:

- Its torchvision extension lacks HIP implementations of deformable convolution,
  NMS and ROI Align. `build-torchvision.sh` builds the same torchvision source commit
  for gfx1151, retaining the base image's torch/torchvision ABI pairing.
- Its distro Mesa predates this GPU. `build-mesa.sh` installs Mesa 26.2.5 and libdrm
  2.4.134 privately under `/opt/mesa` for headless EGL rendering. This does not change
  the host graphics driver.

`requirements-rocm.txt` omits torch and torchvision. Constraints protect the base
image's ROCm packages from replacement by CUDA wheels. Do not install the regular
`requirements.txt` over this image. The default architecture build argument is
`PYTORCH_ROCM_ARCH=gfx1151`; changing it only retargets torchvision, not the base
image's prebuilt torch kernels.

## Assets and device check

Use the same downloaded weights and assets as the standard installation. SAM 3 and
DINOv3 still require their upstream license acceptance. Mount persistent workspace
and Hugging Face cache directories:

```bash
mkdir -p "$HOME/hairdeck-root" "$HOME/.cache/huggingface"
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --shm-size=16g \
  -v "$PWD:/workspace/HairDeck" \
  -v "$HOME/hairdeck-root:/workspace/hairdeck-root" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  hairdeck-rocm 'hf auth login; python scripts/setup/download_weights.py'
```

Use the same `docker run` arguments with command `python tools/rocm/smoke.py` to
exercise GPU BF16 attention, torchvision operations and an actual EGL render. The
renderer should identify the Radeon GPU, rather than llvmpipe software rendering.
The host must provide a working amdgpu/KFD driver and access to both device nodes.

## Demo

```bash
RENDER_GID=$(getent group render | cut -d: -f3) \
VIDEO_GID=$(getent group video | cut -d: -f3) \
  docker compose -f docker/compose.rocm.yml up -d
```

Open `http://127.0.0.1:7860`. Compose publishes only on host loopback. Inside the
container, `GRADIO_SERVER_NAME=0.0.0.0` allows Docker port forwarding; without this
explicit override the normal local demo still binds to `127.0.0.1`.

Override `HAIRDECK_DATA` for the host workspace, `HAIRDECK_HF_CACHE` for its HF cache,
or `HAIRDECK_IMAGE` for the image tag. These Compose mount variables do not replace
the standard `HAIRDECK_ROOT` value inside the container. The sample keeps compiler
and MIOpen caches in the persistent workspace. First-use compilation/convolution
search can be substantially slower than warm requests.

## Real-request regression check

With assets installed, run the same harness and environment against the unchanged
upstream checkout and this branch:

```bash
python tools/rocm/regression.py --repo /path/to/checkout --out /new/result/directory
```

The check runs example 2, seeds 1 and 2, 50 steps, CFG 2, colors, chains and GLBs.
It saves the first request and three warm request timings, tensor allocation peaks,
raw token arrays and hashes. It then validates a VRM export, rest preview and 90-frame
motion preview. Compare output arrays and JSON between revisions on the **same GPU**;
different GPUs need not produce identical diffusion samples. Disk caches are retained,
so these are not empty-cache startup measurements. Allocator peaks exclude other
processes and do not represent total physical memory use.

`--no-exports` explicitly records inference-only coverage for environments without
a working Blender/renderer; it must not be reported as a full workflow pass.

The [recorded compatibility comparison](measurements/rocm-compat/README.md) covers
Strix Halo, RTX 5060 Ti and PGX GB10, with exact per-device output hashes, allocation
measurements and explicit export-coverage limits.
