#!/usr/bin/env bash
# Builds a headless radeonsi Mesa (EGL device/surfaceless + GBM, GL/GLES through glvnd) and a matching libdrm into
# /opt/mesa. Ubuntu 22.04's Mesa 23.2 predates RDNA3.5 (gfx1150/gfx1151): "amdgpu: unknown (family_id,
# chip_external_rev): (150, 193)" and eglInitialize fails. Shaders compile with ACO, so no LLVM is needed.
set -euo pipefail
MESA_VERSION=${MESA_VERSION:-26.2.5}
LIBDRM_VERSION=${LIBDRM_VERSION:-2.4.134}
PREFIX=${PREFIX:-/opt/mesa}
BV=/opt/mesa-build-venv
apt-get update
apt-get install -y --no-install-recommends build-essential pkg-config ninja-build bison flex curl xz-utils \
  libexpat1-dev zlib1g-dev libzstd-dev libglvnd-dev libpciaccess-dev
# Build-time Python tools in their own venv (the image's python3 is 3.12, without apt's python modules).
/usr/bin/python3.12 -m venv "$BV"
"$BV/bin/pip" install --no-cache-dir "meson>=1.6,<2" mako pyyaml packaging ply
export PATH="$BV/bin:$PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/x86_64-linux-gnu/pkgconfig:$PREFIX/lib/pkgconfig"
work=$(mktemp -d)
cd "$work"
curl -fsSL "https://dri.freedesktop.org/libdrm/libdrm-${LIBDRM_VERSION}.tar.xz" | tar xJ
curl -fsSL "https://archive.mesa3d.org/mesa-${MESA_VERSION}.tar.xz" | tar xJ
(cd "libdrm-$LIBDRM_VERSION" && meson setup b --prefix="$PREFIX" --buildtype=release \
   -Dintel=disabled -Dradeon=disabled -Dnouveau=disabled -Dvmwgfx=disabled -Dtests=false -Dman-pages=disabled \
   -Dvalgrind=disabled -Dcairo-tests=disabled -Damdgpu=enabled && ninja -C b install)
(cd "mesa-$MESA_VERSION" && meson setup b --prefix="$PREFIX" --buildtype=release \
   -Dgallium-drivers=radeonsi -Dvulkan-drivers= -Dplatforms= -Degl=enabled -Dgbm=enabled -Dglvnd=enabled \
   -Dgles1=disabled -Dgles2=enabled -Dopengl=true -Dglx=disabled -Dllvm=disabled \
   -Dvalgrind=disabled -Dlibunwind=disabled -Dzstd=enabled -Dexpat=enabled -Dbuild-tests=false \
   && ninja -C b install)
cd /
rm -rf "$work" "$BV"
echo "$PREFIX/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/00-hairdeck-mesa.conf
ldconfig
