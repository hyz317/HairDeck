#!/bin/bash
# Run Blender with every user, cache and temp path inside its own folder (third_party/blender/ of the workspace: the
# Blender build, user/ with the VRM Add-on, home/, tmp/), so that nothing is written under ~ or /tmp. Installed as
# third_party/blender/bl.sh by scripts/setup/download_weights.py; hairdeck.vrm runs `bash bl.sh -b --python ...`.
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$B/home" "$B/tmp"
export HOME=$B/home XDG_CONFIG_HOME=$B/home/.config XDG_CACHE_HOME=$B/home/.cache XDG_DATA_HOME=$B/home/.local/share
export TMPDIR=$B/tmp TEMP=$B/tmp TMP=$B/tmp
export BLENDER_USER_RESOURCES=$B/user BLENDER_USER_CONFIG=$B/user/config BLENDER_USER_SCRIPTS=$B/user/scripts
export BLENDER_USER_DATAFILES=$B/user/datafiles BLENDER_USER_EXTENSIONS=$B/user/extensions
export __GL_SHADER_DISK_CACHE_PATH=$B/home/nvcache MESA_SHADER_CACHE_DIR=$B/home/mesacache
export BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION=true
unset DISPLAY WAYLAND_DISPLAY
exec "$B/blender-4.2.23-linux-x64/blender" "$@"
