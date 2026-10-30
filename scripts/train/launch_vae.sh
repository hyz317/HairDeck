#!/bin/bash
# Card VAE card_vae: the exact command of the released run (one GPU, about 8M parameters) into ckpt/train_card_vae (a
# folder of its own: the downloaded release stays in ckpt/card_vae), then the card latents in the dataset frame and in
# the head frame (export_latents.py).
# Needs HAIRDECK_ROOT (or a configs/paths.json) and the data of scripts/data.py (data/{train,test}_templates.npz,
# cards_{train,test}.npz, head_frame_{train,test}.npz).
# usage: launch_vae.sh [GPU]      PY=python of an env with torch, numpy, trimesh, scipy (default: python)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${1:-0} OMP_NUM_THREADS=3
$PY "$HERE/train_vae.py" --name train_card_vae --zdim 32 --nf 6 --steps 80000 --dec ada --geo_loss --uniform_target --accum 2
$PY "$HERE/export_latents.py" train_card_vae
