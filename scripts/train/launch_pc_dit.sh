#!/bin/bash
# Point-cloud card DiT pointcloud_model: the exact command of the released run (one GPU, 181.3M parameters, 80K
# steps; the released checkpoint is its ema_080000.pt) into ckpt/train_pointcloud_model, on the card VAE of
# launch_vae.sh, then the point-cloud test (scripts/test.py pointcloud; its scan renders through EGL on EGL_DEVICE_ID,
# default: the same GPU index).
# Needs: data/cards_train.npz and data/cond_dora_{train,test}.npy (scripts/data.py),
# ckpt/train_card_vae/{model.pt,stats.json,latents_train.npy,latent_stats.json} (launch_vae.sh).
# usage: launch_pc_dit.sh [GPU]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${1:-0} EGL_DEVICE_ID=${EGL_DEVICE_ID:-${1:-0}} OMP_NUM_THREADS=4
$PY "$HERE/train_pc_dit.py" --cond dora --name train_pointcloud_model --vae train_card_vae --dim 768 --depth 12 --heads 12 --steps 80000
$PY "$HERE/../test.py" pointcloud --ckpt ckpt/train_pointcloud_model/ema_080000.pt --out results/train_pointcloud_model_s080000
