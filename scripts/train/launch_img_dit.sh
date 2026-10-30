#!/bin/bash
# Image DiT image_model: the command of the released run (one GPU, 182.3M parameters, 100K steps, batch 48; the
# released checkpoint is its ema_100000.pt) into ckpt/train_image_model, on the card VAE of launch_vae.sh and from the
# point-cloud DiT of launch_pc_dit.sh; the released run also mixed in restyled renders (p 0.33), which this release
# does not include yet. The script sets seed 0; the released run had no fixed seed,
# so its order of training views is not reproduced (train_img_dit.py).
# Needs: the renders data/renders and data/renders_test, data/cards_train.npz, the head frame
# data/head_frame_{train,test}.npz and data/token_stats_head.json (scripts/data.py), the head-frame latents
# ckpt/train_card_vae/latents_train_head.npy and latent_stats_head.json (launch_vae.sh), ckpt/train_pointcloud_model/ema_080000.pt
# (launch_pc_dit.sh) and DINOv3 (scripts/setup/download_weights.py).
# usage: launch_img_dit.sh [GPU]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${1:-0} OMP_NUM_THREADS=2
$PY "$HERE/train_img_dit.py" --style_cfg "$HERE/../../configs/style_aug.json" --name train_image_model --vae train_card_vae \
  --res 448 --bs 48 --workers 32 --ckpt_every 10000 --steps 100000 --init ckpt/train_pointcloud_model/ema_080000.pt
