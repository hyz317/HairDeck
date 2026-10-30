#!/bin/bash
# Head-frame chain DiT chain_model: the exact command of the released run (one GPU, 30.5M parameters) into
# ckpt/train_chain_model, on the card VAE of launch_vae.sh, then the chain test (scripts/test.py chains, on the
# artists' cards) of the 60K and 80K checkpoints (the released checkpoint is ema_080000.pt).
# Needs the head-frame data (data/head_frame_{train,test}.npz, data/token_stats_head.json; scripts/data.py;
# ckpt/<vae>/latents_{train,test}_head.npy, ckpt/<vae>/latent_stats_head.json; launch_vae.sh) and
# data/spring/chains_{train,test}_head.npz + chain_stats_head.json (scripts/data.py).
# usage: launch_chain_dit.sh [GPU]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${1:-0} OMP_NUM_THREADS=4
$PY "$HERE/train_chain_dit.py" --name train_chain_model --vae train_card_vae --steps 80000
for s in 060000 080000; do
  $PY "$HERE/../test.py" chains --ckpt ckpt/train_chain_model/ema_$s.pt --out results/train_chain_model_s$s
done
