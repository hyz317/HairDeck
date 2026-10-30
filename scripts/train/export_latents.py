"""Encode all train / test cards with a card VAE, in the dataset frame and in the head frame (the targets of the DiTs):
  ckpt/<vae>/latents_{train,test}.npy       the cards of data/cards_train_pad<L>.npz and data/test_templates.npz;
                                            latent_stats.json, latent mu / sd over the train cards (point-cloud DiT)
  ckpt/<vae>/latents_{train,test}_head.npy  the same cards re-encoded after scaling every card by its hairstyle's k
                                            (hairdeck.data.head_frame; card width and thickness scale by k; hairstyles
                                            without a head fit keep their dataset-frame latents and are never used in
                                            the head frame); latent_stats_head.json, latent mu / sd over the cards of
                                            valid train hairstyles (image DiT, chain DiT)
and prints a VAE reconstruction check on 20K train cards (head frame vs original).

python export_latents.py card_vae [--keep_stats] [--device cuda]      (one GPU)
Needs ckpt/<vae>, data/train_templates.npz (its padded cache data/cards_train_pad<L>.npz is made on first use, as
scripts/train/train_vae.py does), data/test_templates.npz,
data/cards_{train,test}.npz and the head frame data/head_frame_{train,test}.npz (scripts/data.py).
--keep_stats: keep ckpt/<vae>/latent_stats_head.json, the file the released models were trained with (it comes with
the checkpoints); with the released card VAE the latents are then those of the released models.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hairdeck import paths  # noqa: E402
from hairdeck.cards.packed import Packed  # noqa: E402
from hairdeck.cards.vae import build_pad_cache, load_vae, pad_cards  # noqa: E402
from hairdeck.data.head_frame import card_hair, encode_head, recon_err  # noqa: E402


def head_latents(vae, split, X, lens, lat):
    """lat (the dataset-frame latents of the cards X, lens of a split) with the cards of every hairstyle with a head fit
    re-encoded after scaling by its k -> (latents, the re-encoded rows, k of every card)."""
    hz, c = np.load(paths.data(f'head_frame_{split}.npz')), np.load(paths.data(f'cards_{split}.npz'))
    if split == 'test':  # head_frame_test.npz follows test_meta.json, whose rows 0..75 are the test hairstyles with cards
        assert (c['cond_idx'] == np.arange(len(c['cond_idx']))).all()
    card_h = card_hair(c['hair_ptr'])
    assert len(card_h) == len(X) == len(lat)
    kc, vc = hz['k'][card_h], hz['valid'][card_h]
    idx = np.nonzero(vc)[0]
    return encode_head(vae, X, lens, lat, kc, idx), idx, kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', help='card VAE: a checkpoint directory under the checkpoint root')
    ap.add_argument('--keep_stats', action='store_true', help='keep latent_stats_head.json')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    d = paths.ckpt(a.name)
    vae = load_vae(d, a.device)
    z = np.load(build_pad_cache(vae.L))
    X, lens = z['X'], z['lens']
    ztr = vae.encode_padded(X, lens)
    Xte, lte = pad_cards(Packed(paths.data('test_templates.npz')), vae.L)
    zte = vae.encode_padded(Xte, lte)
    np.save(f'{d}/latents_train.npy', ztr)
    np.save(f'{d}/latents_test.npy', zte)
    sd = ztr.std(0)
    json.dump({'mu': ztr.mean(0).tolist(), 'sd': sd.tolist()}, open(f'{d}/latent_stats.json', 'w'))
    print(a.name, 'train', ztr.shape, 'test', zte.shape, 'latent sd min/median/max %.4f %.4f %.4f' % (sd.min(), np.median(sd), sd.max()), flush=True)

    htr, idx, kc = head_latents(vae, 'train', X, lens, ztr)
    np.save(f'{d}/latents_train_head.npy', htr)
    if not a.keep_stats:
        json.dump({'mu': htr[idx].mean(0).tolist(), 'sd': htr[idx].std(0).tolist()}, open(f'{d}/latent_stats_head.json', 'w'))
    s = idx[np.random.default_rng(0).choice(len(idx), min(20000, len(idx)), replace=False)]
    e0 = recon_err(vae, X[s], lens[s], ztr[s])
    e1 = recon_err(vae, X[s] * kc[s, None, None], lens[s], htr[s])
    print('VAE reconstruction on 20K cards: original xyz err / length %.4f width err %.4f | head frame %.4f %.4f' % (e0 + e1))
    print('head frame: train cards %d, re-encoded %d, k of sampled cards p5/p50/p95 %s' % (
        len(X), len(idx), np.percentile(kc[s], [5, 50, 95]).round(3)), flush=True)
    hte, idx, kc = head_latents(vae, 'test', Xte, lte, zte)
    np.save(f'{d}/latents_test_head.npy', hte)
    print('head frame: test cards %d, re-encoded %d, k p5/p50/p95 %s' % (len(Xte), len(idx), np.percentile(kc[idx], [5, 50, 95]).round(3)))


if __name__ == '__main__':
    main()
