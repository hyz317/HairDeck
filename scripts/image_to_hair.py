"""Hair cards, spring-bone chains and GLB files from character images with the released image model
(hairdeck.infer.image.Pipeline: preprocessing, DINOv3, image DiT, card VAE, chain DiT).

python image_to_hair.py IMAGE [IMAGE ...] --out OUT_DIR [--seed 0] [--n 1] [--steps 50] [--cfg 2.0] [--no-chains]
       [--ckpt CKPT] [--chain_ckpt CKPT]
Writes OUT_DIR/<image name>/ per image: input.png, prep.png (masks and crop box), model_input.png, and per variation
(seeds seed .. seed + n - 1) cards_seed<s>.json, tokens_seed<s>.npy, chains_seed<s>.json, hair_seed<s>.glb (cards and
head proxy), hair_seed<s>_nohead.glb, the *_chains.glb pair with the chains drawn; info.json with timings and notes;
OUT_DIR/summary.json. Exits with an error when an image fails.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hairdeck.infer.image import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=1, help='variations per image (1..4), seeds seed, seed + 1, ...')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--cfg', type=float, default=2.0)
    ap.add_argument('--no-chains', action='store_true')
    ap.add_argument('--ckpt', default=None, help='image DiT checkpoint (default: the released one under paths.ckpt())')
    ap.add_argument('--chain_ckpt', default=None, help='chain DiT checkpoint, one on the card VAE of --ckpt (default: the '
                    'released one under paths.ckpt())')
    a = ap.parse_args()
    names = [os.path.splitext(os.path.basename(f))[0] for f in a.images]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:  # each image writes OUT_DIR/<its name>/ and one entry of summary.json
        sys.exit(f'images with the same name {dup}: rename them or run them separately')
    P = Pipeline(ckpt=a.ckpt, chain_ckpt=a.chain_ckpt, chains=not a.no_chains)
    print(f'loaded in {P.load_sec} s' + ('' if P.chain is not None or a.no_chains else f'; chains off ({P.chain_error})'),
          flush=True)
    summary = {}
    for f in a.images:
        name = os.path.splitext(os.path.basename(f))[0]
        r = P.run(f, seed=a.seed, steps=a.steps, cfg=a.cfg, n_samples=a.n, out_dir=os.path.join(a.out, name),
                  chains=not a.no_chains)
        summary[name] = dict(ok=r['ok'], message=r['message'], status=r['status'], notes=r['notes'],
                             n_cards=[s['n_cards'] for s in r['samples']], n_chains=[s['n_chains'] for s in r['samples']],
                             timings=r['timings'])
        print(name, 'ok' if r['ok'] else 'FAILED', summary[name]['n_cards'], r['message'], flush=True)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    if not all(s['ok'] for s in summary.values()):
        sys.exit(f'FAILED: {[n for n, s in summary.items() if not s["ok"]]}')


if __name__ == '__main__':
    main()
