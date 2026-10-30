"""Test-set sampling and evaluation of a released model: one command gives the paper's rows of a test (hairdeck.testset).

python test.py pointcloud|chains [--ckpt CKPT] [--seeds 0 1 2] [--out OUT] [--workers 12] [--cards gt|DIR]
  pointcloud  CHARM's 100 test point clouds: the Dora condition (computed in memory), sampling, the CHARM-protocol scan,
              the point metrics against CHARM's test point clouds, the orientation and the card metrics against the
              original meshes. Table 1: CD, EMD, HD, IoU (the CHARM protocol), Orient.
  chains      the 58 test hairstyles with chains: chains on the artists' cards (--cards gt) or on the cards of the
              output folder DIR of test.py pointcloud (chain seed k on DIR/seed<k>), against the characters'
              own chains, the hair motion they give (artists' cards only) and their fit to the cards.
              Table 4: CD, Rec., Prec., Motion, Ampl., Fit.
Each seed of --seeds (default 0 1 2; the paper's rows are their mean) samples and is evaluated into OUT/seed<k>; then
OUT/eval.json holds every seed's table and their mean, and the table is printed with one row per seed and the mean.
--ckpt: default the released checkpoint ckpt/<pointcloud | chain>_model/model.safetensors. --out: default
results/<pointcloud | chains> (chains on DIR: results/chains_on_<name of DIR>). Paths are relative to the
workspace layout (hairdeck.paths.resolve). Released setting: Euler 50, CFG 2. --workers: evaluation processes.
The scan renders through EGL on EGL_DEVICE_ID (default 0; CUDA_VISIBLE_DEVICES does not apply to it); sampling a seed
again removes its earlier point clouds, so the scan is redone.
Needs (scripts/setup/download_weights.py): pointcloud: --only hairdeck test_data charm_test_pc dora; chains: --only
hairdeck test_data.
Writes per seed RUN = OUT/seed<k>: RUN/JsonResults ({idx:03d}.json cards, .glb meshes), RUN/PointClouds (the scan),
RUN/timing.json (pointcloud), RUN/chains.json (chains), and RUN/eval.json with the seed's table, every per-sample
value and the means; OUT/eval.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hairdeck.testset import chains, pointcloud  # noqa: E402
from hairdeck.testset.common import SEEDS  # noqa: E402

TESTS = dict(pointcloud=pointcloud, chains=chains)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=list(TESTS))
    ap.add_argument('--ckpt', default=None, help='default: the released checkpoint')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS), help='sampling seeds (default 0 1 2)')
    ap.add_argument('--out', default=None, help='default: results/<pointcloud | chains>')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--cards', default='gt', help="chains: 'gt' or the output folder of test.py pointcloud")
    a = ap.parse_args()
    T = TESTS[a.model]
    kw = dict(cards=a.cards) if a.model == 'chains' else {}
    T.run(a.ckpt or T.CKPT, a.out, seeds=a.seeds, workers=a.workers, **kw)


if __name__ == '__main__':
    main()
