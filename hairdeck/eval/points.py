"""Point metrics with CHARM's evaluation protocol: a run's CHARM-protocol point clouds (RUN/PointClouds/{idx:03d}.ply,
hairdeck.sampling.scan) against reference point clouds of the same test rows (default CHARM's released test point
clouds data/charm_test_pc, pc/ of the dataset repository hyz317/CHARM): CD, EMD, HD and voxel IoU of
hairdeck.eval.metrics.point_metrics (the test row seeds the EMD subsampling), in worker processes.
"""
import os

import numpy as np

from .. import paths
from .metrics import FIELDS, point_metrics, read_ply_points
from .pool import worker_pool


def work(args):
    run, i, ref_dir, fields = args
    a, b = read_ply_points(f'{run}/PointClouds/{i:03d}.ply'), read_ply_points(f'{ref_dir}/{i:03d}.ply')
    return i, point_metrics(a, b, seed=i, fields=fields)


def evaluate_run(run, idxs, ref_dir=None, fields=FIELDS, workers=12):
    """point_metrics (the given fields) of RUN/PointClouds/{idx:03d}.ply against REF_DIR/{idx:03d}.ply for the test rows
    idxs: dict(mean, per_sample (in idxs order), n, failed). A row without a point cloud (an empty generation, which
gets a 0-byte .glb and no scan) is listed in failed, gets {'status': 'no point cloud'} and is left out of the means, as
Table 1 leaves out the outputs that fail to scan; n counts the rows in the means."""
    ref_dir = ref_dir or paths.data('charm_test_pc')
    failed = [i for i in idxs if not os.path.isfile(f'{run}/PointClouds/{i:03d}.ply')]
    with worker_pool(workers) as p:
        res = dict(p.imap_unordered(work, [(run, i, ref_dir, tuple(fields)) for i in idxs if i not in failed]))
    per = {i: res.get(i, {'status': 'no point cloud'}) for i in idxs}
    mean = {k: float(np.mean([res[i][k] for i in res])) if res else float('nan') for k in fields}
    return dict(mean=mean, per_sample=per, n=len(res), failed=failed)
