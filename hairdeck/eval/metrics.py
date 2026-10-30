"""Point-cloud metrics between a generated and a reference point cloud, with the conventions of the CHARM evaluation
protocol (Chamfer distance and Hausdorff distance on the full clouds, EMD on 1,000 random points of each, voxel IoU at
1/32), plus surface sampling helpers. Only numpy, scipy and trimesh.

CHARM's evaluation computes CD, HD and EMD with point-cloud-utils (https://github.com/fwilliams/point-cloud-utils, 0.34.0);
the functions here give its values: chamfer_distance and hausdorff_distance find the nearest neighbours with a kd-tree
(scipy) and then apply the numpy operations that point_cloud_utils.chamfer_distance applies after its kNN query (the
Hausdorff distance: the largest nearest-neighbour distance in either direction); earth_movers_distance, sinkhorn and
pairwise_distances are point_cloud_utils/_sinkhorn.py (its log-domain Sinkhorn in numpy), shortened to the arguments used
here. point-cloud-utils: MIT License, Copyright (c) 2018 Francis Williams.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright
  notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .. import ply


def uniform_sample(mesh, n, seed=0):
    """n area-uniform surface samples of a mesh or scene (trimesh.sample.sample_surface with a seed)."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts)


def voxel_iou(a, b, voxel_size=1 / 32.):
    va = set(map(tuple, np.floor(a / voxel_size).astype(int)))
    vb = set(map(tuple, np.floor(b / voxel_size).astype(int)))
    u = len(va | vb)
    return len(va & vb) / u if u else 0.0


def chamfer_distance(x, y):
    """point_cloud_utils.chamfer_distance(x, y): the mean distance from every point of y to its nearest point of x plus
    the mean distance from every point of x to its nearest point of y."""
    corrs_x_to_y = cKDTree(y).query(x)[1]
    corrs_y_to_x = cKDTree(x).query(y)[1]
    dists_x_to_y = np.linalg.norm(x[corrs_y_to_x] - y, axis=-1, ord=2).mean()
    dists_y_to_x = np.linalg.norm(y[corrs_x_to_y] - x, axis=-1, ord=2).mean()
    return np.mean(dists_x_to_y) + np.mean(dists_y_to_x)


def hausdorff_distance(x, y):
    """point_cloud_utils.hausdorff_distance(x, y): the largest nearest-neighbour distance, x to y or y to x."""
    return max(cKDTree(y).query(x)[0].max(), cKDTree(x).query(y)[0].max())


def pairwise_distances(a, b, p=2):
    """(n, m) distances between the points of a (n, d) and b (m, d) (point_cloud_utils.pairwise_distances)."""
    return np.squeeze(np.linalg.norm(a[np.newaxis, :, np.newaxis, :] - b[np.newaxis, np.newaxis, :, :], axis=-1, ord=p))


def sinkhorn(a, b, M, eps, max_iters=100, stop_thresh=1e-3):
    """point_cloud_utils.sinkhorn for one pair of distributions: weights a (n,), b (m,), costs M (n, m) -> the
    correspondences P (n, m) of the log-domain Sinkhorn iteration."""
    M, a, b = np.squeeze(M)[np.newaxis], np.squeeze(a)[np.newaxis], np.squeeze(b)[np.newaxis]
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    M_t = np.transpose(M, axes=(0, 2, 1))

    def stabilized_log_sum_exp(x):
        max_x = x.max(2)
        x = x - max_x[:, :, np.newaxis]
        return np.log(np.sum(np.exp(x), axis=2)) + max_x

    for _ in range(max_iters):
        u_prev, v_prev = u, v
        summand_u = (-M + np.expand_dims(v, 1)) / eps
        u = eps * (np.log(a) - stabilized_log_sum_exp(summand_u))
        summand_v = (-M_t + np.expand_dims(u, 1)) / eps
        v = eps * (np.log(b) - stabilized_log_sum_exp(summand_v))
        err_u = np.sum(np.abs(u_prev - u), axis=1).max()
        err_v = np.sum(np.abs(v_prev - v), axis=1).max()
        if err_u < stop_thresh and err_v < stop_thresh:
            break
    log_P = (-M + np.expand_dims(u, 2) + np.expand_dims(v, 1)) / eps
    return np.squeeze(np.exp(log_P))


def earth_movers_distance(p, q, eps=1e-4, max_iters=100, stop_thresh=1e-3):
    """point_cloud_utils.earth_movers_distance(p, q)[0]: sum(P * M) of the Sinkhorn correspondences P between the uniform
    distributions on the points of p and q, M their distances."""
    M = pairwise_distances(p, q, 2)
    a = np.ones(p.shape[0]) / p.shape[0]
    b = np.ones(q.shape[0]) / q.shape[0]
    P = sinkhorn(a, b, M, eps, max_iters, stop_thresh)
    return (P * M).sum()


FIELDS = ('cd', 'emd', 'hd', 'iou')


def point_metrics(pred, gt, seed=0, fields=FIELDS):
    """dict of the fields among cd, emd, hd, iou (in that order). Set HAIRDECK_NO_EMD=1 to skip the EMD (recorded as
    NaN)."""
    out = {}
    if 'cd' in fields:
        out['cd'] = float(chamfer_distance(pred, gt))
    if 'emd' in fields:
        rng = np.random.default_rng(seed)
        a = pred[rng.choice(pred.shape[0], 1000, replace=False)]
        b = gt[rng.choice(gt.shape[0], 1000, replace=False)]
        if os.environ.get('HAIRDECK_NO_EMD') or os.environ.get('CHARM_EVAL_NO_EMD'):
            out['emd'] = float('nan')
        else:
            out['emd'] = float(earth_movers_distance(a, b))
    if 'hd' in fields:
        out['hd'] = float(hausdorff_distance(pred, gt))
    if 'iou' in fields:
        out['iou'] = float(voxel_iou(pred, gt))
    return out


def fscore(a, b, tau):
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    p, r = np.mean(da < tau), np.mean(db < tau)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def read_ply_points(path):
    """xyz of a .ply point cloud as float64 (hairdeck.ply; the values open3d's reader gives)."""
    return ply.read_points(path)
