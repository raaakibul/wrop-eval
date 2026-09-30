"""
align.py — calibration-free alignment between a ground-truth 3D world
trajectory and a predicted 2D image-plane trajectory.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SimilarityTransform:
    scale: float
    tx: float
    ty: float
    flip_y: bool

    def apply(self, world_xy: np.ndarray) -> np.ndarray:
        x = world_xy[..., 0] * self.scale + self.tx
        y = world_xy[..., 1] * (-self.scale if self.flip_y else self.scale) + self.ty
        return np.stack([x, y], axis=-1)
    
    
def fit_similarity(world_xy: np.ndarray, image_xy: np.ndarray) -> SimilarityTransform:
    best = None
    for flip_y in (False, True):
        wx, wy = world_xy[:, 0], world_xy[:, 1] * (-1.0 if flip_y else 1.0)
        ix, iy = image_xy[:, 0], image_xy[:, 1]
        n = len(wx)
        A = np.zeros((2 * n, 3))
        b = np.zeros(2 * n)
        A[0:n, 0] = wx
        A[0:n, 1] = 1.0
        b[0:n] = ix
        A[n:2 * n, 0] = wy
        A[n:2 * n, 2] = 1.0
        b[n:2 * n] = iy
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        scale, tx, ty = sol
        t = SimilarityTransform(scale=float(scale), tx=float(tx), ty=float(ty), flip_y=flip_y)
        pred = t.apply(world_xy)
        resid = float(np.mean(np.sum((pred - image_xy) ** 2, axis=-1)))
        if best is None or resid < best[0]:
            best = (resid, t)
    return best[1]

def fit_and_score(world_xy_full: np.ndarray, image_xy_full: np.ndarray):
    valid = (~np.isnan(world_xy_full).any(axis=-1)) & (~np.isnan(image_xy_full).any(axis=-1))
    if valid.sum() < 3:
        return None, None, valid
    t = fit_similarity(world_xy_full[valid], image_xy_full[valid])
    pred = t.apply(world_xy_full)
    resid = np.linalg.norm(pred - image_xy_full, axis=-1)  # NaN where either is NaN
    return t, resid, valid

