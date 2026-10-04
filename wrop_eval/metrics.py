"""
metrics.py — the core methodological contribution: an automatic,
trajectory-grounded, per-cognitive-category Object Permanence Score (OPS)
for a *generated* video continuation, computed against WROP's own
ground-truth per-frame trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import align, colors
from .schema import GroundTruthTrajectory, ObjectTrack


def _fit_const_accel_residual(xyz: np.ndarray) -> float:
    valid = ~np.isnan(xyz).any(axis=1)
    if valid.sum() < 4:
        return np.inf
    t = np.arange(len(xyz))[valid].astype(np.float64)
    t = t - t.mean()
    X = np.stack([np.ones_like(t), t, t ** 2], axis=1)
    resid_total = 0.0
    for d in range(3):
        y = xyz[valid, d]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        resid_total += np.mean((pred - y) ** 2)
    return float(np.sqrt(resid_total / 3))


def is_kinematically_gated(gt_input_window: np.ndarray, tolerance: float) -> bool:
    return _fit_const_accel_residual(gt_input_window) <= tolerance


@dataclass
class MatchedPair:
    gt_name: str
    pred_id: Optional[str]
    transform_residual: float
    valid_frames: np.ndarray


def match_predicted_to_gt(gt_targets: List[ObjectTrack],
                           pred_tracks: Dict[str, np.ndarray]) -> List[MatchedPair]:
    matches = []
    for gt in gt_targets:
        world_xy = gt.loc[:, [0, 2]]
        best = None
        for pid, arr in pred_tracks.items():
            image_xy = arr[:, :2]
            t, resid, valid = align.fit_and_score(world_xy, image_xy)
            if t is None:
                continue
            mean_resid = float(np.nanmean(resid))
            if best is None or mean_resid < best[0]:
                best = (mean_resid, pid, resid, valid)
        if best is None:
            matches.append(MatchedPair(gt.name, None, np.inf,
                                        np.zeros(len(gt.loc), dtype=bool)))
        else:
            matches.append(MatchedPair(gt.name, best[1], best[0], best[3]))
    return matches

@dataclass
class SampleScore:
    sample_id: str
    task_id: str
    category: str
    disappearance_rate: float
    identity_swap_rate: float
    hallucinated_object_rate: float
    reappearance_position_error: Optional[float]  # None if not gated
    aligned_ade: Optional[float]
    aligned_fde: Optional[float]
    gated: bool
    n_gt_objects: int
    n_matched: int
    ops: float = 0.0

HARD_PENALTY_WEIGHT = 1.0
SOFT_PENALTY_WEIGHT = 0.5
SOFT_ERROR_SCALE = 0.15


def _squash(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(1.0 - np.exp(-x / SOFT_ERROR_SCALE))