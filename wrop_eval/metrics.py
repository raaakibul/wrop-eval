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