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

def score_sample(sample_id: str, task_id: str, category: str,
                  gt: GroundTruthTrajectory,
                  pred_tracks: Dict[str, np.ndarray],
                  n_input_frames: int,
                  pred_colors: Optional[Dict[str, np.ndarray]] = None,
                  expected_visible_frac: float = 0.2,
                  kinematic_tolerance: float = 0.02,
                  identity_color_thresh: float = 70.0) -> SampleScore:
    targets = gt.target_objects() or list(gt.objects.values())[:1]
    n = gt.n_frames
    tail_start = max(n_input_frames, int(round(n * (1 - expected_visible_frac))))

    matches = match_predicted_to_gt(targets, pred_tracks)

    disappear_flags, swap_flags = [], []
    rpe_vals, ade_vals, fde_vals = [], [], []
    any_gated = False

    for gtobj, m in zip(targets, matches):
        tail_gt_valid = ~np.isnan(gtobj.loc[tail_start:n]).any(axis=1)
        should_be_visible = tail_gt_valid.any()

        if m.pred_id is None:
            # No predicted track matched this GT object at all.
            disappear_flags.append(1.0 if should_be_visible else 0.0)
            swap_flags.append(0.0)
        else:
            pred_arr = pred_tracks[m.pred_id]
            pred_tail_valid = ~np.isnan(pred_arr[tail_start:n, :2]).any(axis=1)
            reappeared = pred_tail_valid.any()
            disappear_flags.append(1.0 if (should_be_visible and not reappeared) else 0.0)

            # Identity check via color signature.
            swap = 0.0
            if reappeared and pred_colors and m.pred_id in pred_colors and gtobj.color:
                tail_colors = pred_colors[m.pred_id][tail_start:n]
                tail_colors = tail_colors[~np.isnan(tail_colors).any(axis=1)]
                if len(tail_colors):
                    observed_bgr = tail_colors.mean(axis=0)
                    observed_rgb = observed_bgr[::-1]  # OpenCV is BGR
                    expected_rgb = np.array(colors.color_name_to_rgb(gtobj.color), dtype=np.float64)
                    dist = float(np.linalg.norm(observed_rgb - expected_rgb))
                    swap = 1.0 if dist > identity_color_thresh else 0.0
            swap_flags.append(swap)

            # Kinematic gating uses only the INPUT window.
            if should_be_visible and reappeared:
                gt_input = gtobj.loc[:n_input_frames]
                gated = is_kinematically_gated(
                    gt_input,
                    tolerance=kinematic_tolerance * max(1.0, np.nanstd(gt_input)),
                )
                if gated:
                    any_gated = True
                    world_xy = gtobj.loc[:, [0, 2]]
                    t, resid, valid = align.fit_and_score(world_xy, pred_arr[:, :2])
                    if t is not None:
                        tail_idx = np.arange(tail_start, n)
                        tail_resid = resid[tail_idx]
                        tail_resid = tail_resid[~np.isnan(tail_resid)]
                        if len(tail_resid):
                            first_seen = np.argmax(pred_tail_valid) + tail_start
                            if first_seen < n and not np.isnan(resid[first_seen]):
                                rpe_vals.append(resid[first_seen])
                            ade_vals.append(float(np.mean(tail_resid)))
                            fde_vals.append(float(tail_resid[-1]))

    n_pred_only = sum(1 for pid in pred_tracks
                       if pid not in {m.pred_id for m in matches})
    hallucinated_rate = n_pred_only / max(1, len(pred_tracks)) if pred_tracks else 0.0

    disappearance_rate = float(np.mean(disappear_flags)) if disappear_flags else 0.0
    identity_swap_rate = float(np.mean(swap_flags)) if swap_flags else 0.0

    rpe = float(np.mean(rpe_vals)) if rpe_vals else None
    ade = float(np.mean(ade_vals)) if ade_vals else None
    fde = float(np.mean(fde_vals)) if fde_vals else None

    hard_penalty = (disappearance_rate + identity_swap_rate + hallucinated_rate) / 3.0
    soft_penalty = np.mean([_squash(rpe), _squash(ade), _squash(fde)]) if any_gated else 0.0
    ops = 1.0 - (HARD_PENALTY_WEIGHT * hard_penalty + SOFT_PENALTY_WEIGHT * soft_penalty) \
        / (HARD_PENALTY_WEIGHT + SOFT_PENALTY_WEIGHT)
    ops = float(np.clip(ops, 0.0, 1.0))

    return SampleScore(
        sample_id=sample_id, task_id=task_id, category=category,
        disappearance_rate=disappearance_rate,
        identity_swap_rate=identity_swap_rate,
        hallucinated_object_rate=hallucinated_rate,
        reappearance_position_error=rpe, aligned_ade=ade, aligned_fde=fde,
        gated=any_gated, n_gt_objects=len(targets),
        n_matched=sum(1 for m in matches if m.pred_id is not None),
        ops=ops,
    )