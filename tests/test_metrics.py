import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval.metrics import (
    _squash,
    is_kinematically_gated,
    match_predicted_to_gt,
    score_sample,
)
from wrop_eval.schema import GroundTruthTrajectory, ObjectTrack

N = 60
N_INPUT = 30
RED_BGR = (30, 30, 200)
BLUE_BGR = (200, 60, 30)


def _gt(color="red"):
    """Ball moving in a straight line at constant speed."""
    x = np.linspace(0.1, 0.9, N)
    loc = np.stack([x, np.zeros(N), np.full(N, 0.5)], axis=1)
    ball = ObjectTrack(name="ball", color=color, is_target=True, loc=loc)
    return GroundTruthTrajectory(fps=24, frame_start=1, frame_end=N,
                                 objects={"ball": ball})


def _pred(hidden=None):
    """A predicted track that follows the ground truth exactly.
    `hidden` = (start, end) frames to blank out."""
    x = np.linspace(0.1, 0.9, N)
    arr = np.stack([x, np.full(N, 0.5), np.ones(N)], axis=1)
    if hidden:
        arr[hidden[0]:hidden[1]] = np.nan
    return arr


def _colors(bgr):
    return np.tile(np.array(bgr, dtype=float), (N, 1))


# --- kinematic gating ------------------------------------------------------

def test_smooth_motion_is_gated():
    smooth = np.stack([np.linspace(0, 1, 30), np.zeros(30), np.zeros(30)], axis=1)
    assert is_kinematically_gated(smooth, tolerance=0.01)


def test_motion_with_a_kink_is_not_gated():
    kinked = np.stack([np.linspace(0, 1, 30), np.zeros(30), np.zeros(30)], axis=1)
    kinked[15:, 0] = kinked[15, 0] - np.linspace(0, 0.5, 15)  # reverses direction
    assert not is_kinematically_gated(kinked, tolerance=0.01)


def test_too_few_points_is_not_gated():
    tiny = np.zeros((3, 3))
    assert not is_kinematically_gated(tiny, tolerance=1.0)


# --- squash helper ---------------------------------------------------------

def test_squash_values():
    assert _squash(None) == 0.0
    assert _squash(0.0) == 0.0
    assert _squash(0.15) == pytest.approx(1 - np.exp(-1))
    assert _squash(5.0) > 0.99


# --- track matching --------------------------------------------------------

def test_matching_prefers_the_good_track():
    rng = np.random.default_rng(0)
    noise = rng.random((N, 3))
    matches = match_predicted_to_gt(list(_gt().objects.values()),
                                    {"good": _pred(), "bad": noise})
    assert matches[0].pred_id == "good"


def test_matching_with_no_tracks_gives_none():
    matches = match_predicted_to_gt(list(_gt().objects.values()), {})
    assert matches[0].pred_id is None


# --- score_sample ----------------------------------------------------------

def test_perfect_prediction_scores_high():
    s = score_sample("s", "t", "c", _gt(), {"p": _pred()}, N_INPUT,
                     pred_colors={"p": _colors(RED_BGR)})
    assert s.disappearance_rate == 0.0
    assert s.identity_swap_rate == 0.0
    assert s.hallucinated_object_rate == 0.0
    assert s.gated is True
    assert s.ops > 0.97


def test_object_that_never_returns_is_a_disappearance():
    s = score_sample("s", "t", "c", _gt(), {"p": _pred(hidden=(40, N))}, N_INPUT)
    assert s.disappearance_rate == 1.0
    assert s.ops < 0.85


def test_no_predicted_tracks_counts_as_disappearance():
    s = score_sample("s", "t", "c", _gt(), {}, N_INPUT)
    assert s.disappearance_rate == 1.0
    assert s.n_matched == 0


def test_wrong_color_is_an_identity_swap():
    s = score_sample("s", "t", "c", _gt(), {"p": _pred()}, N_INPUT,
                     pred_colors={"p": _colors(BLUE_BGR)})
    assert s.identity_swap_rate == 1.0
    assert s.disappearance_rate == 0.0


def test_slightly_washed_out_color_is_not_a_swap():
    # Mean color measured from a real rendered ball was ~(60, 59, 193) BGR.
    s = score_sample("s", "t", "c", _gt(), {"p": _pred()}, N_INPUT,
                     pred_colors={"p": _colors((60, 59, 193))})
    assert s.identity_swap_rate == 0.0


def test_extra_unmatched_track_is_hallucination():
    rng = np.random.default_rng(1)
    extra = rng.random((N, 3))
    s = score_sample("s", "t", "c", _gt(), {"p": _pred(), "extra": extra}, N_INPUT)
    assert s.hallucinated_object_rate == pytest.approx(0.5)


def test_position_error_lowers_score_but_is_not_a_hard_failure():
    good = score_sample("s", "t", "c", _gt(), {"p": _pred()}, N_INPUT)
    drifted = _pred()
    drifted[:, 1] += np.linspace(0, 0.3, N)  # slides off the true path
    bad = score_sample("s", "t", "c", _gt(), {"p": drifted}, N_INPUT)
    assert bad.disappearance_rate == 0.0
    assert bad.identity_swap_rate == 0.0
    assert bad.ops < good.ops

# --- context-only alignment ------------------------------------------------

def test_offset_after_the_context_is_fully_penalized():
    # The model matches the context exactly, then shifts by a constant 0.2.
    shifted = _pred()
    shifted[N_INPUT:, 1] += 0.2
    s = score_sample("s", "t", "c", _gt(), {"p": shifted}, N_INPUT)
    assert s.gated is True
    assert s.reappearance_position_error == pytest.approx(0.2, abs=0.02)
    assert s.aligned_ade == pytest.approx(0.2, abs=0.02)
    assert s.ops < 0.9


def test_stationary_object_is_not_marked_gated():
    # No motion in the context -> the alignment scale is undetermined, so we
    # must not claim the soft metrics were computed.
    still = np.stack([np.full(N, 0.3), np.zeros(N), np.full(N, 0.5)], axis=1)
    ball = ObjectTrack(name="ball", color="red", is_target=True, loc=still)
    gt = GroundTruthTrajectory(fps=24, frame_start=1, frame_end=N, objects={"ball": ball})
    pred = np.stack([np.full(N, 0.3), np.full(N, 0.5), np.ones(N)], axis=1)
    s = score_sample("s", "t", "c", gt, {"p": pred}, N_INPUT)
    assert s.gated is False
    assert s.reappearance_position_error is None
    assert s.disappearance_rate == 0.0