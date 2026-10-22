import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval.align import SimilarityTransform, fit_and_score, fit_similarity


def test_apply_scale_and_translate():
    t = SimilarityTransform(scale=2.0, tx=1.0, ty=0.5, flip_y=False)
    out = t.apply(np.array([[1.0, 1.0]]))
    assert out == pytest.approx(np.array([[3.0, 2.5]]))


def test_fit_recovers_known_transform():
    world = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    image = np.array([[1.0, 2.0], [3.0, 2.0], [1.0, 4.0]])  # scale 2, tx 1, ty 2
    t = fit_similarity(world, image)
    assert t.scale == pytest.approx(2.0)
    assert t.tx == pytest.approx(1.0)
    assert t.ty == pytest.approx(2.0)
    assert t.flip_y is False


def test_fit_detects_flipped_y():
    world = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    image = np.array([[1.0, 2.0], [3.0, 2.0], [1.0, 0.0]])  # y axis inverted
    t = fit_similarity(world, image)
    assert t.flip_y is True
    assert t.scale == pytest.approx(2.0)


def test_fit_and_score_skips_missing_frames():
    world = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    image = np.array([[1.0, 2.0], [3.0, 2.0], [np.nan, np.nan], [7.0, 2.0]])
    t, resid, valid = fit_and_score(world, image)
    assert t is not None
    assert list(valid) == [True, True, False, True]
    assert np.isnan(resid[2])
    assert resid[0] == pytest.approx(0.0, abs=1e-9)


def test_fit_and_score_needs_three_frames():
    world = np.array([[0.0, 0.0], [1.0, 0.0]])
    image = np.array([[1.0, 2.0], [3.0, 2.0]])
    t, resid, valid = fit_and_score(world, image)
    assert t is None
    assert resid is None
    
    
def _line(n=60):
    x = np.linspace(0.1, 0.9, n)
    return np.stack([x, np.full(n, 0.5)], axis=1)


def test_all_frame_fit_hides_half_of_a_continuation_offset():
    world = _line()
    image = world.copy()
    image[30:, 1] += 0.2
    _, resid, _ = fit_and_score(world, image)
    assert np.nanmean(resid[48:]) == pytest.approx(0.1, abs=0.01)


def test_context_fit_exposes_the_full_continuation_offset():
    world = _line()
    image = world.copy()
    image[30:, 1] += 0.2
    t, resid, used = fit_and_score(world, image, fit_until=30)
    assert used.sum() == 30
    assert np.nanmean(resid[48:]) == pytest.approx(0.2, abs=0.01)
    assert np.nanmax(resid[:30]) < 1e-6   # context is matched exactly


def test_context_fit_refuses_a_stationary_object():
    still = np.stack([np.full(60, 0.3), np.full(60, 0.5)], axis=1)
    t, resid, _ = fit_and_score(still, still, fit_until=30)
    assert t is None and resid is None


def test_context_fit_needs_enough_visible_context_frames():
    world = _line()
    image = world.copy()
    image[:28] = np.nan                    # only 2 context frames visible
    t, resid, _ = fit_and_score(world, image, fit_until=30)
    assert t is None