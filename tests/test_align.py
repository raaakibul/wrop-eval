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