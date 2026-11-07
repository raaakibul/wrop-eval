import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval import leaderboard as lb


def test_fourteen_models_in_three_classes():
    assert len(lb.ELO) == 14
    classes = [v[0] for v in lb.ELO.values()]
    assert classes.count("continuation") == 4
    assert classes.count("reference") == 3
    assert classes.count("edit") == 7


def test_elo_scale_has_mean_1500():
    # The paper rescales to a mean of 1500; a transcription slip would break this.
    mean = sum(lb.elo_by_name().values()) / 14
    assert mean == pytest.approx(1500.0, abs=0.5)


def test_confidence_intervals_contain_the_estimate():
    for name, (_, elo, lo, hi, games) in lb.ELO.items():
        assert lo <= elo <= hi, name
        assert 50 <= games <= 52, name


def test_ordered_by_elo_descending():
    values = [v[1] for v in lb.ELO.values()]
    assert values == sorted(values, reverse=True)


def test_native_output_covers_the_same_models():
    assert set(lb.NATIVE_OUTPUT) == set(lb.ELO)


def test_name_matching_ignores_punctuation_and_case():
    assert lb.find_model("wan-vace-14b") == "Wan-VACE 14B"
    assert lb.find_model("PWM_WROP") == "PWM-WROP"
    assert lb.find_model("no-such-model") is None