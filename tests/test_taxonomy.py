import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval.taxonomy import CATEGORIES, build_task_category_map, classify


def test_six_categories():
    assert len(CATEGORIES) == 6
    assert len(set(CATEGORIES)) == 6


def test_containment():
    assert classify("cabinet_task", "an object is placed inside a cabinet") == "containment"


def test_identity_beats_containment():
    # "swap" is checked before "drawer", so identity rules win.
    assert classify("three_drawers_swap", "objects swap between drawers") == "identity_and_tracking"


def test_support_and_gravity():
    assert classify("ball_drop", "a ball is dropped") == "support_and_gravity"


def test_occlusion():
    assert classify("turntable_behind_screen", "a ball behind a screen") == "occlusion_reveal"


def test_unknown_defaults_to_occlusion():
    assert classify("mystery_task", "") == "occlusion_reveal"


def test_every_result_is_a_known_category():
    for name in ["cabinet_task", "ball_drop", "x", "billiard_carom", "size_gate"]:
        assert classify(name) in CATEGORIES


def test_build_task_category_map():
    rows = [("G01", "cabinet_task", "inside a cabinet"), ("G02", "ball_drop", "dropped")]
    assert build_task_category_map(rows) == {
        "G01": "containment",
        "G02": "support_and_gravity",
    }