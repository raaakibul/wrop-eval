import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wrop_eval.colors import color_name_to_rgb

def test_known_color():
    assert color_name_to_rgb("red") == (200, 30, 30)


def test_compound_name_matches_base_color():
    assert color_name_to_rgb("cobalt_blue") == (30, 60, 200)


def test_unknown_falls_back_to_gray():
    assert color_name_to_rgb("mystery_shade") == (130, 130, 130)


def test_empty_and_none_do_not_crash():
    assert color_name_to_rgb("") == (130, 130, 130)
    assert color_name_to_rgb(None) == (130, 130, 130)