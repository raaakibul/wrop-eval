import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval.colors import PALETTE, color_name_to_rgb


def test_palette_has_the_16_generator_colors():
    assert len(PALETTE) == 16
    for name in ["orange", "navy", "yellow", "gray", "black"]:
        assert name in PALETTE


def test_known_colors_convert_to_0_255():
    assert color_name_to_rgb("red") == (230, 38, 38)
    assert color_name_to_rgb("navy") == (26, 46, 128)


def test_name_is_case_and_hyphen_insensitive():
    assert color_name_to_rgb("  Navy ") == (26, 46, 128)


def test_compound_name_uses_the_palette_word():
    assert color_name_to_rgb("dark_navy") == (26, 46, 128)


def test_unknown_color_returns_none_not_a_guess():
    assert color_name_to_rgb("mystery_shade") is None


def test_empty_and_none_return_none():
    assert color_name_to_rgb("") is None
    assert color_name_to_rgb(None) is None