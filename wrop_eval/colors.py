"""colors.py: small color-name -> RGB table for matching WROP's ground-truth
`objects[name]["color"]` string field (e.g. "red", "cobalt_blue") against
a tracker's observed mean pixel color, for the identity-swap check.
"""
from __future__ import annotations

from typing import Tuple

_TABLE = {
    "red": (200, 30, 30), "blue": (30, 60, 200), "green": (30, 160, 60),
    "yellow": (230, 210, 30), "orange": (230, 130, 30), "purple": (140, 40, 160),
    "cyan": (40, 190, 200), "magenta": (200, 40, 170), "pink": (230, 130, 180),
    "white": (235, 235, 235), "black": (25, 25, 25), "gray": (130, 130, 130),
    "grey": (130, 130, 130), "brown": (110, 70, 40), "teal": (30, 130, 130),
    "gold": (210, 175, 55), "silver": (190, 190, 195), "lime": (140, 220, 40),
}

def color_name_to_rgb(name: str) -> Tuple[int, int, int]:
    name = (name or "").lower().replace("-", "_")
    for key, rgb in _TABLE.items():
        if key in name:
            return rgb
    return (130, 130, 130)