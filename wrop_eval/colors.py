"""colors.py: map the color names in WROP's ground truth to RGB, for the
identity-swap check.

The 16 object colors below are copied from PALETTE in the generator
(object_permanence/generator/core/render.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

PALETTE = {
    "orange": (0.97, 0.45, 0.10), "red": (0.90, 0.15, 0.15),
    "pink": (0.96, 0.58, 0.74), "magenta": (0.85, 0.18, 0.62),
    "purple": (0.55, 0.25, 0.80), "navy": (0.10, 0.18, 0.50),
    "blue": (0.18, 0.40, 0.95), "cyan": (0.10, 0.72, 0.82),
    "teal": (0.08, 0.52, 0.52), "green": (0.18, 0.68, 0.30),
    "lime": (0.62, 0.85, 0.20), "yellow": (0.98, 0.80, 0.15),
    "gold": (0.83, 0.62, 0.12), "brown": (0.42, 0.26, 0.14),
    "gray": (0.45, 0.46, 0.48), "black": (0.05, 0.05, 0.06),
}


def color_name_to_rgb(name: str) -> Optional[Tuple[int, int, int]]:
    """RGB (0-255) for a color name, or None if the name is unknown.

    Returning None (instead of a made-up default) lets the caller skip the
    identity check, rather than flag a false swap against a guessed color.
    """
    key = (name or "").strip().lower().replace("-", "_")
    if not key:
        return None
    if key in PALETTE:
        match = PALETTE[key]
    else:
        # compound names such as "dark_navy": use the first palette word found
        found = [w for w in PALETTE if w in key.split("_")]
        if not found:
            return None
        match = PALETTE[found[0]]
    return tuple(int(round(255 * v)) for v in match)