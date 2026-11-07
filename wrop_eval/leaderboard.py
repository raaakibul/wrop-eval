"""leaderboard.py: the human-preference results published with WROP, so the
OPS-vs-Elo validation can use real numbers instead of invented ones.

Source: Zhang et al., "Training Object Permanence in World Models",
arXiv:2609.28654, Table 2 (overall Bradley-Terry Elo on the 300-question
exam; 20 raters, 361 pairwise judgments; Elo scale with mean 1500) and
Table 1 (native output geometry of each model).
"""
from __future__ import annotations

import re
from typing import Dict

# name -> (interface class, elo, ci_low, ci_high, games)
# classes: "continuation", "reference", "edit"
ELO = {
    "Wan 3.0 Prime":               ("reference",    1723.6, 1629.2, 1864.9, 52),
    "MiniMax H3":                  ("reference",    1723.6, 1644.5, 1837.6, 52),
    "PWM-WROP":                    ("continuation", 1679.5, 1603.5, 1781.5, 52),
    "Seedance 2.5":                ("reference",    1649.6, 1554.2, 1751.5, 51),
    "Runway Aleph 2":              ("edit",         1518.3, 1425.1, 1621.6, 51),
    "Wan-VACE 14B":                ("edit",         1506.7, 1429.9, 1590.8, 52),
    "Gemini Omni Flash 1.1":       ("edit",         1492.5, 1404.1, 1572.9, 52),
    "Kling O3 Pro":                ("edit",         1471.3, 1404.0, 1534.0, 52),
    "Grok Imagine (video extend)": ("continuation", 1457.0, 1362.7, 1555.1, 52),
    "LTX-2.3 Extend":              ("continuation", 1453.4, 1363.6, 1545.8, 51),
    "Cosmos3 Super":               ("edit",         1409.2, 1318.2, 1488.7, 51),
    "LTX-2.3 Dev":                 ("edit",         1398.9, 1297.7, 1482.3, 52),
    "HY-OmniWeaving":              ("edit",         1268.5, 1159.4, 1344.2, 50),
    "MAGI-1 24B":                  ("continuation", 1248.0, 1137.2, 1315.6, 52),
}

# name -> (width, height, fps, frames as emitted, note)   [paper Table 1]
NATIVE_OUTPUT = {
    "PWM-WROP":                    (320, 192, 24, 117, "replays 57 conditioning frames; 60 predicted"),
    "MAGI-1 24B":                  (1280, 720, 24, 77, ""),
    "LTX-2.3 Dev":                 (1536, 1024, 24, 121, ""),
    "Wan-VACE 14B":                (832, 480, 16, 57, ""),
    "HY-OmniWeaving":              (848, 480, 24, 57, ""),
    "Cosmos3 Super":               (1280, 720, 24, 60, ""),
    "LTX-2.3 Extend":              (1920, 1080, 24, 145, "stitched with source prefix; trimmed"),
    "Grok Imagine (video extend)": (1280, 720, 24, 132, "stitched with source prefix; trimmed"),
    "Seedance 2.5":                (1280, 720, 24, 97, ""),
    "Wan 3.0 Prime":               (1280, 720, 30, 90, ""),
    "MiniMax H3":                  (1344, 768, 24, 124, ""),
    "Kling O3 Pro":                (1920, 1080, 24, 73, "stitched or padded; trimmed"),
    "Gemini Omni Flash 1.1":       (1280, 720, 24, 60, ""),
    "Runway Aleph 2":              (1920, 1080, 24, 60, ""),
}


def normalize_name(name: str) -> str:
    """Lowercase and drop everything except letters and digits, so that
    'Wan-VACE 14B' and 'wan_vace_14b' compare equal."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def elo_by_name() -> Dict[str, float]:
    return {name: v[1] for name, v in ELO.items()}


def find_model(directory_name: str):
    """Match a results/<directory> name to a paper model name, or None.
    Matches when the normalized names are equal. Directory names in the
    benchmark repo may differ from the paper's names; add aliases here
    once you have seen them."""
    target = normalize_name(directory_name)
    for name in ELO:
        if normalize_name(name) == target:
            return name
    return None