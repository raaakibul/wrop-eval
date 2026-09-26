from __future__ import annotations

from typing import Dict, List

CATEGORIES: List[str] = [
    "occlusion_reveal",
    "containment",
    "support_and_gravity",
    "solidity_and_collision",
    "identity_and_tracking",
    "aperture_and_transfer",
]

# Ordered rule list: first matching keyword set wins. Keywords are matched
# against f"{task_name} {description}".lower().
_RULES = [
    ("identity_and_tracking", [
        "swap", "identity", "three_drawers", "three_lanes", "two_boxes",
        "two_carts", "cart_swap", "parallel_cars", "crossing", "lanes",
    ]),
    ("aperture_and_transfer", [
        "size_gate", "gate", "funnel", "sorter", "one_way", "flap_gate",
        "hole_box", "u_track", "u_tube",
    ]),
    ("containment", [
        "box", "drawer", "cabinet", "capsule", "partition", "container",
        "nested_cup", "chest", "lidded",
    ]),
    ("support_and_gravity", [
        "support_removed", "drop", "fall", "topple", "tilt", "release",
        "hanging", "pillar", "rollers_part", "shelf", "roll_off", "rolloff",
    ]),
    ("solidity_and_collision", [
        "collision", "blocked_by_wall", "no_penetration", "barrier",
        "billiard", "bank_shot", "carom", "pool_rack", "rebound", "strikes",
        "shoves", "knock", "pendulum", "wall", "bridge",
    ]),
    ("occlusion_reveal", [
        "screen", "panel", "tunnel", "turntable", "cover", "curtain",
        "blind", "occlu", "window", "slit", "mask", "post", "column",
        "fence", "flicker",
    ]),
]

def classify(task_name: str, description: str = "") -> str:
    text = f"{task_name} {description}".lower()
    for category, keywords in _RULES:
        if any(k in text for k in keywords):
            return category
    return "occlusion_reveal"

def build_task_category_map(task_rows) -> Dict[str, str]:
    return {tid: classify(name, desc) for tid, name, desc in task_rows}

