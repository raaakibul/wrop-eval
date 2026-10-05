"""
aggregate.py — rolls per-sample SampleScore objects up into per-task,
per-cognitive-category, and overall-per-model reports.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import List

import numpy as np

from .metrics import SampleScore
from .taxonomy import CATEGORIES


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None

def aggregate_model_report(scores: List[SampleScore]) -> dict:
    by_task = defaultdict(list)
    by_cat = defaultdict(list)
    for s in scores:
        by_task[s.task_id].append(s)
        by_cat[s.category].append(s)

    def summarize(group: List[SampleScore]) -> dict:
        return {
            "n_samples": len(group),
            "OPS": _mean([s.ops for s in group]),
            "disappearance_rate": _mean([s.disappearance_rate for s in group]),
            "identity_swap_rate": _mean([s.identity_swap_rate for s in group]),
            "hallucinated_object_rate": _mean([s.hallucinated_object_rate for s in group]),
            "reappearance_position_error": _mean([s.reappearance_position_error for s in group]),
            "aligned_ade": _mean([s.aligned_ade for s in group]),
            "aligned_fde": _mean([s.aligned_fde for s in group]),
            "frac_kinematically_gated": _mean([1.0 if s.gated else 0.0 for s in group]),
        }

    report = {
        "overall": summarize(scores),
        "by_category": {cat: summarize(by_cat[cat]) for cat in CATEGORIES if by_cat[cat]},
        "by_task": {tid: summarize(g) for tid, g in sorted(by_task.items())},
        "n_total_samples": len(scores),
    }
    return report


def save_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def scores_to_jsonl(scores: List[SampleScore], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s)) + "\n")