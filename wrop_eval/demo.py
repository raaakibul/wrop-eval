"""
demo.py: end-to-end synthetic demonstration of the OPS pipeline 

It renders tiny synthetic videos for four caricatured model behaviors:
  correct         the ball reappears exactly where it should
  position_drift  the ball reappears, but off the true path
  identity_swap   the ball reappears with the wrong color
  disappear       the ball never reappears
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import aggregate, validate
from .constants import FPS, N_INPUT_FRAMES, N_TOTAL_FRAMES
from .metrics import score_sample
from .schema import GroundTruthTrajectory, ObjectTrack
from .tracker import BaselineBlobTracker, make_synthetic_video

RED_BGR = (30, 30, 200)     # ground-truth object color ("red"), in OpenCV BGR order
BLUE_BGR = (200, 60, 30)    # the wrong color used by the identity-swap model
OCCL_FRAMES = (60, 100)     # frames during which the ball can be hidden

# One synthetic task per cognitive category, so the per-category breakdown
# is exercised too. All three paths cross the occluder band.
TASK_SPECS = [
    ("synthetic_occlusion_pass", "occlusion_reveal",
     {"xy0": (0.08, 0.5), "xy1": (0.92, 0.5)}),
    ("synthetic_box_reveal", "containment",
     {"xy0": (0.08, 0.62), "xy1": (0.92, 0.40)}),
    ("synthetic_support_slide", "support_and_gravity",
     {"xy0": (0.08, 0.30), "xy1": (0.92, 0.70)}),
]

def _make_gt(event) -> GroundTruthTrajectory:
    """Build a ground-truth trajectory (straight-line motion) for one event."""
    t = np.linspace(0, 1, N_TOTAL_FRAMES)
    x = event["xy0"][0] + t * (event["xy1"][0] - event["xy0"][0])
    y = event["xy0"][1] + t * (event["xy1"][1] - event["xy0"][1])
    loc = np.stack([x, np.zeros_like(x), y], axis=1)  # align.py reads columns [0, 2]
    track = ObjectTrack(name="ball", shape="sphere", role="target",
                        color="red", is_target=True, loc=loc,
                        rot=np.zeros_like(loc))
    return GroundTruthTrajectory(fps=FPS, frame_start=1, frame_end=N_TOTAL_FRAMES,
                                 objects={"ball": track})


def _events_for_model(event, behavior: str):
    """Return the synthetic-video event list for one caricatured behavior."""
    ev = dict(event, color=RED_BGR, radius=5, occlude_frames=OCCL_FRAMES)
    if behavior == "correct":
        return [ev]
    if behavior == "disappear":
        return [dict(ev, vanish_after=OCCL_FRAMES[0] + 2)]
    if behavior == "identity_swap":
        return [dict(ev, identity_swap_color_after=BLUE_BGR,
                     identity_swap_frame=OCCL_FRAMES[0] + 10)]
    if behavior == "position_drift":
        drift_xy1 = (event["xy1"][0], min(1.0, event["xy1"][1] + 0.22))
        return [dict(ev, xy1=drift_xy1)]
    raise ValueError(f"unknown behavior: {behavior}")

MODELS = ["correct", "position_drift", "identity_swap", "disappear"]


def score_all_models(video_dir: str):
    """Render a synthetic video for every (task, behavior) pair, track it,
    score it, and return {behavior: [SampleScore, ...]}."""
    os.makedirs(video_dir, exist_ok=True)
    tracker = BaselineBlobTracker()
    per_model_scores = {m: [] for m in MODELS}

    for task_name, category, event in TASK_SPECS:
        gt = _make_gt(event)
        for behavior in MODELS:
            events = _events_for_model(event, behavior)
            vpath = os.path.join(video_dir, f"{task_name}__{behavior}.mp4")
            make_synthetic_video(vpath, events, n_frames=N_TOTAL_FRAMES, fps=FPS)
            pred_tracks, pred_colors = tracker.track_with_color(vpath)
            per_model_scores[behavior].append(score_sample(
                sample_id=f"{task_name}_0000", task_id=task_name, category=category,
                gt=gt, pred_tracks=pred_tracks, n_input_frames=N_INPUT_FRAMES,
                pred_colors=pred_colors,
            ))
    return per_model_scores

def write_reports(per_model_scores, out_dir: str):
    """Aggregate and save one JSON report, one JSONL score file, and one
    markdown table per behavior. Returns {behavior: report_dict}."""
    os.makedirs(out_dir, exist_ok=True)
    reports = {}
    for model, scores in per_model_scores.items():
        report = aggregate.aggregate_model_report(scores)
        reports[model] = report
        aggregate.save_report(report, os.path.join(out_dir, f"report_{model}.json"))
        aggregate.scores_to_jsonl(scores, os.path.join(out_dir, f"scores_{model}.jsonl"))
        # utf-8 matters on Windows: the table contains the "—" character.
        with open(os.path.join(out_dir, f"report_{model}.md"), "w", encoding="utf-8") as f:
            f.write(aggregate.markdown_table(report, model_name=model))
    return reports

