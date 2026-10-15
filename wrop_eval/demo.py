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

# Mock human Elo ratings, chosen so a worse failure mode gets a lower
# rating. In a real study these come from the WROP leaderboard.
MOCK_ELO = {"correct": 1650, "position_drift": 1520,
            "identity_swap": 1400, "disappear": 1280}


def validate_against_mock_elo(reports, out_dir: str):
    """Correlate each behavior's overall OPS with its mock Elo, save the
    result as validation_summary.json, and return the summary dict."""
    ops_per_model = {m: reports[m]["overall"]["OPS"] for m in MODELS}
    corr = validate.correlate(ops_per_model, MOCK_ELO)
    summary = {"ops_per_model": ops_per_model, "mock_elo": MOCK_ELO,
               "correlation": corr}
    with open(os.path.join(out_dir, "validation_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

def plot_category_breakdown(reports, out_path: str):
    """Grouped bar chart: OPS per cognitive category, one bar per behavior."""
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt

    cats = sorted(set().union(*[r["by_category"].keys() for r in reports.values()]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(MODELS)
    x = np.arange(len(cats))
    for i, m in enumerate(MODELS):
        vals = [reports[m]["by_category"].get(c, {}).get("OPS") or 0 for c in cats]
        ax.bar(x + i * width, vals, width=width, label=m)
    ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Object Permanence Score (OPS)")
    ax.set_title("OPS per cognitive category (synthetic models)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ops_vs_elo(summary, out_path: str):
    """Scatter of automatic OPS against (mock) Elo, one point per behavior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [summary["ops_per_model"][m] for m in MODELS]
    ys = [summary["mock_elo"][m] for m in MODELS]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(xs, ys)
    for m, x, y in zip(MODELS, xs, ys):
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Automatic OPS")
    ax.set_ylabel("Human Elo (mock)")
    ax.set_title("OPS vs. Elo")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
def run_demo(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    scores = score_all_models(os.path.join(out_dir, "videos"))
    reports = write_reports(scores, out_dir)
    summary = validate_against_mock_elo(reports, out_dir)
    plot_category_breakdown(reports, os.path.join(out_dir, "category_breakdown.png"))
    plot_ops_vs_elo(summary, os.path.join(out_dir, "ops_vs_elo.png"))

    print("=== Per-model OPS (automatic) vs mock Elo ===")
    for m in MODELS:
        print(f"  {m:16s} OPS={summary['ops_per_model'][m]:.3f}   Elo={MOCK_ELO[m]}")
    corr = summary["correlation"]
    print(f"\nSpearman r = {corr['spearman_r']:.3f}")
    print(f"Pearson  r = {corr['pearson_r']:.3f}")
    print(f"\nOutputs written to: {out_dir}")
    return summary, reports


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="./demo_out")
    args = p.parse_args(argv)
    run_demo(args.out_dir)


if __name__ == "__main__":
    main()