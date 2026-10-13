"""
cli.py: run the OPS pipeline over a directory layout matching the WROP
Usage:
    python -m wrop_eval.cli --eval-root eval/ --results-root results/ --model pwm-wrop
"""
from __future__ import annotations

import argparse
import os

from . import aggregate, taxonomy
from .constants import N_INPUT_FRAMES
from .metrics import score_sample
from .schema import load_ground_truth, load_metadata
from .tracker import BaselineBlobTracker


def find_samples(eval_root: str):
    """Yield (task_name, sample_id, sample_dir) for every sample under
    eval_root, regardless of how deeply shards are nested."""
    for dirpath, dirnames, filenames in os.walk(eval_root):
        has_gt = ("scene_state_graph.json" in filenames) or ("trajectory.npz" in filenames)
        if has_gt:
            sample_id = os.path.basename(dirpath)
            task_dir = os.path.basename(os.path.dirname(dirpath))
            task_name = task_dir.replace("_task", "")
            yield task_name, sample_id, dirpath
            
            
def run(eval_root: str, results_root: str, model_name: str,
        tracker=None, limit: int = None):
    tracker = tracker or BaselineBlobTracker()
    scores = []
    for task_name, sample_id, sample_dir in find_samples(eval_root):
        pred_video = os.path.join(results_root, model_name, f"{task_name}_task",
                                  f"{sample_id}.mp4")
        if os.path.isfile(pred_video):
            gt = load_ground_truth(sample_dir)
            meta = load_metadata(sample_dir)
            if hasattr(tracker, "track_with_color"):
                pred_tracks, pred_colors = tracker.track_with_color(pred_video)
            else:
                pred_tracks, pred_colors = tracker.track(pred_video), None
            category = taxonomy.classify(task_name, meta.get("description", ""))
            scores.append(score_sample(
                sample_id=sample_id, task_id=task_name, category=category,
                gt=gt, pred_tracks=pred_tracks, n_input_frames=N_INPUT_FRAMES,
                pred_colors=pred_colors,
            ))
        if limit and len(scores) >= limit:
            break
    return scores

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    scores = run(args.eval_root, args.results_root, args.model, limit=args.limit)
    report = aggregate.aggregate_model_report(scores)
    print(aggregate.markdown_table(report, model_name=args.model))
    if args.out:
        aggregate.save_report(report, args.out)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()