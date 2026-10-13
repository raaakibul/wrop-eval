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