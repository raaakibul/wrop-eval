"""
Builds a tiny fake benchmark folder with the same layout as the real WROP
benchmark, so wrop_eval.cli can be tested without the real dataset.

Layout produced (under test_out/fake):
  eval/shard1/ball_task/ball_0000/scene_state_graph.json
  results/good_model/ball_task/ball_0000.mp4       (reappears correctly)
  results/vanish_model/ball_task/ball_0000.mp4     (never reappears)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval.constants import FPS, N_TOTAL_FRAMES
from wrop_eval.tracker import make_synthetic_video

ROOT = os.path.join("test_out", "fake")
RED_BGR = (30, 30, 200)
OCCLUDED = (60, 100)  # frames where the ball is hidden behind the occluder


def write_ground_truth(sample_dir):
    n = N_TOTAL_FRAMES
    xs = np.linspace(0.08, 0.92, n)
    frames = []
    for i in range(n):
        frames.append({
            "frame": i + 1,
            "objects": {"ball": {"loc": [float(xs[i]), 0.0, 0.5],
                                 "rot": [0.0, 0.0, 0.0]}},
        })
    gt = {
        "fps": FPS, "frame_start": 1, "frame_end": n,
        "objects": {"ball": {"shape": "sphere", "role": "target",
                             "color": "red", "is_target": True}},
        "frames": frames,
    }
    with open(os.path.join(sample_dir, "scene_state_graph.json"), "w") as f:
        json.dump(gt, f)


def write_model_video(path, behavior):
    ev = {"xy0": (0.08, 0.5), "xy1": (0.92, 0.5), "color": RED_BGR,
          "radius": 5, "occlude_frames": OCCLUDED}
    if behavior == "vanish":
        ev["vanish_after"] = OCCLUDED[0] + 2
    make_synthetic_video(path, [ev], n_frames=N_TOTAL_FRAMES, fps=FPS)


def main():
    sample_dir = os.path.join(ROOT, "eval", "shard1", "ball_task", "ball_0000")
    os.makedirs(sample_dir, exist_ok=True)
    write_ground_truth(sample_dir)

    for model, behavior in [("good_model", "good"), ("vanish_model", "vanish")]:
        out_dir = os.path.join(ROOT, "results", model, "ball_task")
        os.makedirs(out_dir, exist_ok=True)
        write_model_video(os.path.join(out_dir, "ball_0000.mp4"), behavior)

    print("Fake benchmark written to", ROOT)


if __name__ == "__main__":
    main()