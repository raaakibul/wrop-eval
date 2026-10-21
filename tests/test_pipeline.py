import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval import cli
from wrop_eval.constants import FPS, N_TOTAL_FRAMES
from wrop_eval.demo import MODELS, run_demo, score_all_models
from wrop_eval.tracker import make_synthetic_video

RED_BGR = (30, 30, 200)
OCCLUDED = (60, 100)


def _build_fake_benchmark(root, models=("good_model", "vanish_model")):
    sample_dir = os.path.join(root, "eval", "shard1", "ball_task", "ball_0000")
    os.makedirs(sample_dir)
    n = N_TOTAL_FRAMES
    xs = np.linspace(0.08, 0.92, n)
    gt = {
        "fps": FPS, "frame_start": 1, "frame_end": n,
        "objects": {"ball": {"shape": "sphere", "role": "target",
                             "color": "red", "is_target": True}},
        "frames": [{"frame": i + 1,
                    "objects": {"ball": {"loc": [float(xs[i]), 0.0, 0.5]}}}
                   for i in range(n)],
    }
    with open(os.path.join(sample_dir, "scene_state_graph.json"), "w") as f:
        json.dump(gt, f)

    for model in models:
        out = os.path.join(root, "results", model, "ball_task")
        os.makedirs(out)
        ev = {"xy0": (0.08, 0.5), "xy1": (0.92, 0.5), "color": RED_BGR,
              "radius": 5, "occlude_frames": OCCLUDED}
        if model == "vanish_model":
            ev["vanish_after"] = OCCLUDED[0] + 2
        make_synthetic_video(os.path.join(out, "ball_0000.mp4"), [ev],
                             n_frames=n, fps=FPS)


def test_find_samples(tmp_path):
    _build_fake_benchmark(str(tmp_path))
    found = list(cli.find_samples(str(tmp_path / "eval")))
    assert len(found) == 1
    task, sample_id, _ = found[0]
    assert (task, sample_id) == ("ball", "ball_0000")


def test_cli_ranks_good_model_above_vanishing_model(tmp_path):
    _build_fake_benchmark(str(tmp_path))
    ev, res = str(tmp_path / "eval"), str(tmp_path / "results")
    good = cli.run(ev, res, "good_model")[0]
    vanish = cli.run(ev, res, "vanish_model")[0]
    assert good.disappearance_rate == 0.0
    assert vanish.disappearance_rate == 1.0
    assert good.ops > vanish.ops


def test_cli_skips_models_with_no_video(tmp_path):
    _build_fake_benchmark(str(tmp_path))
    scores = cli.run(str(tmp_path / "eval"), str(tmp_path / "results"), "no_such_model")
    assert scores == []


def test_cli_main_writes_report(tmp_path):
    _build_fake_benchmark(str(tmp_path))
    out = str(tmp_path / "report.json")
    cli.main(["--eval-root", str(tmp_path / "eval"),
              "--results-root", str(tmp_path / "results"),
              "--model", "good_model", "--out", out])
    with open(out) as f:
        report = json.load(f)
    assert report["n_total_samples"] == 1
    assert report["overall"]["OPS"] > 0.9


def test_demo_models_rank_by_severity(tmp_path):
    scores = score_all_models(str(tmp_path / "videos"))
    ops = {m: np.mean([s.ops for s in scores[m]]) for m in MODELS}
    assert ops["correct"] > 0.95
    assert ops["correct"] > ops["position_drift"]
    assert ops["position_drift"] > ops["identity_swap"]
    assert ops["position_drift"] > ops["disappear"]


def test_demo_flags_the_right_failure_modes(tmp_path):
    scores = score_all_models(str(tmp_path / "videos"))
    for s in scores["disappear"]:
        assert s.disappearance_rate == 1.0
    for s in scores["identity_swap"]:
        assert s.identity_swap_rate == 1.0
        assert s.disappearance_rate == 0.0
    for s in scores["correct"]:
        assert s.hallucinated_object_rate == 0.0


def test_run_demo_writes_outputs_and_positive_correlation(tmp_path):
    summary, _ = run_demo(str(tmp_path))
    assert summary["correlation"]["spearman_r"] > 0.5
    for name in ["validation_summary.json", "category_breakdown.png",
                 "ops_vs_elo.png", "report_correct.json"]:
        assert (tmp_path / name).exists()