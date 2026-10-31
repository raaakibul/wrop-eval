import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval.schema import (
    load_ground_truth,
    load_metadata,
    load_scene_state_graph,
    load_trajectory_npz,
)


def _write_ssg(path):
    """3-frame scene with one target ball; the ball is missing in frame 2."""
    data = {
        "fps": 24, "frame_start": 1, "frame_end": 3,
        "objects": {
            "ball": {"shape": "sphere", "role": "target", "color": "red", "is_target": True},
            "wall": {"shape": "box", "role": "occluder", "color": "gray", "is_target": False},
        },
        "frames": [
            {"frame": 1, "objects": {"ball": {"loc": [0.0, 0.0, 1.0]},
                                     "wall": {"loc": [5.0, 0.0, 0.0]}}},
            {"frame": 2, "objects": {"wall": {"loc": [5.0, 0.0, 0.0]}}},
            {"frame": 3, "objects": {"ball": {"loc": [2.0, 0.0, 1.0]},
                                     "wall": {"loc": [5.0, 0.0, 0.0]}}},
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def test_load_scene_state_graph(tmp_path):
    p = tmp_path / "scene_state_graph.json"
    _write_ssg(p)
    gt = load_scene_state_graph(str(p))
    assert gt.n_frames == 3
    assert gt.fps == 24
    ball = gt.objects["ball"]
    assert ball.color == "red"
    assert ball.is_target is True
    assert list(ball.loc[0]) == [0.0, 0.0, 1.0]
    assert np.isnan(ball.loc[1]).all()      # missing frame stays NaN
    assert list(ball.loc[2]) == [2.0, 0.0, 1.0]


def test_target_objects_filters_by_flag(tmp_path):
    p = tmp_path / "scene_state_graph.json"
    _write_ssg(p)
    gt = load_scene_state_graph(str(p))
    assert [o.name for o in gt.target_objects()] == ["ball"]


def test_load_ground_truth_finds_json(tmp_path):
    _write_ssg(tmp_path / "scene_state_graph.json")
    gt = load_ground_truth(str(tmp_path))
    assert "ball" in gt.objects


def test_load_ground_truth_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ground_truth(str(tmp_path))


def test_load_metadata_missing_returns_empty(tmp_path):
    assert load_metadata(str(tmp_path)) == {}


def test_load_metadata_reads_json(tmp_path):
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"task": "ball_drop"}, f)
    assert load_metadata(str(tmp_path)) == {"task": "ball_drop"}

def _write_real_layout_npz(path, n_frames=120, n_bodies=4):
    """An .npz with the same keys, shapes and dtypes as a real WROP sample."""
    xpos = np.zeros((n_frames, n_bodies, 3), np.float32)
    xpos[:, 1, 0] = np.linspace(0.0, 3.0, n_frames)   # body 1 moves along x
    xpos[:, 1, 2] = 0.5
    np.savez(
        path,
        xpos=xpos, xquat=np.zeros((n_frames, n_bodies, 4), np.float32),
        qpos=np.zeros((n_frames, 91), np.float32), qvel=np.zeros((n_frames, 78), np.float32),
        body_names=np.array(["accordion_panel_0", "ball", "cube", "table"]),
        body_roles=np.array(["occluder_panel", "target", "target", "support"]),
        body_shapes=np.array(["cube", "sphere", "cube", "cube"]),
        body_colors=np.array(["gray", "orange", "navy", "gray"]),
        is_target=np.array([False, True, True, False]),
        fps=np.float32(24.0), frame_start=np.int32(1), frame_end=np.int32(n_frames),
    )


def test_load_real_layout_npz(tmp_path):
    p = tmp_path / "trajectory.npz"
    _write_real_layout_npz(p)
    gt = load_trajectory_npz(str(p))
    assert gt.n_frames == 120 and gt.fps == 24
    assert set(gt.objects) == {"accordion_panel_0", "ball", "cube", "table"}
    assert [o.name for o in gt.target_objects()] == ["ball", "cube"]
    ball = gt.objects["ball"]
    assert ball.color == "orange" and ball.shape == "sphere" and ball.role == "target"
    assert ball.loc.shape == (120, 3)
    assert ball.loc[-1, 0] == pytest.approx(3.0)
    assert ball.loc.dtype == np.float64


def test_load_ground_truth_finds_npz(tmp_path):
    _write_real_layout_npz(tmp_path / "trajectory.npz")
    gt = load_ground_truth(str(tmp_path))
    assert "ball" in gt.objects


def test_npz_with_unknown_layout_gives_a_clear_error(tmp_path):
    p = tmp_path / "trajectory.npz"
    np.savez(p, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="missing"):
        load_trajectory_npz(str(p))


def test_duplicate_body_names_are_kept_apart(tmp_path):
    p = tmp_path / "trajectory.npz"
    np.savez(p, xpos=np.zeros((5, 2, 3), np.float32), body_names=np.array(["box", "box"]))
    gt = load_trajectory_npz(str(p))
    assert len(gt.objects) == 2


def test_real_sample_if_downloaded():
    # Runs only on a machine where scripts/inspect_dataset.py was run.
    here = os.path.join(os.path.dirname(__file__), "..")
    found = glob.glob(os.path.join(here, "data", "inspect", "**", "trajectory.npz"),
                      recursive=True)
    if not found:
        pytest.skip("no real sample in data/inspect")
    gt = load_trajectory_npz(found[0])
    assert gt.n_frames == 120
    assert len(gt.target_objects()) >= 1