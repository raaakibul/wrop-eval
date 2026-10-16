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


def test_load_trajectory_npz_positions_layout(tmp_path):
    # Tests the permissive guess-layout loader, NOT the confirmed real layout.
    pos = np.zeros((4, 2, 3))
    pos[:, 0, 0] = np.arange(4)
    p = tmp_path / "trajectory.npz"
    np.savez(p, positions=pos, object_names=np.array(["ball", "wall"]),
             is_target=np.array([True, False]))
    gt = load_trajectory_npz(str(p))
    assert gt.n_frames == 4
    assert gt.objects["ball"].is_target is True
    assert list(gt.objects["ball"].loc[:, 0]) == [0.0, 1.0, 2.0, 3.0]