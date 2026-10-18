import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wrop_eval.tracker import BaselineBlobTracker, make_synthetic_video

RED_BGR = (30, 30, 200)


def _track(tmp_path, events, n_frames=120):
    path = os.path.join(str(tmp_path), "clip.mp4")
    make_synthetic_video(path, events, n_frames=n_frames)
    return BaselineBlobTracker().track_with_color(path)


def _observed(arr):
    return ~np.isnan(arr[:, :2]).any(axis=1)


def test_visible_object_gives_one_track(tmp_path):
    ev = {"xy0": (0.1, 0.3), "xy1": (0.4, 0.3), "color": RED_BGR, "radius": 5}
    positions, colors = _track(tmp_path, [ev], n_frames=60)
    assert len(positions) == 1
    arr = next(iter(positions.values()))
    assert _observed(arr).sum() >= 55  # essentially every frame


# def test_color_is_reported_in_bgr(tmp_path):
#     ev = {"xy0": (0.1, 0.3), "xy1": (0.4, 0.3), "color": RED_BGR, "radius": 5}
#     positions, colors = _track(tmp_path, [ev], n_frames=60)
#     carr = next(iter(colors.values()))
#     mean_bgr = np.nanmean(carr, axis=0)
#     assert np.allclose(mean_bgr, RED_BGR, atol=15)

def _mean_color(tmp_path, color_bgr):
    ev = {"xy0": (0.1, 0.3), "xy1": (0.4, 0.3), "color": color_bgr, "radius": 5}
    _, colors = _track(tmp_path, [ev], n_frames=60)
    return np.nanmean(next(iter(colors.values())), axis=0)


def test_color_is_reported_in_bgr_order(tmp_path):
    # Edge blur and video compression wash colors out, so we check which
    # channel dominates rather than exact values. A red ball must have its
    # largest value in channel 2 (R in BGR order); a blue ball in channel 0.
    red = _mean_color(tmp_path, (30, 30, 200))
    assert np.argmax(red) == 2
    assert red[2] > 150 and red[0] < 100 and red[1] < 100

    blue = _mean_color(tmp_path, (200, 60, 30))
    assert np.argmax(blue) == 0
    assert blue[0] > 150 and blue[2] < 100

def test_occlusion_shows_up_as_missing_frames(tmp_path):
    ev = {"xy0": (0.08, 0.5), "xy1": (0.92, 0.5), "color": RED_BGR,
          "radius": 5, "occlude_frames": (60, 100)}
    positions, _ = _track(tmp_path, [ev])
    arr = next(iter(positions.values()))
    seen = _observed(arr)
    assert seen[10]          # visible before the occluder
    assert not seen[80]      # hidden behind the occluder
    assert seen[110]         # visible again afterwards


def test_object_rejoins_its_track_after_long_occlusion(tmp_path):
    # Regression test: a ~28-frame gap used to split the ball into two tracks.
    ev = {"xy0": (0.08, 0.5), "xy1": (0.92, 0.5), "color": RED_BGR,
          "radius": 5, "occlude_frames": (60, 100)}
    positions, _ = _track(tmp_path, [ev])
    assert len(positions) == 1


def test_vanished_object_never_reappears(tmp_path):
    ev = {"xy0": (0.08, 0.5), "xy1": (0.92, 0.5), "color": RED_BGR,
          "radius": 5, "occlude_frames": (60, 100), "vanish_after": 62}
    positions, _ = _track(tmp_path, [ev])
    arr = next(iter(positions.values()))
    assert not _observed(arr)[96:].any()


def test_missing_video_gives_empty_result(tmp_path):
    positions, colors = BaselineBlobTracker().track_with_color(
        os.path.join(str(tmp_path), "does_not_exist.mp4"))
    assert positions == {} and colors == {}