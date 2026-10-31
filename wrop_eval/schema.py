"""
schema.py — canonical in-memory representation of a WROP sample's ground
truth.

Ground truth format, confirmed from the generator's source
(object_permanence/generator/core/render.py::record_scene_state_graph
in https://github.com/hokindeng/object-permanence):

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import json
import os


@dataclass
class ObjectTrack:
    name: str
    shape: str = ""
    role: str = ""
    color: str = ""
    is_target: bool = False
    loc: Optional[np.ndarray] = None
    rot: Optional[np.ndarray] = None


@dataclass
class GroundTruthTrajectory:
    fps: int
    frame_start: int
    frame_end: int
    objects: Dict[str, ObjectTrack] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start + 1

    def target_objects(self) -> List[ObjectTrack]:
        return [o for o in self.objects.values() if o.is_target]
    

def load_scene_state_graph(path: str) -> GroundTruthTrajectory:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    fps = int(d.get("fps", 24))
    frame_start = int(d["frame_start"])
    frame_end = int(d["frame_end"])
    n = frame_end - frame_start + 1

    meta = d.get("objects", {})
    tracks: Dict[str, ObjectTrack] = {
        name: ObjectTrack(
            name=name,
            shape=m.get("shape", ""),
            role=m.get("role", ""),
            color=m.get("color", ""),
            is_target=bool(m.get("is_target", False)),
            loc=np.full((n, 3), np.nan, dtype=np.float64),
            rot=np.full((n, 3), np.nan, dtype=np.float64),
        )
        for name, m in meta.items()
    }

    for fr in d.get("frames", []):
        idx = int(fr["frame"]) - frame_start
        if not (0 <= idx < n):
            continue
        for name, st in fr.get("objects", {}).items():
            if name not in tracks:
                # object appears in frames but not in the objects manifest
                tracks[name] = ObjectTrack(
                    name=name,
                    loc=np.full((n, 3), np.nan, dtype=np.float64),
                    rot=np.full((n, 3), np.nan, dtype=np.float64),
                )
            tracks[name].loc[idx] = st.get("loc", [np.nan] * 3)
            if "rot" in st:
                tracks[name].rot[idx] = st["rot"]

    return GroundTruthTrajectory(fps=fps, frame_start=frame_start,
                                  frame_end=frame_end, objects=tracks)
    
def _from_npz_dict(npz: np.lib.npyio.NpzFile) -> GroundTruthTrajectory:
    """Adapter for the REAL trajectory.npz layout, confirmed by streaming a
    sample from the dataset (MuJoCo-style arrays):

        xpos         (n_frames, n_bodies, 3)  world position of each body
        xquat        (n_frames, n_bodies, 4)  orientation quaternion
        qpos, qvel   (n_frames, ...)          generalized coordinates
        body_names   (n_bodies,)  body_roles, body_shapes, body_colors
        is_target    (n_bodies,) bool
        fps, frame_start, frame_end           scalars

    Only the fields the metric needs are read.
    """
    keys = set(npz.files)
    missing = {"xpos", "body_names"} - keys
    if missing:
        raise ValueError(
            f"trajectory.npz is missing {sorted(missing)}; it has keys "
            f"{sorted(keys)}. Update wrop_eval.schema._from_npz_dict."
        )

    xpos = np.asarray(npz["xpos"], dtype=np.float64)
    n_frames, n_bodies = xpos.shape[0], xpos.shape[1]
    names = [str(x) for x in npz["body_names"]]

    def text_column(key):
        return [str(x) for x in npz[key]] if key in keys else [""] * n_bodies

    roles = text_column("body_roles")
    shapes = text_column("body_shapes")
    colors_ = text_column("body_colors")
    is_target = (np.asarray(npz["is_target"], dtype=bool) if "is_target" in keys
                 else np.zeros(n_bodies, dtype=bool))
    fps = int(round(float(npz["fps"]))) if "fps" in keys else 24
    frame_start = int(npz["frame_start"]) if "frame_start" in keys else 1

    tracks = {}
    for i, name in enumerate(names):
        key = name if name not in tracks else f"{name}#{i}"
        tracks[key] = ObjectTrack(
            name=key, shape=shapes[i], role=roles[i], color=colors_[i],
            is_target=bool(is_target[i]), loc=xpos[:, i, :].copy(),
        )
    return GroundTruthTrajectory(
        fps=fps, frame_start=frame_start,
        frame_end=frame_start + n_frames - 1, objects=tracks,
    )

def load_trajectory_npz(path: str) -> GroundTruthTrajectory:
    with np.load(path, allow_pickle=True) as npz:
        return _from_npz_dict(npz)
    
    
def load_ground_truth(sample_dir: str) -> GroundTruthTrajectory:
    ssg = os.path.join(sample_dir, "scene_state_graph.json")
    npz = os.path.join(sample_dir, "trajectory.npz")
    if os.path.isfile(ssg):
        return load_scene_state_graph(ssg)
    if os.path.isfile(npz):
        return load_trajectory_npz(npz)
    raise FileNotFoundError(
        f"No scene_state_graph.json or trajectory.npz in {sample_dir}"
    )


def load_metadata(sample_dir: str) -> dict:
    path = os.path.join(sample_dir, "metadata.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
    