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
    keys = set(npz.files)

    # Convention A: explicit position tensor + name list.
    pos_key = next((k for k in ("positions", "loc", "locations") if k in keys), None)
    name_key = next((k for k in ("object_names", "names", "objects") if k in keys), None)
    fps = int(npz["fps"]) if "fps" in keys else 24

    if pos_key is not None:
        pos = np.asarray(npz[pos_key])  # expected (n_frames, n_objects, 3)
        names = (list(npz[name_key]) if name_key is not None
                 else [f"obj_{i}" for i in range(pos.shape[1])])
        n = pos.shape[0]
        is_target = (npz["is_target"] if "is_target" in keys
                     else np.zeros(pos.shape[1], dtype=bool))
        tracks = {}
        for i, nm in enumerate(names):
            nm = nm.decode() if isinstance(nm, bytes) else str(nm)
            tracks[nm] = ObjectTrack(
                name=nm, loc=pos[:, i, :].astype(np.float64),
                is_target=bool(is_target[i]) if hasattr(is_target, "__len__") else False,
            )
        return GroundTruthTrajectory(fps=fps, frame_start=1, frame_end=n, objects=tracks)

    # Convention B (fallback): every (n_frames, 3) float array is one object's track.
    tracks = {}
    n = None
    for k in npz.files:
        arr = np.asarray(npz[k])
        if arr.ndim == 2 and arr.shape[1] == 3 and np.issubdtype(arr.dtype, np.floating):
            n = arr.shape[0] if n is None else n
            tracks[k] = ObjectTrack(name=k, loc=arr.astype(np.float64))
    if not tracks:
        raise ValueError(
            f"trajectory.npz keys {sorted(keys)} did not match any known "
            "layout. Patch wrop_eval.schema._from_npz_dict for your file."
        )
    return GroundTruthTrajectory(fps=fps, frame_start=1, frame_end=n, objects=tracks)



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
    
    