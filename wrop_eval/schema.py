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