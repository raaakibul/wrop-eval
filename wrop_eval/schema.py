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