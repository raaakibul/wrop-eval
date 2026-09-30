"""
tracker.py — recovers *predicted* object tracks from a generated video, so
they can be compared against WROP's ground-truth trajectories.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Protocol
import cv2
import numpy as np


class Tracker(Protocol):
    def track(self, video_path: str) -> Dict[str, np.ndarray]:
        ...


class BaselineBlobTracker:

    def __init__(self, min_area: int = 60, max_tracks: int = 8,
                 match_dist_frac: float = 0.12):
        self.min_area = min_area
        self.max_tracks = max_tracks
        self.match_dist_frac = match_dist_frac

    def _detect_blobs(self, frame_bgr: np.ndarray, bg_bgr: np.ndarray):
        diff = cv2.absdiff(frame_bgr, bg_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        h, w = mask.shape
        blobs = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            cx, cy = centroids[i]
            mean_color = frame_bgr[labels == i].mean(axis=0)
            blobs.append({"xy": (cx / w, cy / h), "area": area, "color": mean_color})
        return blobs