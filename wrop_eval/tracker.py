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
    
    def track(self, video_path: str) -> Dict[str, np.ndarray]:
        positions, _colors = self.track_with_color(video_path)
        return positions

    def track_with_color(self, video_path: str):
        """Like `track`, but also returns {track_id: (n_frames,3) mean
        BGR color per frame, NaN where not observed} -- used later for
        the identity-swap check in metrics.py."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            return {}, {}
        idxs = np.linspace(0, len(frames) - 1, min(15, len(frames))).astype(int)
        bg = np.median(np.stack([frames[i] for i in idxs], axis=0), axis=0).astype(np.uint8)

        n_frames = len(frames)
        tracks: Dict[str, List[Optional[tuple]]] = {}
        track_colors: Dict[str, List[Optional[np.ndarray]]] = {}
        next_id = 0
        active: Dict[str, dict] = {}  # id -> last known {xy, color}

        for t, fr in enumerate(frames):
            blobs = self._detect_blobs(fr, bg)
            unmatched_tracks = set(active.keys())
            used_blobs = set()

            pairs = []
            for tid, st in active.items():
                for bi, b in enumerate(blobs):
                    d = np.hypot(b["xy"][0] - st["xy"][0], b["xy"][1] - st["xy"][1])
                    if d <= self.match_dist_frac:
                        pairs.append((d, tid, bi))
            pairs.sort(key=lambda p: p[0])
            for d, tid, bi in pairs:
                if tid not in unmatched_tracks or bi in used_blobs:
                    continue
                b = blobs[bi]
                tracks[tid][t] = (b["xy"][0], b["xy"][1], 1.0)
                track_colors[tid][t] = b["color"]
                active[tid] = {"xy": b["xy"], "color": b["color"]}
                unmatched_tracks.discard(tid)
                used_blobs.add(bi)

            for tid in unmatched_tracks:
                tracks[tid][t] = None
                track_colors[tid][t] = None

            if len(active) < self.max_tracks:
                for bi, b in enumerate(blobs):
                    if bi in used_blobs:
                        continue
                    tid = f"pred_{next_id}"
                    next_id += 1
                    tracks[tid] = [None] * n_frames
                    track_colors[tid] = [None] * n_frames
                    tracks[tid][t] = (b["xy"][0], b["xy"][1], 1.0)
                    track_colors[tid][t] = b["color"]
                    active[tid] = {"xy": b["xy"], "color": b["color"]}

        positions, colors = {}, {}
        for tid, seq in tracks.items():
            arr = np.full((n_frames, 3), np.nan, dtype=np.float64)
            carr = np.full((n_frames, 3), np.nan, dtype=np.float64)
            for t, v in enumerate(seq):
                if v is not None:
                    arr[t] = v
                cv_ = track_colors[tid][t]
                if cv_ is not None:
                    carr[t] = cv_
            positions[tid] = arr
            colors[tid] = carr
        return positions, colors
    
def make_synthetic_video(path: str, events: List[dict], size=(160, 90),
                          n_frames: int = 60, fps: int = 24) -> None:
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    occ_x0, occ_x1 = int(0.55 * w), int(0.75 * w)
    for t in range(n_frames):
        frame = np.full((h, w, 3), 235, dtype=np.uint8)
        cv2.rectangle(frame, (occ_x0, 0), (occ_x1, h), (60, 60, 60), -1)
        for ev in events:
            frac = t / max(1, n_frames - 1)
            x = ev["xy0"][0] + frac * (ev["xy1"][0] - ev["xy0"][0])
            y = ev["xy0"][1] + frac * (ev["xy1"][1] - ev["xy0"][1])
            px, py = int(x * w), int(y * h)
            occl = ev.get("occlude_frames")
            behind_occluder = occ_x0 <= px <= occ_x1
            vanish_after = ev.get("vanish_after")
            if vanish_after is not None and t >= vanish_after:
                continue
            color = ev["color"]
            swap_after = ev.get("identity_swap_color_after")
            swap_frame = ev.get("identity_swap_frame")
            if swap_after is not None and swap_frame is not None and t >= swap_frame:
                color = swap_after
            if behind_occluder and occl is not None and occl[0] <= t <= occl[1]:
                continue
            cv2.circle(frame, (px, py), ev.get("radius", 6), color, -1)
        vw.write(frame)
    vw.release()