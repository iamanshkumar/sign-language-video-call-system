"""Decode a video with PyAV and run MediaPipe Holistic on every decoded frame.

Frames are streamed (never all held in RAM). A fresh Holistic instance is used
per video so tracking state never carries over between clips.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import av
import numpy as np

from asl.landmarks.layout import GROUP_INDEX, GROUP_ORDER, GROUPS, N_LANDMARKS, SLICES

_RESULT_FIELDS = {"pose": "pose_landmarks", "face": "face_landmarks",
                  "left_hand": "left_hand_landmarks", "right_hand": "right_hand_landmarks"}


@dataclass
class RawLandmarks:
    times: np.ndarray       # [N] seconds
    coords: np.ndarray      # [N, 543, 3] MediaPipe image-normalized x, y, z
    observed: np.ndarray    # [N, 4] bool per group
    pose_vis: np.ndarray    # [N, 33]
    repaired_nonfinite: int  # group-frames dropped because MediaPipe returned NaN/Inf
    width: int
    height: int
    codec: str
    container_fps: float | None
    decode_s: float
    mediapipe_s: float


def make_holistic(mp_cfg: dict):
    import mediapipe as mp  # imported lazily: heavy, and only needed in worker processes

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=mp_cfg["model_complexity"],
        refine_face_landmarks=mp_cfg["refine_face_landmarks"],
        min_detection_confidence=mp_cfg["min_detection_confidence"],
        min_tracking_confidence=mp_cfg["min_tracking_confidence"],
    )


def extract_landmarks(path: str, mp_cfg: dict) -> RawLandmarks:
    """Raises on decode errors; the caller records them as rejections."""
    times, coords, observed, vis = [], [], [], []
    repaired = 0
    decode_s = mp_s = 0.0
    with av.open(path, metadata_errors="ignore") as container, make_holistic(mp_cfg) as holistic:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        cc = stream.codec_context
        fps_guess = float(stream.average_rate) if stream.average_rate else 30.0
        t0 = time.perf_counter()
        for i, frame in enumerate(container.decode(stream)):
            rgb = frame.to_ndarray(format="rgb24")
            t1 = time.perf_counter()
            res = holistic.process(rgb)
            t2 = time.perf_counter()
            decode_s += t1 - t0
            mp_s += t2 - t1

            c = np.zeros((N_LANDMARKS, 3), np.float32)
            o = np.zeros(len(GROUP_ORDER), bool)
            v = np.zeros(GROUPS["pose"], np.float32)
            for g in GROUP_ORDER:
                lms = getattr(res, _RESULT_FIELDS[g])
                if lms is None:
                    continue
                arr = np.array([(p.x, p.y, p.z) for p in lms.landmark], np.float32)
                if arr.shape[0] != GROUPS[g] or not np.isfinite(arr).all():
                    repaired += 1  # treated as a missing detection, repaired by gap handling
                    continue
                c[SLICES[g]] = arr
                o[GROUP_INDEX[g]] = True
                if g == "pose":
                    v = np.array([p.visibility for p in lms.landmark], np.float32)
            times.append(frame.time if frame.time is not None else i / fps_guess)
            coords.append(c)
            observed.append(o)
            vis.append(v)
            t0 = time.perf_counter()
    if not times:
        raise ValueError("no decodable frames")
    return RawLandmarks(
        times=np.asarray(times, np.float64), coords=np.stack(coords), observed=np.stack(observed),
        pose_vis=np.stack(vis), repaired_nonfinite=repaired, width=cc.width, height=cc.height,
        codec=cc.name, container_fps=float(stream.average_rate) if stream.average_rate else None,
        decode_s=decode_s, mediapipe_s=mp_s,
    )
