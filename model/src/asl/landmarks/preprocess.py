"""Pure-numpy landmark preprocessing: 30 FPS resampling, gap handling,
shoulder normalization and rejection rules. No MediaPipe here, so every rule is unit-testable.

Array conventions (per video):
    coords  [T, L, 3]  landmark coordinates
    present [T, G]     per-group presence (bool)
    group slices map landmarks to groups (see asl.landmarks.layout)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asl.landmarks.layout import GROUP_INDEX, GROUP_ORDER, LEFT_SHOULDER, RIGHT_SHOULDER, SLICES


# ---------------------------------------------------------------- temporal resampling
def standardized_length(n_frames: int, fps: float, target_fps: float) -> int:
    """Frames needed to represent the clip's duration (n_frames / fps) at target_fps."""
    return max(1, int(round(n_frames / fps * target_fps)))


def resample_uniform(
    times: np.ndarray, coords: np.ndarray, present: np.ndarray, vis: np.ndarray, fps: float, target_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform temporal resampling of landmark time series onto a target_fps grid.

    Target times are t0 + k / target_fps. A group's value at a target time is the
    linear interpolation of the two neighbouring source frames when the group is
    present in both; otherwise the nearest source frame is used (value and presence).
    Frames are never zero-padded, and for upsampled clips landmarks are interpolated
    between neighbours rather than copied from repeated frames.
    """
    n = len(times)
    m = standardized_length(n, fps, target_fps)
    tt = times[0] + np.arange(m) / target_fps
    tt = np.minimum(tt, times[-1])
    hi = np.clip(np.searchsorted(times, tt, side="left"), 1, n - 1) if n > 1 else np.zeros(m, int)
    lo = np.maximum(hi - 1, 0)
    span = np.where(times[hi] > times[lo], times[hi] - times[lo], 1.0)
    a = np.clip((tt - times[lo]) / span, 0.0, 1.0)  # weight of the `hi` frame
    nearest = np.where(a >= 0.5, hi, lo)

    out_c = np.empty((m,) + coords.shape[1:], coords.dtype)
    out_p = present[nearest].copy()
    for g in GROUP_ORDER:
        sl, gi = SLICES[g], GROUP_INDEX[g]
        both = present[lo, gi] & present[hi, gi]
        lin = coords[lo, sl] * (1 - a)[:, None, None] + coords[hi, sl] * a[:, None, None]
        out_c[:, sl] = np.where(both[:, None, None], lin, coords[nearest, sl])
        out_p[:, gi] = np.where(both, True, present[nearest, gi])
    pose = GROUP_INDEX["pose"]
    both_pose = (present[lo, pose] & present[hi, pose])[:, None]
    out_v = np.where(both_pose, vis[lo] * (1 - a)[:, None] + vis[hi] * a[:, None], vis[nearest])
    return tt, out_c, out_p, out_v


# ---------------------------------------------------------------- missing landmarks
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of consecutive True runs."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def longest_run(mask: np.ndarray) -> int:
    r = runs(mask)
    return max((length for _, length in r), default=0)


@dataclass
class GapResult:
    coords: np.ndarray       # [T, n, 3]
    present: np.ndarray      # [T] bool: observed or interpolated
    interpolated: np.ndarray  # [T] bool
    zero_filled: np.ndarray  # [T] bool


def fill_gaps(coords: np.ndarray, present: np.ndarray, max_gap: int) -> GapResult:
    """Interior gaps of <= max_gap frames are linearly interpolated between the
    neighbouring observed frames. Longer gaps, and gaps at the start or end of the
    clip (no observation on both sides), are zero-filled with presence 0."""
    c = coords.copy()
    p = np.asarray(present, bool).copy()
    interp = np.zeros(len(p), bool)
    t = len(p)
    for s, length in runs(~p):
        e = s + length  # first observed frame after the gap
        if s == 0 or e >= t or length > max_gap:
            continue
        w = (np.arange(1, length + 1) / (length + 1))[:, None, None]
        c[s:e] = c[s - 1] * (1 - w) + c[e] * w
        p[s:e] = True
        interp[s:e] = True
    zero = ~p
    c[zero] = 0.0
    return GapResult(c, p, interp, zero)


# ---------------------------------------------------------------- normalization
@dataclass
class BodyFrame:
    center: np.ndarray    # [T, 3]
    scale: np.ndarray     # [T]
    measured: np.ndarray  # [T] bool: shoulders valid in this frame
    ok: bool              # at least one valid frame exists


def body_frame(pose: np.ndarray, pose_present: np.ndarray, pose_vis: np.ndarray, vis_min: float,
               min_scale: float = 1e-6) -> BodyFrame:
    """Shoulder midpoint (origin) and 2D shoulder distance (scale) per frame.

    Frames without valid shoulders reuse the last valid frame; frames before the
    first valid frame use the first valid one. If no frame is valid, ok=False.
    """
    ls, rs = pose[:, LEFT_SHOULDER], pose[:, RIGHT_SHOULDER]
    center = (ls + rs) / 2.0
    scale = np.linalg.norm((ls - rs)[:, :2], axis=1)
    valid = (np.asarray(pose_present, bool)
             & (pose_vis[:, LEFT_SHOULDER] >= vis_min) & (pose_vis[:, RIGHT_SHOULDER] >= vis_min)
             & (scale > min_scale) & np.isfinite(center).all(axis=1) & np.isfinite(scale))
    t = len(valid)
    if not valid.any():
        return BodyFrame(np.zeros((t, 3)), np.ones(t), valid, False)
    idx = np.where(valid, np.arange(t), -1)
    idx = np.maximum.accumulate(idx)           # last valid frame (forward fill)
    idx[idx < 0] = int(np.flatnonzero(valid)[0])  # leading frames: first valid frame
    return BodyFrame(center[idx], scale[idx], valid, True)


def normalize(coords: np.ndarray, present_per_landmark: np.ndarray, bf: BodyFrame) -> np.ndarray:
    """(coordinate - shoulder_center) / shoulder_distance; non-present landmarks stay 0."""
    out = (coords - bf.center[:, None, :]) / bf.scale[:, None, None]
    return np.where(present_per_landmark[:, :, None], out, 0.0)


def to_pixel_space(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    """MediaPipe x, y are normalized by width/height; z uses roughly the x scale.
    Converting to pixels keeps distances isotropic before shoulder normalization."""
    return coords * np.array([width, height, width], dtype=coords.dtype)


# ---------------------------------------------------------------- active signing interval
@dataclass
class ActiveInterval:
    start: int  # first frame (inclusive), 30 FPS grid
    end: int    # last frame (inclusive)

    @property
    def frames(self) -> int:
        return self.end - self.start + 1


def active_interval(hands_observed: np.ndarray, min_run: int) -> ActiveInterval | None:
    """Active signing stretch on the 30 FPS grid.

    A frame "has a hand" when either hand is detected. A meaningful hand detection
    is a run of at least `min_run` consecutive hand frames (isolated 1-2 frame
    detections, e.g. a hand briefly entering the frame edge, are not meaningful on
    their own). The interval runs from the first frame of the first meaningful run
    to the last frame of the last meaningful run. Detections between them, and gaps
    between them, are inside the interval. Returns None when no meaningful run exists.
    """
    any_hand = np.asarray(hands_observed, bool).any(axis=1)
    meaningful = [(s, n) for s, n in runs(any_hand) if n >= min_run]
    if not meaningful:
        return None
    first, last = meaningful[0], meaningful[-1]
    return ActiveInterval(first[0], last[0] + last[1] - 1)


def hand_quality(hands_observed: np.ndarray) -> dict:
    """Both-hands-missing fraction and longest run for a [T, 2] detection slice."""
    both = ~np.asarray(hands_observed, bool).any(axis=1)
    return {"both_missing_frac": float(both.mean()) if len(both) else 1.0,
            "longest_both_missing_run": longest_run(both), "both_missing_frames": int(both.sum())}


# ---------------------------------------------------------------- rejection rules
def rejection_reason(decoded_frames: int, hands_observed: np.ndarray | None, rules: dict) -> str | None:
    """Step 2 rejection rules (methodology v2).

    - fewer than `min_decoded_frames` decoded frames in the whole clip;
    - no active signing interval (no meaningful hand detection);
    - inside the active interval only: both hands missing in more than
      `max_both_hands_missing_frac` of frames, or for more than
      `max_both_hands_missing_run` consecutive frames.
    Resting frames before/after the interval are not counted as missing. A short active
    interval is only flagged in the metadata (`short_active_interval`), never rejected.
    Decode/MediaPipe errors and invalid values are handled by the caller.
    """
    if decoded_frames < rules["min_decoded_frames"]:
        return "too_few_decoded_frames"
    iv = active_interval(hands_observed, rules["active_interval_min_hand_run"])
    if iv is None:
        return "no_active_signing_interval"
    q = hand_quality(np.asarray(hands_observed, bool)[iv.start : iv.end + 1])
    if q["both_missing_frac"] > rules["max_both_hands_missing_frac"]:
        return "active_both_hands_missing_fraction"
    if q["longest_both_missing_run"] > rules["max_both_hands_missing_run"]:
        return "active_both_hands_missing_run"
    return None


def fullclip_rejection_reason(decoded_frames: int, hands_observed: np.ndarray, rules: dict) -> str | None:
    """Methodology v1 rule (whole clip, including rest frames). Recorded for comparison only."""
    if decoded_frames < rules["min_decoded_frames"]:
        return "too_few_decoded_frames"
    q = hand_quality(hands_observed)
    if q["both_missing_frac"] > rules["max_both_hands_missing_frac"]:
        return "both_hands_missing_fraction"
    if q["longest_both_missing_run"] > rules["max_both_hands_missing_run"]:
        return "both_hands_missing_run"
    return None
