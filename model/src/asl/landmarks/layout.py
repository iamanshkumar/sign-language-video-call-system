"""Landmark layout of the stored arrays (format v2) and feature assembly per configuration.

One Holistic pass per video is stored once, one array per landmark group, so a
configuration only reads the groups it needs (np.load is lazy per key):
    pose, face, left_hand, right_hand   [T, n, 3]  shoulder-normalized x, y, z (float16 or float32)
    pose_vis                            [T, 33]    pose visibility (same dtype)
    flags                               [T, 4]     uint8 bit field per group (GROUP_ORDER):
                                                   OBSERVED | PRESENT | INTERPOLATED
    center [T, 3], scale [T] (float32, pixel space), shoulder_measured [T] (uint8)
    active_interval [2] int32: first/last frame of the active signing stretch (-1, -1 if none)
Coordinates are 0 wherever the group is not PRESENT. All T frames (including rest
frames outside the active interval) are kept.
"""
from __future__ import annotations

import numpy as np

GROUPS: dict[str, int] = {"pose": 33, "face": 468, "left_hand": 21, "right_hand": 21}
GROUP_ORDER = list(GROUPS)
N_LANDMARKS = sum(GROUPS.values())  # 543
GROUP_INDEX = {g: i for i, g in enumerate(GROUP_ORDER)}
_start = np.cumsum([0] + list(GROUPS.values()))
SLICES: dict[str, slice] = {g: slice(int(_start[i]), int(_start[i + 1])) for i, g in enumerate(GROUP_ORDER)}

# Pose landmark indices (MediaPipe): 11 = left shoulder, 12 = right shoulder.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# flags bits
OBSERVED, PRESENT, INTERPOLATED = 1, 2, 4


def group_feature_dim(group: str) -> int:
    """x, y, z per landmark + 1 presence flag (+ per-landmark visibility for pose)."""
    n = GROUPS[group]
    return n * 3 + 1 + (n if group == "pose" else 0)


def feature_dim(groups: list[str]) -> int:
    return sum(group_feature_dim(g) for g in groups)


def feature_names(groups: list[str]) -> list[str]:
    names: list[str] = []
    for g in groups:
        for i in range(GROUPS[g]):
            names += [f"{g}_{i}_x", f"{g}_{i}_y", f"{g}_{i}_z"]
        names.append(f"{g}_present")
        if g == "pose":
            names += [f"pose_{i}_vis" for i in range(GROUPS["pose"])]
    return names


def flag_mask(flags: np.ndarray, group: str, bit: int) -> np.ndarray:
    return (flags[:, GROUP_INDEX[group]] & bit) != 0


def assemble_features(arrays, groups: list[str]) -> np.ndarray:
    """Stored arrays (dict or NpzFile) -> float32 [T, F] for the chosen groups (F = feature_dim)."""
    flags = arrays["flags"]
    t = flags.shape[0]
    parts = []
    for g in groups:
        present = flag_mask(flags, g, PRESENT).astype(np.float32)[:, None]
        parts.append(np.asarray(arrays[g], np.float32).reshape(t, -1))
        parts.append(present)
        if g == "pose":
            parts.append(np.asarray(arrays["pose_vis"], np.float32) * present)
    return np.concatenate(parts, axis=1)
