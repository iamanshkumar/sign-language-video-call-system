"""Uniform temporal resampling of a stored clip's active interval to exactly W frames.

Positions p_k = start + k (L - 1) / (W - 1), k = 0..W-1 (first and last active frame
included). Per landmark group: linear interpolation between the two neighbouring stored
frames when the group is PRESENT in both, otherwise the nearest frame's value and flags
(the same rule as Step 2's 30 FPS resampling). No padding, no random selection, no frame
copying. Works on the mixed-precision v2 arrays; outputs are float32.
"""
from __future__ import annotations

import numpy as np

from asl.landmarks.layout import GROUP_INDEX, PRESENT

_SKIP = {"flags", "center", "scale", "shoulder_measured", "active_interval", "version", "pose_vis"}


def positions(start: int, end: int, window: int) -> np.ndarray:
    """Fractional frame positions covering [start, end] inclusive with `window` samples."""
    if end < start:
        raise ValueError("empty active interval")
    if window < 2:
        raise ValueError("window must be >= 2")
    length = end - start + 1
    return start + np.arange(window, dtype=np.float64) * ((length - 1) / (window - 1))


def resample_interval(arrays, groups: list[str], start: int, end: int, window: int) -> dict[str, np.ndarray]:
    """Resample the needed groups of a v2 array set (dict or NpzFile) to `window` frames.

    Returns a dict with the same keys/semantics as the stored arrays (group coordinates
    float32 [W, n, 3], `pose_vis` float32 [W, 33] if pose is used, `flags` uint8 [W, 4]).
    """
    p = positions(start, end, window)
    i = np.floor(p).astype(np.int64)
    j = np.minimum(i + 1, end)
    a = (p - i).astype(np.float32)
    nearest = np.where(a >= 0.5, j, i)

    flags_all = np.asarray(arrays["flags"])
    flags = flags_all[nearest].copy()
    out: dict[str, np.ndarray] = {}
    for g in groups:
        gi = GROUP_INDEX[g]
        present = (flags_all[:, gi] & PRESENT) != 0
        both = present[i] & present[j]
        x = np.asarray(arrays[g], dtype=np.float32)
        lin = x[i] * (1 - a)[:, None, None] + x[j] * a[:, None, None]
        out[g] = np.where(both[:, None, None], lin, x[nearest]).astype(np.float32)
        # presence: interpolated samples are present; otherwise the nearest frame decides
        flags[:, gi] = np.where(both, flags_all[nearest, gi] | PRESENT, flags_all[nearest, gi])
        if g == "pose":
            v = np.asarray(arrays["pose_vis"], dtype=np.float32)
            out["pose_vis"] = np.where(both[:, None], v[i] * (1 - a)[:, None] + v[j] * a[:, None], v[nearest])
    out["flags"] = flags.astype(np.uint8)
    return out
