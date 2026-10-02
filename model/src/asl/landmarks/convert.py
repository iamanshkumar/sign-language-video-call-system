"""Convert Phase A (format v1) landmark arrays to the current storage format (v2).

v1 stored one float32 [T, 543, 3] array plus per-group masks. v2 splits groups, uses the
configured (mixed) precision and packs masks into uint8 flags. The conversion goes
through `pack_arrays`, the same function the extraction pipeline uses, so a converted
file is what the v2 pipeline would have written for the same MediaPipe output.
"""
from __future__ import annotations

import numpy as np

from asl.landmarks.preprocess import ActiveInterval
from asl.landmarks.process import pack_arrays


def v2_from_v1(a: dict, interval: ActiveInterval | None, out_cfg: dict) -> dict:
    return pack_arrays(a["coords"].astype(np.float64), a["present"].astype(bool), a["observed"].astype(bool),
                       a["interpolated"].astype(bool), a["pose_vis"], a["center"], a["scale"],
                       a["shoulder_measured"].astype(bool), interval, out_cfg)
