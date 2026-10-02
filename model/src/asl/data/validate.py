"""Combine the manifest with probe results into per-sample eligibility.

Only non-landmark rules are applied here (Step 1). Landmark-based rejection
rules belong to MediaPipe preprocessing (Step 2) and are recorded later as
flags on the frozen split, never by re-splitting.
"""
from __future__ import annotations

import pandas as pd

# Exclusion reasons, in the order they are checked.
NOT_PROBED = "not_probed"
DOWNLOAD_FAILED = "download_failed"
SHA_MISMATCH = "sha256_mismatch"
UNDECODABLE = "undecodable"
TOO_FEW_FRAMES = "fewer_than_min_decoded_frames"


def validate(manifest: pd.DataFrame, probe: pd.DataFrame, min_decoded_frames: int) -> pd.DataFrame:
    """Return every manifest row with probe columns, `exclusion_reason` and `eligible`."""
    v = manifest.merge(probe, on="repo_path", how="left", validate="many_to_one")
    reason = pd.Series("", index=v.index, dtype=object)
    reason[v["issue"] != ""] = v["issue"]

    open_ = reason == ""
    probed = v["download_ok"].notna()
    dl_ok = v["download_ok"].fillna(False).astype(bool)
    sha_ok = v["sha256_match"].fillna(False).astype(bool)
    dec_ok = v["decode_ok"].fillna(False).astype(bool)
    frames = v["decoded_frames"].fillna(0)

    for mask, code in [
        (~probed, NOT_PROBED),
        (probed & ~dl_ok, DOWNLOAD_FAILED),
        (dl_ok & ~sha_ok, SHA_MISMATCH),
        (sha_ok & ~dec_ok, UNDECODABLE),
        (dec_ok & (frames < min_decoded_frames), TOO_FEW_FRAMES),
    ]:
        hit = open_ & mask
        reason[hit] = code
        open_ &= ~hit

    v["exclusion_reason"] = reason
    v["eligible"] = reason == ""
    return v
