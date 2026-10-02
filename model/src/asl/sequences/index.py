"""Sequence index: one row per accepted video, linking the frozen split, the label mapping
and the landmark store. No sequence arrays are materialized (see docs/step3_methodology.md §5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asl.landmarks.outputs import _has_file, check_split_preserved
from asl.sequences.labels import encode

SPLITS = ("train", "val", "test")
INTERVAL_BINS = [(0, 30, "<30"), (30, 45, "30-44"), (45, 60, "45-59"), (60, 10**9, ">=60")]


def interval_bin(length: int) -> str:
    for lo, hi, name in INTERVAL_BINS:
        if lo <= length < hi:
            return name
    raise ValueError(length)


def build_index(meta: pd.DataFrame, frozen: pd.DataFrame, mapping: dict, arrays_dir: Path,
                windows: list[int], short_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (index of accepted videos, excluded videos with reasons). Fails loudly on any
    disagreement with the frozen split or the stored arrays."""
    problems = check_split_preserved(meta, frozen)
    if problems:
        raise ValueError(f"landmark metadata disagrees with the frozen split: {problems[:5]}")
    if meta["sample_id"].duplicated().any():
        raise ValueError("duplicate sample ids in landmark metadata")

    rejected = meta["rejected"].astype(bool) | meta["rejection_reason"].fillna("").ne("")
    has_file = meta["array_file"].map(_has_file)
    excluded = meta.loc[rejected | ~has_file, ["sample_id", "word", "split", "rejection_reason"]].copy()
    excluded["rejection_reason"] = excluded["rejection_reason"].fillna("no_array_file")

    acc = meta.loc[~rejected & has_file, ["sample_id", "array_file", "active_start_frame", "active_end_frame",
                                          "std_frame_count"]].copy()
    cols = ["sample_id", "repo_path", "word", "split", "sha256", "dup_group_size", "split_group", "aug_source"]
    idx = acc.merge(frozen[cols], on="sample_id", how="left", validate="one_to_one")  # split from frozen file
    idx["label"] = encode(idx["word"], mapping)
    idx["active_start_frame"] = idx["active_start_frame"].astype(int)
    idx["active_end_frame"] = idx["active_end_frame"].astype(int)

    # The stored array must agree with the metadata interval and frame count.
    for r in idx.itertuples(index=False):
        with np.load(arrays_dir / r.array_file) as z:
            s, e = (int(x) for x in z["active_interval"])
            t = int(z["flags"].shape[0])
        if (s, e) != (r.active_start_frame, r.active_end_frame) or t != int(r.std_frame_count):
            raise ValueError(f"{r.sample_id}: array interval/length disagree with metadata")
    idx["active_frames"] = idx["active_end_frame"] - idx["active_start_frame"] + 1
    if (idx["active_frames"] < 2).any():
        raise ValueError("active interval shorter than 2 frames cannot be resampled")
    idx["active_bin"] = idx["active_frames"].map(interval_bin)
    idx["short_active_interval"] = idx["active_frames"] < short_frames
    for w in windows:
        idx[f"stretch_{w}"] = w / idx["active_frames"]  # > 1 upsampling, < 1 downsampling
    idx = idx.sort_values("sample_id", ignore_index=True)

    leaks = check_index_leakage(idx)
    if leaks:
        raise ValueError(f"leakage in sequence index: {leaks}")
    return idx, excluded.sort_values("sample_id", ignore_index=True)


def check_index_leakage(idx: pd.DataFrame) -> list[str]:
    """Problems that would leak information between splits (empty list = OK)."""
    p = []
    if idx["sample_id"].duplicated().any():
        p.append("a source video appears more than once")
    if not idx["split"].isin(SPLITS).all():
        p.append("invalid split value")
    for col, what in (("repo_path", "video path"), ("sha256", "SHA-256 duplicate group"),
                      ("split_group", "duplicate/augmentation split group")):
        n = int((idx.groupby(col)["split"].nunique() > 1).sum())
        if n:
            p.append(f"{n} {what}(s) in more than one split")
    # an augmented copy and its source (if present) must share a split
    by_name = idx.assign(filename=idx["repo_path"].str.split("/", n=1).str[-1]).set_index("filename")["split"]
    aug = idx[idx["aug_source"].fillna("").ne("")]
    bad = sum(1 for r in aug.itertuples() if r.aug_source in by_name.index and by_name[r.aug_source] != r.split)
    if bad:
        p.append(f"{bad} augmented file(s) in a different split from their source")
    return p
