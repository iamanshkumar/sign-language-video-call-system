"""Consistency checks between saved landmark arrays (format v2), their metadata and the frozen split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from asl.landmarks.layout import GROUP_ORDER, GROUPS, PRESENT, INTERPOLATED, assemble_features, feature_dim


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _has_file(value) -> bool:
    """Missing values may be None or NaN depending on the pandas dtype."""
    return isinstance(value, str) and value != ""


def unreferenced_arrays(meta: pd.DataFrame, arrays_dir: Path) -> list[str]:
    """Array files with no metadata record (e.g. written by a batch that was interrupted)."""
    referenced = {f for f in meta.get("array_file", pd.Series(dtype=object)) if _has_file(f)}
    return sorted(p.name for p in arrays_dir.glob("*.npz") if p.name not in referenced)


def check_arrays(a: dict, t: int, stored_groups: list[str], coord_dtype) -> list[str]:
    """Problems with one v2 array set of T frames. coord_dtype: str or {group: str}."""
    p: list[str] = []
    for g in stored_groups:
        want = np.dtype(coord_dtype[g] if isinstance(coord_dtype, dict) else coord_dtype)
        if a[g].shape != (t, GROUPS[g], 3):
            p.append(f"{g} shape {a[g].shape}")
        elif a[g].dtype != want:
            p.append(f"{g} dtype {a[g].dtype}")
    for k, shape in {"pose_vis": (t, 33), "flags": (t, 4), "center": (t, 3), "scale": (t,),
                     "shoulder_measured": (t,), "active_interval": (2,)}.items():
        if a[k].shape != shape:
            p.append(f"{k} shape {a[k].shape} != {shape}")
    if p:
        return p
    if a["flags"].dtype != np.uint8:
        p.append("flags not uint8")
    if not all(np.isfinite(a[k].astype(np.float32)).all() for k in stored_groups + ["pose_vis", "center", "scale"]):
        p.append("non-finite values")
    for i, g in enumerate(GROUP_ORDER):
        if g not in stored_groups:
            continue
        absent = (a["flags"][:, i] & PRESENT) == 0
        if np.abs(a[g][absent].astype(np.float32)).sum() != 0:
            p.append(f"{g} non-zero coords where not present")
        if ((a["flags"][:, i] & INTERPOLATED) != 0)[absent].any():
            p.append(f"{g} interpolated frame marked not present")
    s, e = (int(x) for x in a["active_interval"])
    if not ((s == e == -1) or (0 <= s <= e < t)):
        p.append(f"active_interval {s},{e} outside [0,{t})")
    return p


def check_outputs(meta: pd.DataFrame, arrays_dir: Path, configurations: dict[str, list[str]]) -> list[str]:
    """Return a list of human-readable problems (empty = consistent)."""
    problems: list[str] = []
    for r in meta.itertuples(index=False):
        sid = r.sample_id
        kept = bool(getattr(r, "array_kept_for_rejected", False))
        if r.rejected is None or pd.isna(r.rejected):
            continue  # download failure; retried later
        if bool(r.rejected):
            if not r.rejection_reason:
                problems.append(f"{sid}: rejected without reason")
            if _has_file(r.array_file) and not kept:
                problems.append(f"{sid}: rejected but has array file")
            if not kept:
                continue
        elif isinstance(r.rejection_reason, str) and r.rejection_reason:
            problems.append(f"{sid}: accepted but has rejection reason")
        if not _has_file(r.array_file) or not (arrays_dir / r.array_file).exists():
            problems.append(f"{sid}: missing array file")
            continue
        a = load_arrays(arrays_dir / r.array_file)
        stored = json.loads(r.stored_groups)
        problems += [f"{sid}: {x}" for x in check_arrays(a, int(r.std_frame_count), stored,
                                                              json.loads(r.coord_dtype))]
        s, e = (int(x) for x in a["active_interval"])
        if pd.notna(r.active_start_frame) and (s, e) != (int(r.active_start_frame), int(r.active_end_frame)):
            problems.append(f"{sid}: active interval array {s},{e} != metadata")
        dims = json.loads(r.configurations)
        for name, groups in configurations.items():
            if not set(groups) <= set(stored):
                continue
            f = assemble_features(a, groups)
            if f.shape != (int(r.std_frame_count), feature_dim(groups)) or dims.get(name) != f.shape[1]:
                problems.append(f"{sid}: {name} feature shape {f.shape} vs metadata {dims.get(name)}")
    return problems


def check_split_preserved(meta: pd.DataFrame, frozen: pd.DataFrame) -> list[str]:
    """Every processed sample exists in the frozen split with the same split, word and path."""
    m = meta.merge(frozen[["sample_id", "split", "word", "repo_path"]], on="sample_id", how="left",
                   suffixes=("", "_frozen"), indicator=True)
    problems = [f"{r.sample_id}: not in frozen split" for r in m[m["_merge"] != "both"].itertuples()]
    both = m[m["_merge"] == "both"]
    for col in ("split", "word", "repo_path"):
        bad = both[both[col] != both[f"{col}_frozen"]]
        problems += [f"{r.sample_id}: {col} differs from frozen split" for r in bad.itertuples()]
    return problems
