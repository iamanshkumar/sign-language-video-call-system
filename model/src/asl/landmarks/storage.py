"""Storage profiles: which landmark groups a store holds and in which dtype.

A profile never changes extraction or preprocessing - only `output.groups_to_store` and
`output.coord_dtype` - so a hands+pose store is a strict subset of a full store.
"""
from __future__ import annotations

import copy
import json

import pandas as pd


def apply_profile(cfg: dict, name: str | None) -> dict:
    """Step 2 config with `storage_profiles[name]` applied (a copy; `cfg` is not modified)."""
    out = copy.deepcopy(cfg)
    if not name:
        return out
    if name not in cfg.get("storage_profiles", {}):
        raise ValueError(f"unknown storage profile {name!r}; known: {sorted(cfg.get('storage_profiles', {}))}")
    prof = cfg["storage_profiles"][name]
    out["output"]["groups_to_store"] = list(prof["groups_to_store"])
    out["output"]["coord_dtype"] = dict(prof["coord_dtype"])
    out["configurations"] = available_configurations(cfg["configurations"], prof["groups_to_store"])
    out["storage_profile"] = name
    return out


def available_configurations(configurations: dict[str, list[str]], stored_groups) -> dict[str, list[str]]:
    """Feature configurations whose groups are all stored."""
    stored = set(stored_groups)
    return {k: v for k, v in configurations.items() if set(v) <= stored}


def store_groups(meta: pd.DataFrame) -> list[str]:
    """The stored groups of a landmark store (must be the same for every accepted video)."""
    acc = meta[meta["array_file"].map(lambda v: isinstance(v, str) and v != "")]
    groups = {tuple(json.loads(g)) for g in acc["stored_groups"]}
    if len(groups) != 1:
        raise ValueError(f"store mixes stored-group layouts: {groups}")
    return list(groups.pop())
