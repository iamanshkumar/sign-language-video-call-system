"""Reproducible selection of the Step 2 benchmark videos from the frozen split.

Guaranteed-coverage quotas are filled first (short clips, non-640x480, augmented
files, duplicate-group members); the remainder is stratified over decoded-frame
count quintiles, preferring one video per class. Video properties are known for
videos probed in the Step 1 EDA sample, so the stratified part draws from those.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_benchmark(split: pd.DataFrame, probed: pd.DataFrame, *, seed: int, n: int, quotas: dict,
                     which: str = "train") -> pd.DataFrame:
    """split: frozen split rows; probed: sample_id, decoded_frames, width, height (Step 1 EDA sample)."""
    rng = np.random.default_rng(seed)
    pool = split[split["split"] == which].sort_values("sample_id", ignore_index=True)
    known = pool.merge(probed[["sample_id", "decoded_frames", "width", "height"]], on="sample_id", how="inner")
    known["resolution"] = known["width"].astype(int).astype(str) + "x" + known["height"].astype(int).astype(str)
    chosen: list[pd.DataFrame] = []
    taken: set[str] = set()

    def take(cands: pd.DataFrame, k: int, reason: str) -> None:
        cands = cands[~cands["sample_id"].isin(taken)]
        if k <= 0 or cands.empty:
            return
        pick = cands.iloc[rng.permutation(len(cands))[: min(k, len(cands))]]
        taken.update(pick["sample_id"])
        chosen.append(pick.assign(benchmark_reason=reason))

    is_aug = pool["aug_source"].fillna("").ne("")
    take(known[known["decoded_frames"] < 30], quotas["short_clips"], "short_clip")
    take(known[known["resolution"] != "640x480"], quotas["non_default_resolution"], "non_default_resolution")
    take(pool[is_aug], quotas["augmented"], "augmented")
    take(pool[(pool["dup_group_size"].astype(int) > 1) & ~is_aug], quotas["duplicate_groups"], "duplicate_group")

    rest = n - sum(len(c) for c in chosen)
    rem = known[~known["sample_id"].isin(taken)].copy()
    rem["bin"] = pd.qcut(rem["decoded_frames"].rank(method="first"), 5, labels=False)
    per_bin = [rest // 5 + (1 if b < rest % 5 else 0) for b in range(5)]
    for b in range(5):
        cands = rem[rem["bin"] == b]
        cands = cands.iloc[rng.permutation(len(cands))]
        cands = pd.concat([cands.drop_duplicates("word"), cands[cands.duplicated("word")]])  # distinct classes first
        cands = cands[~cands["sample_id"].isin(taken)].head(per_bin[b])
        taken.update(cands["sample_id"])
        chosen.append(cands.drop(columns="bin").assign(benchmark_reason=f"frame_count_quintile_{b + 1}"))

    out = pd.concat(chosen, ignore_index=True)
    assert out["sample_id"].is_unique and (out["split"] == which).all()
    return out.sort_values("sample_id", ignore_index=True)
