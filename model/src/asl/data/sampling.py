"""Reproducible, class-balanced sample of videos for Representative Sample EDA.

Every class receives `per_class` videos. The remaining slots up to `target_total`
give one extra video to classes with unusual sizes: half to the smallest classes,
half to the largest. Files are drawn without replacement. Duplicate-content
groups are NOT collapsed (this sample is for video EDA, not leakage control);
membership is recorded instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_sample(pool: pd.DataFrame, *, seed: int, per_class: int, target_total: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pool: one row per eligible file (sample_id, word, ...). Returns (sample, per_class_table)."""
    rng = np.random.default_rng(seed)
    pool = pool.sort_values("sample_id", ignore_index=True)
    sizes = pool.groupby("word").size().sort_index()
    classes = sizes.index.to_numpy()

    quota = pd.Series(np.minimum(per_class, sizes.to_numpy()), index=classes)
    extra = max(0, target_total - int(quota.sum()))
    # Rank classes by size with a seeded random tie-break.
    order = pd.DataFrame({"n": sizes.to_numpy(), "tie": rng.random(len(classes))}, index=classes)
    by_size = order.sort_values(["n", "tie"]).index
    n_small = extra // 2
    small = list(by_size[:n_small])
    large = [c for c in by_size[::-1] if c not in set(small)][: extra - n_small]
    reason = pd.Series("base", index=classes, dtype=object)
    reason[small], reason[large] = "extra_smallest_class", "extra_largest_class"
    quota[small] += 1
    quota[large] += 1
    quota = np.minimum(quota, sizes)

    picked = []
    for word, rows in pool.groupby("word", sort=True).indices.items():
        choice = rng.choice(rows, size=int(quota[word]), replace=False)
        picked.extend(sorted(choice))
    sample = pool.iloc[sorted(picked)].reset_index(drop=True)
    assert sample["sample_id"].is_unique

    table = pd.DataFrame({
        "class_size": sizes,
        "sampled": sample.groupby("word").size().reindex(classes, fill_value=0),
        "selection": reason,
    })
    table.index.name = "word"
    return sample, table
