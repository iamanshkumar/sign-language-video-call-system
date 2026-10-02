"""Leakage and integrity checks for a split. Each check returns (name, passed, detail)."""
from __future__ import annotations

import pandas as pd

from asl.data.split import SPLITS

Check = tuple[str, bool, str]


def run_checks(
    split_df: pd.DataFrame,
    eligible_ids: set[str],
    ratios: dict[str, float],
    tolerance: float = 0.01,
    expected: set[str] | None = None,
) -> list[Check]:
    """`expected`: classes that must appear in every split (default: all classes).

    Classes outside `expected` are those that cannot be in all splits because they
    have fewer than 3 indivisible content groups; they are reported, not failed.
    """
    checks: list[Check] = []
    ids = split_df["sample_id"]

    dup_ids = int(ids.duplicated().sum())
    checks.append(("unique sample_id", dup_ids == 0, f"{dup_ids} duplicated ids"))

    dup_paths = int(split_df["repo_path"].duplicated().sum())
    checks.append(("each video file in exactly one split", dup_paths == 0, f"{dup_paths} paths repeated"))

    valid_split = split_df["split"].isin(SPLITS).all()
    checks.append(("split values valid", bool(valid_split), f"values: {sorted(split_df['split'].unique())}"))

    per_hash = split_df.groupby("sha256")["split"].nunique()
    crossing = int((per_hash > 1).sum())
    checks.append(
        ("no identical content (SHA-256) across splits", crossing == 0, f"{crossing} hashes span >1 split")
    )

    if "split_group" in split_df.columns:
        per_group = split_df.groupby("split_group")["split"].nunique()
        crossing_g = int((per_group > 1).sum())
        checks.append(("no split group (duplicates + augmentation family) across splits", crossing_g == 0,
                       f"{crossing_g} groups span >1 split"))

    got = set(ids)
    extra, missing = got - eligible_ids, eligible_ids - got
    checks.append(("only eligible samples in split", not extra, f"{len(extra)} ineligible samples present"))
    checks.append(("all eligible samples assigned", not missing, f"{len(missing)} eligible samples missing"))

    expected = set(split_df["word"].unique()) if expected is None else expected
    for s in SPLITS:
        present = set(split_df.loc[split_df["split"] == s, "word"])
        miss = sorted(expected - present)
        checks.append((
            f"every expected class present in {s}", not miss,
            f"{len(expected & present)}/{len(expected)} expected classes"
            + (f"; missing: {miss[:10]}" if miss else ""),
        ))

    n = len(split_df)
    for s in SPLITS:
        frac = float((split_df["split"] == s).sum()) / n
        ok = abs(frac - ratios[s]) <= tolerance
        checks.append((f"{s} ratio ~ {ratios[s]:.2f}", ok, f"{frac:.4f}"))
    return checks


def split_class_ratio_spread(split_df: pd.DataFrame) -> pd.DataFrame:
    """Per-class fraction in each split (for reporting stratification quality)."""
    ct = pd.crosstab(split_df["word"], split_df["split"]).reindex(columns=list(SPLITS), fill_value=0)
    return ct.div(ct.sum(axis=1), axis=0)
