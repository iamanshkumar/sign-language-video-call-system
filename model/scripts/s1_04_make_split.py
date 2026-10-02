"""Step 1.4: create and freeze the group-aware stratified 70/15/15 split (runs once).

Built from the COMPLETE metadata manifest (all reconciled files); videos are not
decoded for this. Video-level rejections (undecodable, < 30 frames, landmark
rules) happen in Step 2 and are recorded as flags inside the frozen split.
Refuses to run if a split already exists.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from asl.config import PROJECT_ROOT, load_config
from asl.data.groups import build_split_groups
from asl.data.split import SPLITS, assign_splits, expected_classes, write_frozen_split


def main() -> None:
    cfg = load_config()
    meta, sp = cfg.path("metadata"), cfg["split"]
    mpath = meta / "manifest.parquet"
    m = pd.read_parquet(mpath)
    elig = build_split_groups(m[m["eligible_pre_probe"]])

    # Known-bad videos seen by the sample probe; recorded, NOT silently removed.
    known_bad = []
    sv = PROJECT_ROOT / cfg["eda_sample"]["dir"] / "sample_validated.parquet"
    if sv.exists():
        s = pd.read_parquet(sv)
        known_bad = sorted(s.loc[s["exclusion_reason"].isin(["undecodable", "fewer_than_min_decoded_frames"]),
                                 "repo_path"].tolist())

    units = elig.groupby("sha256")["word"].agg(["size", "nunique"])
    sg = elig.groupby("split_group")["word"].agg(["size", "nunique"])
    expected = expected_classes(elig)
    infeasible = sorted(set(elig["word"]) - expected)
    per_class = elig.groupby("word").size()
    print("=== data entering the split (complete metadata manifest) ===")
    print(f"samples:                 {len(elig):,}")
    print(f"classes:                 {elig['word'].nunique():,}")
    print(f"samples per class:       min {per_class.min()}, median {per_class.median():.0f}, max {per_class.max()}")
    print(f"content units (SHA-256): {len(units):,}")
    print(f"multi-file groups:       {(units['size'] > 1).sum():,} ({units.loc[units['size'] > 1, 'size'].sum():,} files)")
    print(f"cross-label groups:      {((units['size'] > 1) & (units['nunique'] > 1)).sum():,}")
    print(f"largest SHA-256 group:   {units['size'].max()} files")
    print(f"augmented files:         {elig['aug_source'].notna().sum():,} "
          f"(source present for {elig.loc[elig['aug_source'].notna(), 'aug_source'].isin(set(elig['filename'])).sum():,})")
    print(f"split units (SHA ∪ aug): {len(sg):,}; multi-file units {(sg['size'] > 1).sum():,}; "
          f"largest {sg['size'].max()} files; multi-label units {(sg['nunique'] > 1).sum():,}")
    print(f"classes that cannot appear in all 3 splits (<3 content units): {len(infeasible)} {infeasible[:20]}")
    print(f"excluded from manifest:  {m.loc[~m['eligible_pre_probe'], 'issue'].value_counts().to_dict()}")
    print(f"known bad videos (sample probe, kept, flagged for Step 2): {len(known_bad)}")

    split_df = assign_splits(elig, seed=sp["seed"], val=sp["val"], test=sp["test"])
    meta_info = {
        "seed": sp["seed"],
        "ratios": {"train": sp["train"], "val": sp["val"], "test": sp["test"]},
        "stratified_by": "word",
        "grouping": ("indivisible split unit = SHA-256 duplicate group joined (transitively) with augmentation "
                     "families: <source>.mp4 and every <source>_<word>_aug_<n>.mp4 (asl.data.groups)"),
        "grouping_decision": "augmentation families grouped with their source by user decision, 2026-09-28",
        "algorithm": "asl.data.split.assign_splits: group-aware greedy stratification by word",
        "classes_infeasible_for_all_splits": infeasible,
        "dataset_repo": cfg["dataset"]["repo_id"],
        "dataset_revision": cfg["dataset"]["revision"],
        "source": "complete metadata manifest (eligible_pre_probe)",
        "manifest_excluded_by_issue": m.loc[~m["eligible_pre_probe"], "issue"].value_counts().to_dict(),
        "manifest_sha256": hashlib.sha256(mpath.read_bytes()).hexdigest(),
        "known_bad_videos_from_sample_probe": known_bad,
        "note": "Video-level and landmark rejections (Step 2+) are recorded as flags; they never change this split.",
    }
    out = write_frozen_split(split_df, cfg.path("splits"), sp["version"], meta_info)
    print(f"wrote {out}")
    counts = split_df["split"].value_counts().reindex(SPLITS)
    for k in SPLITS:
        print(f"{k:5s} {counts[k]:7,d}  {100 * counts[k] / len(split_df):6.2f}%")


if __name__ == "__main__":
    main()
