"""Indivisible split units ("split groups").

Two files must share a split when either
  * they are byte-identical (same SHA-256), or
  * one is an augmented copy of the other: `<source>_<word>_aug_<n>.mp4` is
    derived from `<source>.mp4`. Augmented copies of the same source are linked
    even when the source file itself is absent from the dataset.
Links are merged transitively (union-find). No files are removed and no labels
are changed.
"""
from __future__ import annotations

import re

import pandas as pd

AUG_PATTERN = re.compile(r"_aug_\d+\.mp4$", re.I)


def augmentation_source(filename: str, word: str) -> str | None:
    """Filename of the source video for an augmented file, else None."""
    if not AUG_PATTERN.search(filename):
        return None
    stem = AUG_PATTERN.sub("", filename)
    return re.sub("_" + re.escape(word) + "$", "", stem, flags=re.I) + ".mp4"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # deterministic root


def build_split_groups(df: pd.DataFrame) -> pd.DataFrame:
    """df: sample_id, filename, word, sha256. Returns a copy with split_group, split_group_size,
    aug_source (filename of the source for augmented files) and aug_linked (bool)."""
    out = df.copy()
    uf = _UnionFind()
    by_name = dict(zip(out["filename"], out["sample_id"]))
    for sid, sha in zip(out["sample_id"], out["sha256"]):
        uf.union("sample:" + sid, "sha:" + sha)
    sources = [augmentation_source(f, w) for f, w in zip(out["filename"], out["word"])]
    for sid, src in zip(out["sample_id"], sources):
        if src is None:
            continue
        uf.union("sample:" + sid, "augsrc:" + src)
        if src in by_name:
            uf.union("sample:" + by_name[src], "augsrc:" + src)
    out["aug_source"] = sources
    roots = [uf.find("sample:" + sid) for sid in out["sample_id"]]
    # Name each group by its smallest sample_id so ids are stable and readable.
    first = pd.Series(out["sample_id"].to_numpy(), index=roots).groupby(level=0).min()
    out["split_group"] = [first[r] for r in roots]
    out["split_group_size"] = out.groupby("split_group")["sample_id"].transform("size").astype(int)
    linked = set(out.loc[out["aug_source"].notna(), "split_group"])
    out["aug_linked"] = out["split_group"].isin(linked)
    return out
