"""Reconcile the metadata CSV with the actual repository files.

The CSV (`word`, `videos`) stores bare filenames without their `part_N/`
folder, so each row is resolved by filename against the repository index.
Nothing is silently fixed: every problem is recorded in the `issue` column and
the row is marked ineligible.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Issue codes (a row's first detected problem; "" means no problem).
DUPLICATE_CSV_ROW = "duplicate_csv_row"          # same (word, file) listed more than once; first kept
CONFLICTING_LABEL = "conflicting_label_for_file"  # same file listed under different words
FILE_MISSING = "file_missing_in_repo"             # CSV row whose file does not exist in the repo
AMBIGUOUS_FILENAME = "ambiguous_filename"         # filename exists in more than one folder
NOT_IN_CSV = "repo_file_not_in_csv"               # video file with no label
EMPTY_FIELD = "empty_word_or_filename"
MISSING_SHA = "missing_sha256"

# Filename conventions observed in the repo. They hint at the collection a
# video came from; they are NOT signer IDs.
_SOURCE_PATTERNS = [
    ("numericid-WORD", re.compile(r"^\d+-.+\.mp4$", re.I)),
    ("word_timestamp_n", re.compile(r"^.+_\d{8}_\d{6}_\d+\.mp4$", re.I)),
    ("word_video_n", re.compile(r"^.+_video_\d+\.mp4$", re.I)),
    ("word_n", re.compile(r"^.+_\d+\.mp4$", re.I)),
    ("word_only", re.compile(r"^[^_\d]+\.mp4$", re.I)),
]


def source_proxy(filename: str) -> str:
    for name, pattern in _SOURCE_PATTERNS:
        if pattern.match(filename):
            return name
    return "other"


def read_metadata_csv(path: Path) -> pd.DataFrame:
    """Read the CSV exactly as stored: every value is a string, no NA coercion.

    `keep_default_na=False` matters: labels such as "null", "none" or "nan"
    would otherwise become missing values.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(df.columns) != ["word", "videos"]:
        raise ValueError(f"unexpected CSV columns: {list(df.columns)}")
    df = df.rename(columns={"videos": "filename"})
    df.insert(0, "csv_row", range(len(df)))
    return df


def build_manifest(csv_df: pd.DataFrame, repo_df: pd.DataFrame) -> pd.DataFrame:
    """One row per CSV row plus one row per unlabeled repo file."""
    by_name = repo_df.groupby("filename")
    name_counts = by_name.size()
    first_file = by_name.first()

    m = csv_df.copy()
    m["issue"] = ""

    empty = (m["word"].str.strip() == "") | (m["filename"].str.strip() == "")
    m.loc[empty, "issue"] = EMPTY_FIELD

    labels_per_file = m.groupby("filename")["word"].transform("nunique")
    m.loc[(m["issue"] == "") & (labels_per_file > 1), "issue"] = CONFLICTING_LABEL

    dup = m.duplicated(subset=["word", "filename"], keep="first")
    m.loc[(m["issue"] == "") & dup, "issue"] = DUPLICATE_CSV_ROW

    counts = m["filename"].map(name_counts).fillna(0).astype(int)
    m.loc[(m["issue"] == "") & (counts == 0), "issue"] = FILE_MISSING
    m.loc[(m["issue"] == "") & (counts > 1), "issue"] = AMBIGUOUS_FILENAME

    for col in ("repo_path", "folder", "size_bytes", "sha256"):
        m[col] = m["filename"].map(first_file[col])
    m.loc[counts != 1, ["repo_path", "folder", "size_bytes", "sha256"]] = None
    m.loc[(m["issue"] == "") & m["sha256"].isna(), "issue"] = MISSING_SHA

    orphans = repo_df[~repo_df["filename"].isin(set(csv_df["filename"]))].copy()
    orphans["csv_row"] = -1
    orphans["word"] = None
    orphans["issue"] = NOT_IN_CSV

    m = pd.concat([m, orphans[m.columns]], ignore_index=True)
    m["source_proxy"] = m["filename"].map(source_proxy)
    m["eligible_pre_probe"] = m["issue"] == ""
    # A sample is a unique repository file; repo_path is unique for eligible rows.
    m["sample_id"] = m["repo_path"].where(m["eligible_pre_probe"])
    m["size_bytes"] = m["size_bytes"].astype("Int64")
    return m


def duplicate_content_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Groups of eligible samples whose bytes are identical (same LFS SHA-256)."""
    elig = manifest[manifest["eligible_pre_probe"]]
    g = elig.groupby("sha256").agg(
        n_files=("sample_id", "size"),
        n_labels=("word", "nunique"),
        labels=("word", lambda s: "|".join(sorted(set(s)))),
        paths=("repo_path", lambda s: "|".join(sorted(s))),
    )
    return g[g["n_files"] > 1].sort_values(["n_labels", "n_files"], ascending=False).reset_index()


def annotate_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add dup_group_size / dup_cross_label columns (eligible rows only)."""
    m = manifest.copy()
    elig = m["eligible_pre_probe"]
    size = m[elig].groupby("sha256")["sample_id"].transform("size")
    labels = m[elig].groupby("sha256")["word"].transform("nunique")
    m["dup_group_size"] = 0
    m.loc[elig, "dup_group_size"] = size
    m["dup_cross_label"] = False
    m.loc[elig, "dup_cross_label"] = (labels > 1) & (size > 1)
    return m
