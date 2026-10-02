"""Reproducible, frozen, word-stratified 70/15/15 split.

The indivisible unit is a "split group" (see asl.data.groups): byte-identical
videos (same SHA-256, even under different synonym labels) plus augmented copies
and their source video. A unit always lands in one split, so neither exact
duplicates nor augmented copies can leak between train and test. The split
is written once together with a lock file; `load_frozen_split` verifies the
file's checksum so later stages cannot silently use a modified split.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = [
    "sample_id", "repo_path", "word", "sha256", "dup_group_size",
    "split_group", "split_group_size", "aug_source", "source_proxy", "split",
]


def _class_targets(n_c: np.ndarray, ratios: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Integer per-class targets (C x 3) by largest remainder; ties broken randomly (seeded).

    Every split gets at least one sample of a class when the class has >= 3 samples.
    """
    raw = n_c[:, None] * ratios[None, :]
    t = np.floor(raw).astype(int)
    frac = raw - t + rng.random(raw.shape) * 1e-9  # seeded tie-break
    for c in range(len(n_c)):
        for s in np.argsort(-frac[c])[: n_c[c] - t[c].sum()]:
            t[c, s] += 1
        if n_c[c] >= len(ratios):
            for s in range(1, len(ratios)):
                if t[c, s] == 0:
                    t[c, s], t[c, 0] = 1, t[c, 0] - 1
    return t


def assign_splits(
    samples: pd.DataFrame, *, seed: int, val: float, test: float, group_col: str = "split_group"
) -> pd.DataFrame:
    """samples: sample_id, word, sha256, `group_col` (+ others). Returns a copy with `split`.

    If `group_col` is missing, the SHA-256 group is the unit.

    Group-aware stratification by word:
      * a unit is one `group_col` value (default: split group); units are indivisible;
      * per-class targets follow the 70/15/15 ratios (largest remainder);
      * multi-file units are placed first (largest first), each into the split whose
        per-class quotas for *all* labels in the unit are most under-filled;
      * single-file units then fill each class's remaining quotas.
    Deterministic for a given input set, seed and ratios (input order does not matter).
    """
    s = samples.sort_values("sample_id", ignore_index=True)
    if group_col not in s.columns:
        s[group_col] = s["sha256"]
    rng = np.random.default_rng(seed)
    ratios = np.array([1.0 - val - test, val, test])
    classes = sorted(s["word"].unique())
    cid = s["word"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    n_c = np.bincount(cid, minlength=len(classes))
    target = _class_targets(n_c, ratios, rng)
    cur = np.zeros_like(target)

    def score(label_ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
        deficit = (target[label_ids] - cur[label_ids]) / np.maximum(target[label_ids], 1)
        return (counts[:, None] * deficit).sum(axis=0)

    unit_of = s.groupby(group_col, sort=True).indices  # unit -> row positions
    shas = np.array(sorted(unit_of))
    sizes = np.array([len(unit_of[h]) for h in shas])
    split_of_row = np.full(len(s), -1)

    # 1) indivisible multi-file groups, largest first, seeded order within equal size
    multi = shas[sizes > 1]
    perm = rng.permutation(len(multi))
    multi = multi[perm][np.argsort(-sizes[sizes > 1][perm], kind="stable")]
    for h in multi:
        rows = unit_of[h]
        labels, counts = np.unique(cid[rows], return_counts=True)
        sc = score(labels, counts)
        k = int(np.flatnonzero(sc == sc.max())[0])
        split_of_row[rows] = k
        cur[labels, k] += counts

    # 2) single-file units, per class (seeded order), filling remaining quotas
    rows_by_class: dict[int, list[int]] = defaultdict(list)
    for h in shas[sizes == 1]:
        r = int(unit_of[h][0])
        rows_by_class[int(cid[r])].append(r)
    for c in range(len(classes)):
        for r in rng.permutation(np.array(rows_by_class[c], dtype=int)):
            deficit = (target[c] - cur[c]) / np.maximum(target[c], 1)
            k = int(np.flatnonzero(deficit == deficit.max())[0])
            split_of_row[r] = k
            cur[c, k] += 1

    assert (split_of_row >= 0).all()
    s["split"] = np.array(SPLITS)[split_of_row]
    s["dup_group_size"] = s.groupby("sha256")["sample_id"].transform("size").astype(int)
    s["split_group_size"] = s[group_col].map(dict(zip(shas, sizes))).astype(int)
    if group_col != "split_group":
        s["split_group"] = s[group_col]
    if "aug_source" not in s.columns:
        s["aug_source"] = None
    return s


def expected_classes(samples: pd.DataFrame, group_col: str = "split_group") -> set[str]:
    """Classes that *can* appear in all three splits: at least 3 distinct indivisible units."""
    col = group_col if group_col in samples.columns else "sha256"
    n_units = samples.groupby("word")[col].nunique()
    return set(n_units[n_units >= len(SPLITS)].index)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen_split(split_df: pd.DataFrame, out_dir: Path, version: str, meta: dict) -> Path:
    """Write split CSV + per-class counts CSV + lock JSON (checksums of both), all read-only."""
    csv_path = out_dir / f"split_{version}.csv"
    counts_path = out_dir / f"split_{version}_class_counts.csv"
    lock_path = out_dir / f"split_{version}.lock.json"
    if any(p.exists() for p in (csv_path, lock_path, counts_path)) or list(out_dir.glob(f"split_{version}_*.csv")):
        raise FileExistsError(
            f"{csv_path.name} already exists. The split is frozen; it must not be regenerated. "
            "Create a new version explicitly if a new split is really required."
        )
    out = split_df[SPLIT_COLUMNS].sort_values("sample_id")
    out.to_csv(csv_path, index=False, lineterminator="\n")
    class_counts = pd.crosstab(out["word"], out["split"]).reindex(columns=list(SPLITS), fill_value=0)
    class_counts.to_csv(counts_path, lineterminator="\n")
    per_split = {}
    for k in SPLITS:
        pth = out_dir / f"split_{version}_{k}.csv"
        out[out["split"] == k].to_csv(pth, index=False, lineterminator="\n")
        per_split[k] = {"file": pth.name, "sha256": _sha256_file(pth)}

    groups = out[out["dup_group_size"] > 1].groupby("sha256")
    lock = {
        "version": version,
        "file": csv_path.name,
        "sha256": _sha256_file(csv_path),
        "class_counts_file": counts_path.name,
        "class_counts_sha256": _sha256_file(counts_path),
        "per_split_files": per_split,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": int(len(out)),
        "n_classes": int(out["word"].nunique()),
        "counts": {k: int(v) for k, v in out["split"].value_counts().reindex(SPLITS).items()},
        "fractions": {k: round(float(v), 6) for k, v in out["split"].value_counts(normalize=True).reindex(SPLITS).items()},
        "classes_per_split": {k: int(out.loc[out["split"] == k, "word"].nunique()) for k in SPLITS},
        "duplicate_groups": {
            "unit_definition": "SHA-256 of file content (from HF LFS metadata, verified on download)",
            "n_content_units": int(out["sha256"].nunique()),
            "n_multi_file_groups": int(groups.ngroups),
            "n_files_in_multi_file_groups": int(groups.size().sum()) if groups.ngroups else 0,
            "n_cross_label_groups": int((groups["word"].nunique() > 1).sum()) if groups.ngroups else 0,
            "multi_file_groups_per_split": {
                k: int(v) for k, v in groups["split"].first().value_counts().reindex(SPLITS, fill_value=0).items()
            } if groups.ngroups else {},
        },
        "split_groups": {
            "unit_definition": "union of SHA-256 groups and augmentation families (source + *_aug_N copies)",
            "n_units": int(out["split_group"].nunique()),
            "n_multi_file_units": int((out.groupby("split_group").size() > 1).sum()),
            "largest_unit_files": int(out["split_group_size"].max()),
            "augmented_files": int(out["aug_source"].fillna("").ne("").sum()),
            "units_per_split": {k: int(out.loc[out["split"] == k, "split_group"].nunique()) for k in SPLITS},
        },
        **meta,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    for p in (csv_path, counts_path, lock_path, *(out_dir / f["file"] for f in per_split.values())):
        p.chmod(0o444)  # read-only as an extra guard against accidental edits
    return csv_path


def load_frozen_split(splits_dir: Path, version: str = "v1") -> pd.DataFrame:
    """The only supported way for later stages to read the split."""
    csv_path = splits_dir / f"split_{version}.csv"
    lock = json.loads((splits_dir / f"split_{version}.lock.json").read_text())
    actual = _sha256_file(csv_path)
    if actual != lock["sha256"]:
        raise RuntimeError(f"{csv_path} checksum {actual} != frozen {lock['sha256']}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["dup_group_size"] = df["dup_group_size"].astype(int)
    df["split_group_size"] = df["split_group_size"].astype(int)
    return df
