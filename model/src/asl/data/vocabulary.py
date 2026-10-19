"""Evidence-driven subset vocabulary from metadata only (frozen split, manifest, label audit).

A vocabulary filters the frozen split by class; it never re-splits. Every rule is a
column of `class_features`, so each exclusion has an explicit, recorded reason.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
_BATCH_RE = r"^(.*_\d{8}_\d{6})_\d+\.mp4$"


def add_file_columns(split: pd.DataFrame) -> pd.DataFrame:
    """Per-file evidence: cross-label sharing, within-class duplicates, augmentation, batch."""
    s = split.copy()
    fname = s["repo_path"].str.split("/", n=1).str[1]
    s["batch"] = fname.str.extract(_BATCH_RE)[0].where(s["source_proxy"] == "word_timestamp_n")
    s["cross_label"] = s.groupby("sha256")["word"].transform("nunique") > 1
    s["within_dup"] = s.groupby(["word", "sha256"])["sha256"].transform("size") > 1
    s["augmented"] = s["aug_source"].fillna("").ne("")
    return s


def class_features(s: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """One row per class. `units` = independent split units (SHA-256 group + aug family)."""
    g = s.groupby("word")
    f = pd.DataFrame({
        "clips": g.size(),
        "unique_clips": g["sha256"].nunique(),
        "units": g["split_group"].nunique(),
        "cross_label_clips": g["cross_label"].sum(),
        "within_dup_clips": g["within_dup"].sum(),
        "augmented_clips": g["augmented"].sum(),
        "source_patterns": g["source_proxy"].nunique(),
        "max_batch_clips": g["batch"].agg(lambda x: int(x.value_counts().iloc[0]) if x.notna().any() else 0),
        "numericid_clips": g["source_proxy"].agg(lambda x: int((x == "numericid-WORD").sum())),
    })
    for sp in SPLITS:
        part = s[s["split"] == sp].groupby("word")
        f[f"{sp}_clips"] = part.size()
        f[f"{sp}_units"] = part["split_group"].nunique()
    f = f.fillna(0).astype({c: int for c in f.columns})
    f["cross_label_fraction"] = f["cross_label_clips"] / f["clips"]
    f["augmented_fraction"] = f["augmented_clips"] / f["clips"]
    f["within_dup_fraction"] = 1 - f["unique_clips"] / f["clips"]
    f["max_batch_share"] = f["max_batch_clips"] / f["clips"]
    f["train_unit_fraction"] = f["train_units"] / f["units"]
    f["min_eval_units"] = f[["val_units", "test_units"]].min(axis=1)
    f["multiword"] = f.index.str.contains(r"[ _\-]", regex=True)
    anomalies = audit.groupby("label")["anomaly"].agg(lambda x: "|".join(sorted(set(x))))
    f["audit_anomalies"] = f.index.map(anomalies).fillna("")
    f.index.name = "word"
    return f


def exclusion_reasons(f: pd.DataFrame, rules: dict, heavy: float) -> pd.Series:
    """'|'-joined reasons per class ('' = passes every rule)."""
    r = rules
    checks = [
        ("augmentation_dominated", f["augmented_fraction"] >= 0.5),
        ("augmented_copies", (f["augmented_clips"] > r["max_augmented_clips"]) & (f["augmented_fraction"] < 0.5)),
        ("fully_shared_with_other_labels", f["cross_label_fraction"] >= 1.0),
        ("heavily_shared_with_other_labels", (f["cross_label_fraction"] >= heavy) & (f["cross_label_fraction"] < 1.0)),
        ("shares_clips_with_other_labels", (f["cross_label_clips"] > r["max_cross_label_shared_clips"])
         & (f["cross_label_fraction"] < heavy)),
        ("multiword_label", f["multiword"] if r["exclude_multiword_labels"] else False),
        ("label_audit_anomaly", f["audit_anomalies"].ne("") if r["exclude_label_audit_anomalies"] else False),
        ("too_few_val_units", f["val_units"] < r["min_val_units"]),
        ("too_few_test_units", f["test_units"] < r["min_test_units"]),
        ("imbalanced_split", ~f["train_unit_fraction"].between(*r["train_fraction_range"])),
        ("within_class_duplicates", f["within_dup_fraction"] > r["max_within_class_duplicate_fraction"]),
        ("single_batch_dominated", f["max_batch_share"] > r["max_single_batch_share"]),
        ("single_source_collection", f["source_patterns"] < r["min_source_patterns"]),
    ]
    out = pd.Series([[] for _ in range(len(f))], index=f.index)
    for name, mask in checks:
        mask = pd.Series(mask, index=f.index).astype(bool)
        for w in f.index[mask]:
            out[w].append(name)
    return out.map("|".join)


def tie_break(word: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{word}".encode()).hexdigest()


def rank_candidates(pool: pd.DataFrame, keys: list[str], seed: int) -> pd.DataFrame:
    p = pool.copy()
    p["_tb"] = [tie_break(w, seed) for w in p.index]
    p = p.sort_values(keys + ["_tb"], ascending=[False] * len(keys) + [True], kind="mergesort")
    p["rank"] = np.arange(1, len(p) + 1)
    return p.drop(columns="_tb")


def label_pairs(s: pd.DataFrame, clips: pd.Series) -> pd.DataFrame:
    """Every label pair sharing >= 1 byte-identical clip, with the share of each class."""
    cross = s[s["cross_label"]].groupby("sha256")["word"].agg(lambda x: sorted(set(x)))
    pair = Counter()
    for labels in cross:
        for a, b in itertools.combinations(labels, 2):
            pair[(a, b)] += 1
    rows = [{"label_a": a, "label_b": b, "shared_clips": k, "fraction_of_a": k / clips[a],
             "fraction_of_b": k / clips[b]} for (a, b), k in pair.items()]
    df = pd.DataFrame(rows)
    df["max_fraction"] = df[["fraction_of_a", "fraction_of_b"]].max(axis=1)
    df["min_fraction"] = df[["fraction_of_a", "fraction_of_b"]].min(axis=1)
    return df.sort_values(["shared_clips", "label_a", "label_b"], ascending=[False, True, True], ignore_index=True)


def components(pairs: pd.DataFrame) -> list[set[str]]:
    """Connected components of the label-sharing graph, largest first."""
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in zip(pairs["label_a"], pairs["label_b"]):
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    comps = []
    for w in sorted(adj):
        if w in seen:
            continue
        stack, comp = [w], set()
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x)
                stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(comp)
    return sorted(comps, key=lambda c: (-len(c), sorted(c)[0]))


def pair_treatment(row) -> tuple[str, str]:
    """(evidence class, recommended treatment) for a byte-identical sharing pair."""
    if row.min_fraction >= 0.95:
        return ("near-complete identity: >= 95% of BOTH classes are the same files",
                "merge if ever included (metadata gives no way to tell them apart); excluded from both vocabularies")
    if row.max_fraction >= 0.5:
        return ("one class is mostly copies of the other",
                "investigate manually (subset/superset synonym or relabelled copies); excluded")
    if row.shared_clips <= 2:
        return ("sporadic: 1-2 identical clips",
                "keep separate; the shared clip(s) are label noise for at least one label; excluded")
    return ("partial overlap", "investigate manually; excluded")


def summary_counts(f: pd.DataFrame, words: list[str]) -> dict:
    sub = f.loc[words]
    return {"classes": len(sub), "clips": int(sub["clips"].sum()), "unique_clips": int(sub["unique_clips"].sum()),
            "independent_units": int(sub["units"].sum()),
            **{f"{sp}_clips": int(sub[f"{sp}_clips"].sum()) for sp in SPLITS},
            **{f"{sp}_units": int(sub[f"{sp}_units"].sum()) for sp in SPLITS},
            "cross_label_shared_clips": int(sub["cross_label_clips"].sum()),
            "augmented_clips": int(sub["augmented_clips"].sum()),
            "within_class_duplicate_clips": int(sub["within_dup_clips"].sum()),
            "clips_per_class": {"min": int(sub["clips"].min()), "median": float(sub["clips"].median()),
                                "max": int(sub["clips"].max())},
            "class_imbalance_ratio": float(sub["clips"].max() / sub["clips"].min()),
            "min_val_units_per_class": int(sub["val_units"].min()),
            "min_test_units_per_class": int(sub["test_units"].min())}


# ---------------------------------------------------------------- using a vocabulary downstream
def load_vocabulary(path, frozen_split_sha256: str | None = None) -> dict:
    """Load a vocabulary JSON and check it is internally consistent (and built on this split)."""
    import json
    from pathlib import Path

    v = json.loads(Path(path).read_text(encoding="utf-8"))
    words = [c["class"] for c in v["classes"]]
    if words != sorted(words) or [c["index"] for c in v["classes"]] != list(range(len(words))):
        raise ValueError(f"{path}: classes must be sorted with index = position")
    if len(set(words)) != len(words) or len(words) != v["num_classes"]:
        raise ValueError(f"{path}: duplicate classes or wrong num_classes")
    if hashlib.sha256("\n".join(words).encode()).hexdigest() != v["vocabulary_sha256"]:
        raise ValueError(f"{path}: vocabulary_sha256 does not match the class list")
    if frozen_split_sha256 and v["frozen_split"]["sha256"] != frozen_split_sha256:
        raise ValueError(f"{path}: built on a different frozen split")
    return v


def vocabulary_words(v: dict) -> list[str]:
    return [c["class"] for c in v["classes"]]


def select_samples(frozen: pd.DataFrame, words: list[str], per_split: dict[str, int] | None = None,
                   n_classes: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Frozen-split rows of the vocabulary, optionally a deterministic subset.

    n_classes: take this many classes (ordered by sha256("<seed>:<word>")). per_split: take
    this many videos per class and split (ordered by sha256("<seed>:<sample_id>")). The split
    of every row is the frozen one; nothing is re-split.
    """
    rows = frozen[frozen["word"].isin(words)]
    if set(rows["word"]) != set(words):
        raise ValueError(f"vocabulary words missing from the split: {sorted(set(words) - set(rows['word']))[:5]}")
    if n_classes is not None:
        keep = sorted(words, key=lambda w: tie_break(w, seed))[:n_classes]
        rows = rows[rows["word"].isin(keep)]
    if per_split is not None:
        rows = rows.assign(_k=[tie_break(s, seed) for s in rows["sample_id"]]).sort_values("_k")
        rows = pd.concat([g.head(per_split.get(sp, 0)) for (w, sp), g in rows.groupby(["word", "split"])])
        rows = rows.drop(columns="_k")
    return rows.sort_values("sample_id", ignore_index=True)


def subset_label_mapping(frozen: pd.DataFrame, v: dict, split_sha256: str) -> dict:
    """Label mapping (same format as data/sequences/label_mapping.json) for a vocabulary;
    indices equal the vocabulary file's `index`."""
    from asl.sequences.labels import build_label_mapping

    words = vocabulary_words(v)
    m = build_label_mapping(frozen[frozen["word"].isin(words)], split_sha256)
    if [w for w, _ in sorted(m["word_to_index"].items(), key=lambda kv: kv[1])] != words:
        raise ValueError("subset mapping order differs from the vocabulary file")
    m["vocabulary"] = v["name"]
    return m
