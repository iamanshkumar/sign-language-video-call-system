"""Deterministic label encoding from the frozen Step 1 vocabulary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def vocabulary_hash(vocab: list[str]) -> str:
    return hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()


def build_label_mapping(frozen_split: pd.DataFrame, split_sha256: str) -> dict:
    """Vocabulary = every word in the frozen split, sorted by Unicode code point.

    Built from the whole frozen split (never from a split subset or accepted samples),
    so train/val/test share one mapping and classes without samples keep an index.
    """
    vocab = sorted(set(frozen_split["word"]))
    return {"num_classes": len(vocab), "vocabulary_sha256": vocabulary_hash(vocab),
            "frozen_split_sha256": split_sha256, "ordering": "sorted by Unicode code point",
            "word_to_index": {w: i for i, w in enumerate(vocab)}}


def save_label_mapping(mapping: dict, path: Path) -> None:
    """Writes once; refuses to replace an existing, different mapping."""
    if path.exists():
        old = load_label_mapping(path)
        if old != mapping:
            raise FileExistsError(f"{path} exists with a different mapping; refusing to overwrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_label_mapping(path: Path) -> dict:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    vocab = sorted(mapping["word_to_index"], key=mapping["word_to_index"].get)
    if vocabulary_hash(vocab) != mapping["vocabulary_sha256"] or len(vocab) != mapping["num_classes"]:
        raise ValueError(f"{path}: label mapping failed its integrity check")
    if [mapping["word_to_index"][w] for w in vocab] != list(range(len(vocab))):
        raise ValueError(f"{path}: indices are not 0..N-1")
    return mapping


def encode(words: pd.Series, mapping: dict) -> pd.Series:
    """Map words to indices; raises on any unknown word."""
    idx = words.map(mapping["word_to_index"])
    if idx.isna().any():
        raise KeyError(f"unknown labels: {sorted(set(words[idx.isna()]))[:10]}")
    return idx.astype("int64")
