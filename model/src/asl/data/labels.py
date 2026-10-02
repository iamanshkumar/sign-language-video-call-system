"""Label audit. Reports anomalies only; labels are never changed or merged."""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

_SEP = re.compile(r"[\s_\-]+")


def _key(word: str) -> str:
    """Normalization key used only to *detect* near-duplicate labels."""
    return _SEP.sub("", word.strip().lower())


def audit_labels(words: pd.Series) -> pd.DataFrame:
    """Return one row per (label, anomaly) with a human-readable detail."""
    counts = words.value_counts()
    labels = sorted(counts.index)
    label_set = set(labels)
    findings: list[dict] = []

    def add(label: str, kind: str, detail: str) -> None:
        findings.append({"label": label, "n_samples": int(counts[label]), "anomaly": kind, "detail": detail})

    by_key: dict[str, list[str]] = defaultdict(list)
    for w in labels:
        by_key[_key(w)].append(w)
        if w != w.strip() or "  " in w:
            add(w, "whitespace", "leading/trailing/double whitespace")
        if w != w.lower():
            add(w, "uppercase", "contains uppercase characters")
        if re.search(r"[^a-z0-9 \-_']", w.lower()):
            add(w, "unusual_characters", repr(re.findall(r"[^a-z0-9 \-_']", w.lower())))
        if re.search(r"\d", w):
            add(w, "contains_digit", "")
        if re.search(r"-cl$|\bcl\b", w.lower()):
            add(w, "classifier_suffix", "ends with '-cl' (likely ASL classifier gloss)")
        k = _key(w)
        half = len(k) // 2
        if len(k) >= 4 and len(k) % 2 == 0 and k[:half] == k[half:]:
            add(w, "repeated_word", f"'{k[:half]}' repeated")
        if len(k) >= 14 and " " not in w and "-" not in w and "_" not in w:
            add(w, "long_unseparated", f"{len(k)} chars without separators")

    for k, group in by_key.items():
        if len(group) > 1:
            for w in group:
                add(w, "near_duplicate_label", "same after removing case/space/-/_: " + " | ".join(group))

    # Labels that are an exact concatenation of two other labels (e.g. "thankyou").
    keys = {_key(w): w for w in labels}
    for w in labels:
        k = _key(w)
        for i in range(2, len(k) - 1):
            a, b = k[:i], k[i:]
            if a in keys and b in keys and keys[a] != w and keys[b] != w:
                add(w, "concatenation_of_labels", f"{keys[a]} + {keys[b]}")
                break

    if not findings:
        return pd.DataFrame(columns=["label", "n_samples", "anomaly", "detail"])
    assert set(f["label"] for f in findings) <= label_set
    return pd.DataFrame(findings).sort_values(["anomaly", "label"], ignore_index=True)
