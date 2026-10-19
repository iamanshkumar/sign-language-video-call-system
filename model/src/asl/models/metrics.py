"""Classification metrics over the full vocabulary (zero_division=0).

Macro averages are over the classes occurring in y_true or y_pred (sklearn default when
`labels` is the union); classes absent from both are not evaluated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import precision_recall_fscore_support


def _union(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.union1d(np.unique(y), np.unique(p))


def classification_metrics(y_true, y_pred, vocabulary_size: int) -> dict:
    y, p = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    labels = _union(y, p)
    correct = int((y == p).sum())
    out = {"samples": int(len(y)), "correct": correct, "incorrect": int(len(y) - correct),
           "accuracy": correct / len(y) if len(y) else float("nan"), "vocabulary_size": int(vocabulary_size),
           "classes_in_y_true": int(len(np.unique(y))), "classes_evaluated": int(len(labels))}
    for avg in ("macro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average=avg, zero_division=0)
        out.update({f"{avg}_precision": float(pr), f"{avg}_recall": float(rc), f"{avg}_f1": float(f1)})
    return out


def summary(m: dict) -> dict:
    """Short form stored in metadata.json `evaluations`."""
    return {k: m[k] for k in ("samples", "accuracy", "macro_f1", "weighted_f1", "classes_in_y_true")}


def per_class_table(y_true, y_pred, words: list[str]) -> pd.DataFrame:
    y, p = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    labels = _union(y, p)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, average=None, zero_division=0)
    return pd.DataFrame({"label": labels, "word": [words[i] for i in labels], "precision": pr, "recall": rc,
                         "f1": f1, "support": sup})


def confusion(y_true, y_pred, vocabulary_size: int) -> sparse.coo_matrix:
    """Full vocabulary x vocabulary confusion matrix (rows = true, cols = predicted), sparse."""
    y, p = np.asarray(y_true, dtype=np.int32), np.asarray(y_pred, dtype=np.int32)
    cm = sparse.coo_matrix((np.ones(len(y), dtype=np.int32), (y, p)), shape=(vocabulary_size, vocabulary_size))
    cm.sum_duplicates()
    return cm


def top_confusions(cm: sparse.coo_matrix, words: list[str], n: int = 50) -> pd.DataFrame:
    off = cm.row != cm.col
    df = pd.DataFrame({"true": cm.row[off], "pred": cm.col[off], "count": cm.data[off]})
    df = df.sort_values(["count", "true", "pred"], ascending=[False, True, True], ignore_index=True).head(n)
    df["true_word"] = [words[i] for i in df["true"]]
    df["pred_word"] = [words[i] for i in df["pred"]]
    return df
