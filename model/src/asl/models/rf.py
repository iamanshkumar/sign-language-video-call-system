"""Random Forest baseline on temporal statistics (mean | std | min | max over the W frames)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def temporal_statistics(x: np.ndarray) -> np.ndarray:
    """[W, F] -> [4F] or [B, W, F] -> [B, 4F]. Nothing is fitted (no leakage possible)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim not in (2, 3):
        raise ValueError(f"expected [W, F] or [B, W, F], got {x.shape}")
    axis = x.ndim - 2
    return np.concatenate([x.mean(axis), x.std(axis), x.min(axis), x.max(axis)], axis=-1)


def build_rf(rf_cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=rf_cfg["n_estimators"], max_depth=rf_cfg["max_depth"],
                                  min_samples_leaf=rf_cfg["min_samples_leaf"], max_features=rf_cfg["max_features"],
                                  class_weight=rf_cfg["class_weight"], n_jobs=rf_cfg["n_jobs"],
                                  random_state=rf_cfg["random_state"])
