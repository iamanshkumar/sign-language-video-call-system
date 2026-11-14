"""The exact input contract of a trained model, recorded at export and checked at load time.

Everything that decides what a model input looks like - MediaPipe settings, 30 FPS resampling,
rejection rules, gap handling, shoulder normalization, stored groups, feature configuration,
feature order, temporal window - is captured here from the same configs training used. Inference
refuses a checkpoint whose recorded spec differs from the current code/config.
"""
from __future__ import annotations

import hashlib
import json

from asl.landmarks.layout import feature_dim, feature_names
from asl.landmarks.storage import apply_profile


def preprocessing_spec(step2_cfg: dict, step3_cfg: dict, profile: str | None, configuration: str,
                       window: int) -> dict:
    cfg = apply_profile(step2_cfg, profile)
    if configuration not in cfg["configurations"]:
        raise ValueError(f"configuration {configuration!r} is not available with profile {profile!r}")
    groups = cfg["configurations"][configuration]
    names = feature_names(groups)
    return {
        "mediapipe": cfg["mediapipe"], "preprocessing": cfg["preprocessing"], "rejection": cfg["rejection"],
        "storage_profile": profile, "stored_groups": cfg["output"]["groups_to_store"],
        "coord_dtype": cfg["output"]["coord_dtype"], "configuration": configuration, "groups": groups,
        "feature_dim": feature_dim(groups), "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "window": window, "content_region": step3_cfg["sequences"]["content_region"],
        "resampling": step3_cfg["sequences"]["resampling"],
    }


def spec_digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def spec_differences(recorded: dict, current: dict) -> list[str]:
    """Keys whose values differ (empty list = identical input contract)."""
    keys = sorted(set(recorded) | set(current))
    return [k for k in keys if recorded.get(k) != current.get(k)]
