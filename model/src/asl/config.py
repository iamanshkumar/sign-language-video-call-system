"""Load and validate the project YAML config."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "step1_data.yaml"
STEP2_CONFIG = PROJECT_ROOT / "configs" / "step2_landmarks.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        """Resolve a `paths.<key>` entry relative to the project root."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    for section in ("dataset", "paths", "probe", "validation", "split"):
        if section not in raw:
            raise ValueError(f"config missing section: {section}")
    s = raw["split"]
    total = s["train"] + s["val"] + s["test"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {total}")
    if len(raw["dataset"]["revision"]) != 40:
        raise ValueError("dataset.revision must be a full 40-char commit hash")
    return Config(raw)
