"""Step 3.1: build the label mapping from the frozen Step 1 split vocabulary (once)."""
from __future__ import annotations

import json

from asl.config import PROJECT_ROOT, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.sequences.labels import build_label_mapping, load_label_mapping, save_label_mapping

STEP3_CONFIG = PROJECT_ROOT / "configs" / "step3_sequences.yaml"


def main() -> None:
    c1, c3 = load_config(), load_yaml(STEP3_CONFIG)
    splits_dir, version = c1.path("splits"), c1["split"]["version"]
    frozen = load_frozen_split(splits_dir, version)  # checksum-verified
    lock = json.loads((splits_dir / f"split_{version}.lock.json").read_text())
    mapping = build_label_mapping(frozen, lock["sha256"])
    path = PROJECT_ROOT / c3["labels"]["mapping_path"]
    save_label_mapping(mapping, path)  # refuses to replace a different mapping
    m = load_label_mapping(path)
    per_split = {s: int(frozen.loc[frozen["split"] == s, "word"].nunique()) for s in ("train", "val", "test")}
    print(json.dumps({"path": str(path.relative_to(PROJECT_ROOT)), "num_classes": m["num_classes"],
                      "vocabulary_sha256": m["vocabulary_sha256"], "frozen_split_sha256": m["frozen_split_sha256"],
                      "classes_per_split": per_split,
                      "first": list(m["word_to_index"].items())[:3],
                      "last": list(m["word_to_index"].items())[-3:]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
