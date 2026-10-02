"""Step 3.2: build the sequence index for a landmark store (no arrays are copied).

Reads the landmark metadata, the frozen split (authoritative split + groups) and the label
mapping; writes <index_dir>/index.parquet, excluded.csv and index_summary.json. Re-running
must reproduce an identical index (checked; an existing index is never silently replaced).
"""
from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.landmarks.pipeline import DOWNLOAD_FAILED, load_metadata
from asl.sequences.index import build_index
from asl.sequences.labels import load_label_mapping

STEP3_CONFIG = PROJECT_ROOT / "configs" / "step3_sequences.yaml"


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def main() -> None:
    c1, c2, c3 = load_config(), load_yaml(STEP2_CONFIG), load_yaml(STEP3_CONFIG)
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="benchmark_v2_mixed", choices=sorted(c3["stores"]))
    args = ap.parse_args()
    store = c3["stores"][args.store]
    lm_dir, out_dir = PROJECT_ROOT / store["landmarks_dir"], PROJECT_ROOT / store["index_dir"]

    frozen = load_frozen_split(c1.path("splits"), c1["split"]["version"])
    mapping = load_label_mapping(PROJECT_ROOT / c3["labels"]["mapping_path"])
    meta = load_metadata(lm_dir)
    meta = meta[meta["rejection_reason"].ne(DOWNLOAD_FAILED)]
    idx, excluded = build_index(meta, frozen, mapping, lm_dir / "arrays", c3["sequences"]["windows"],
                                c2["rejection"]["short_active_interval_frames"])
    digest = frame_hash(idx)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        if frame_hash(old) != digest:
            raise SystemExit(f"{path} exists and differs from a fresh build; refusing to overwrite")
        print("existing index is identical to a fresh build (reproducible)")
    else:
        idx.to_parquet(path, index=False)
        excluded.to_csv(out_dir / "excluded.csv", index=False)
    summary = {
        "store": args.store, "landmark_videos": int(len(meta)), "accepted_indexed": int(len(idx)),
        "excluded": int(len(excluded)), "excluded_by_reason": excluded["rejection_reason"].value_counts().to_dict(),
        "excluded_by_split": excluded["split"].value_counts().to_dict(),
        "index_sha256": digest, "label_mapping_vocabulary_sha256": mapping["vocabulary_sha256"],
        "windows": c3["sequences"]["windows"],
    }
    (out_dir / "index_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
