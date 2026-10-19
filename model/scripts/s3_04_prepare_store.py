"""Step 3.4: sequence preparation for any landmark store (path-driven; Step 3 code unchanged).

Builds the sequence index with the unchanged `build_index` (frozen split, shared label
mapping, leakage checks), then generates every sequence of every configuration x window
x split twice and checks shape, finiteness, presence flags and bit-identical regeneration.
Writes <index_dir>/index.parquet (never silently replaced), excluded.csv, index_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.landmarks.layout import GROUPS, feature_dim, group_feature_dim
from asl.landmarks.pipeline import DOWNLOAD_FAILED, load_metadata
from asl.landmarks.storage import available_configurations, store_groups
from asl.data.vocabulary import load_vocabulary, subset_label_mapping
from asl.sequences.dataset import SequenceDataset
from asl.sequences.index import INTERVAL_BINS, SPLITS, build_index
from asl.sequences.labels import load_label_mapping, save_label_mapping

STEP3 = PROJECT_ROOT / "configs" / "step3_sequences.yaml"


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def main() -> None:
    c1, c2, c3 = load_config(), load_yaml(STEP2_CONFIG), load_yaml(STEP3)
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-dir", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--vocabulary", default=None,
                    help="vocabulary JSON: use its subset label mapping (data/sequences/label_mapping_<name>.json)")
    args = ap.parse_args()
    lm, out = PROJECT_ROOT / args.landmarks_dir, PROJECT_ROOT / args.index_dir
    frozen = load_frozen_split(c1.path("splits"), c1["split"]["version"])
    if args.vocabulary:
        lock = json.loads((c1.path("splits") / f"split_{c1['split']['version']}.lock.json").read_text())
        voc = load_vocabulary(PROJECT_ROOT / args.vocabulary, lock["sha256"])
        mapping_path = PROJECT_ROOT / "data" / "sequences" / f"label_mapping_{voc['name']}.json"
        save_label_mapping(subset_label_mapping(frozen, voc, lock["sha256"]), mapping_path)  # write-once
    else:
        mapping_path = PROJECT_ROOT / c3["labels"]["mapping_path"]
    mapping = load_label_mapping(mapping_path)
    meta = load_metadata(lm)
    meta = meta[meta["rejection_reason"].ne(DOWNLOAD_FAILED)]
    windows = c3["sequences"]["windows"]
    configs = available_configurations(c2["configurations"], store_groups(meta))  # e.g. no face in a hands+pose store
    idx, excluded = build_index(meta, frozen, mapping, lm / "arrays", windows, c2["rejection"]["short_active_interval_frames"])
    digest = frame_hash(idx)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index.parquet").exists():
        if frame_hash(pd.read_parquet(out / "index.parquet")) != digest:
            raise SystemExit("existing index differs from a fresh build; refusing to overwrite")
    else:
        idx.to_parquet(out / "index.parquet", index=False)
        excluded.to_csv(out / "excluded.csv", index=False)

    checks, seqs = {"shape_finite_presence": True, "regeneration_identical": True}, {}
    for cname, groups in configs.items():
        F = feature_dim(groups)
        pcols, off = [], 0
        for g in groups:  # presence column follows the group's x, y, z columns
            pcols.append(off + GROUPS[g] * 3)
            off += group_feature_dim(g)
        for w in windows:
            for split in SPLITS:
                ds = SequenceDataset(idx, lm / "arrays", split, w, cname, configs, mapping)
                for i in range(len(ds)):
                    x, _ = ds[i]
                    a = x.numpy()
                    ok = a.shape == (w, F) and np.isfinite(a).all() and np.isin(a[:, pcols], (0.0, 1.0)).all()
                    checks["shape_finite_presence"] &= bool(ok)
                    checks["regeneration_identical"] &= bool(np.array_equal(a, ds[i][0].numpy()))
                seqs[f"{cname}/w{w}/{split}"] = len(ds)
    summary = {
        "landmarks_dir": args.landmarks_dir, "index_dir": args.index_dir, "index_sha256": digest,
        "label_mapping": str(mapping_path.relative_to(PROJECT_ROOT)), "num_classes": mapping["num_classes"],
        "configurations": {k: feature_dim(v) for k, v in configs.items()},
        "label_mapping_vocabulary_sha256": mapping["vocabulary_sha256"], "landmark_videos": int(len(meta)),
        "accepted_indexed": int(len(idx)), "excluded": int(len(excluded)),
        "excluded_by_split_reason": excluded.groupby(["split", "rejection_reason"]).size().rename("n").reset_index()
        .to_dict("records"),
        "by_split": {s: {"videos": int((idx.split == s).sum()), "classes": int(idx.loc[idx.split == s, "word"].nunique()),
                         "short_active": int(idx.loc[idx.split == s, "short_active_interval"].sum()),
                         "active_bins": {n: int(((idx.split == s) & (idx.active_bin == n)).sum()) for _, _, n in INTERVAL_BINS}}
                     for s in SPLITS},
        "sequences_per_config_window_split": seqs,
        "unable_to_produce_window": int((idx["active_frames"] < 2).sum()),
        "max_stretch": {w: float(idx[f"stretch_{w}"].max()) for w in windows} if len(idx) else {},
        "checks": checks, "all_checks_pass": all(checks.values()),
    }
    (out / "index_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_pass"]:
        raise SystemExit("sequence checks failed")


if __name__ == "__main__":
    main()
