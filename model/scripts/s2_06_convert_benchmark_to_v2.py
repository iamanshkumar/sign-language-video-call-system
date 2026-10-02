"""Step 2.6: build a v2 (mixed-precision) landmark store from the Phase A benchmark extractions.

Purpose: give Step 3 realistic v2 input without re-running MediaPipe. For every Phase A
video, the v2 rules are re-applied to the stored raw detections (same functions as the
pipeline) and accepted videos are re-packed with the approved output config. Writes a NEW
store (data/landmarks/benchmark_v2_mixed/); Phase A artifacts are only read. Refuses to
overwrite an existing store.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks import preprocess as pp
from asl.landmarks.convert import v2_from_v1
from asl.landmarks.layout import feature_dim
from asl.landmarks.outputs import check_outputs, load_arrays
from asl.landmarks.pipeline import DOWNLOAD_FAILED, load_metadata
from asl.landmarks.process import ARRAY_VERSION, HANDS, save_arrays

KEEP_V1 = ["sample_id", "repo_path", "word", "split", "original_fps", "container_fps", "original_frame_count",
           "original_duration_s", "width", "height", "codec", "std_frame_count", "decode_s", "mediapipe_s"]


def main() -> None:
    cfg = load_yaml(STEP2_CONFIG)
    rules, out_cfg = cfg["rejection"], cfg["output"]
    src = PROJECT_ROOT / cfg["benchmark"]["out_dir"]
    dst = PROJECT_ROOT / "data" / "landmarks" / "benchmark_v2_mixed"
    if dst.exists():
        raise SystemExit(f"{dst} exists; delete it explicitly to rebuild")
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "arrays").mkdir(parents=True)
    (tmp / "meta_parts").mkdir()

    v1 = load_metadata(src)
    v1 = v1[v1["rejection_reason"].ne(DOWNLOAD_FAILED)]
    reeval = pd.read_csv(src / "benchmark_metadata_v2_reevaluated.csv").set_index("sample_id")
    rows = []
    for r in v1.itertuples(index=False):
        rec = {k: getattr(r, k) for k in KEEP_V1}
        rec.update(format_version=ARRAY_VERSION, coord_dtype=json.dumps(out_cfg["coord_dtype"]),
                   stored_groups=json.dumps(out_cfg["groups_to_store"]),
                   configurations=json.dumps({k: feature_dim(v) for k, v in cfg["configurations"].items()}),
                   array_file=None, bytes=0, array_kept_for_rejected=False, error="",
                   source="converted from Phase A v1 extraction (no MediaPipe re-run)")
        if not (isinstance(r.array_file, str) and r.array_file):
            rec.update(rejected=True, rejection_reason=r.rejection_reason)  # e.g. too few decoded frames
            rows.append(rec)
            continue
        a = load_arrays(src / "arrays" / r.array_file)
        hands = a["observed"][:, HANDS].astype(bool)
        iv = pp.active_interval(hands, rules["active_interval_min_hand_run"])
        reason = pp.rejection_reason(int(r.original_frame_count), hands, rules)
        expected = reeval.loc[r.sample_id, "v2_rejection_reason"]
        assert (reason or "") == (expected if isinstance(expected, str) else ""), r.sample_id
        rec.update(rejected=reason is not None, rejection_reason=reason,
                   active_start_frame=iv.start if iv else None, active_end_frame=iv.end if iv else None,
                   active_frames=iv.frames if iv else 0,
                   short_active_interval=bool(iv is not None and iv.frames < rules["short_active_interval_frames"]))
        if reason is None:
            arrays = v2_from_v1(a, iv, out_cfg)
            name = r.array_file  # same stable name (sha1 of sample_id)
            rec.update(array_file=name, bytes=save_arrays(arrays, tmp / "arrays" / name, out_cfg["compress"]))
        rows.append(rec)

    meta = pd.DataFrame(rows)
    meta.to_parquet(tmp / "meta_parts" / "part_00000.parquet", index=False)
    problems = check_outputs(meta, tmp / "arrays", cfg["configurations"])
    if problems:
        raise SystemExit(f"consistency problems: {problems[:10]}")
    tmp.rename(dst)
    summary = {"videos": int(len(meta)), "accepted": int((~meta["rejected"]).sum()),
               "rejection_reasons": meta.loc[meta["rejected"], "rejection_reason"].value_counts().to_dict(),
               "short_active_interval_accepted": int(meta.loc[~meta["rejected"], "short_active_interval"].sum()),
               "arrays_bytes": int(meta["bytes"].sum()), "coord_dtype": out_cfg["coord_dtype"],
               "consistency_problems": 0}
    (dst / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
