"""Step 2.7: plan a (remote) extraction run for frozen splits: sample lists, shards, fingerprint.

Writes data/landmarks/runs/<run>/plan.json and shards/shard_XX_of_NN.csv (+ parity.csv:
benchmark videos already extracted on the M2, re-extracted remotely to measure
cross-platform differences). Refuses to overwrite an existing plan.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.landmarks.cloud import code_fingerprint, extraction_list, host_info, shard

COLS = ["sample_id", "repo_path", "word", "split", "sha256"]


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit-per-split", type=int, default=None, help="dry runs only")
    ap.add_argument("--parity", type=int, default=20, help="benchmark videos to re-extract for platform parity")
    args = ap.parse_args()
    run = PROJECT_ROOT / "data" / "landmarks" / "runs" / args.run_name
    if run.exists():
        raise SystemExit(f"{run} exists; plans are never overwritten")

    frozen = load_frozen_split(c1.path("splits"), c1["split"]["version"])
    todo = extraction_list(frozen, args.splits, args.limit_per_split)
    (run / "shards").mkdir(parents=True)
    for i, part in enumerate(shard(todo, args.shards)):
        part[COLS].to_csv(run / "shards" / f"shard_{i:02d}_of_{args.shards:02d}.csv", index=False)
    if args.parity:
        bench = pd.read_parquet(PROJECT_ROOT / "data/landmarks/benchmark_v2_mixed/meta_parts/part_00000.parquet")
        ok = bench[~bench["rejected"].astype(bool)].sort_values("sample_id").head(args.parity)
        frozen.set_index("sample_id").loc[ok["sample_id"]].reset_index()[COLS].to_csv(run / "parity.csv", index=False)
    est_frames = 84.45 * len(todo)  # Step 2 weighted mean decoded frames per video
    sizes = pd.read_parquet(c1.path("metadata") / "manifest.parquet", columns=["sample_id", "size_bytes"])
    download_gb = float(sizes.loc[sizes["sample_id"].isin(set(todo["sample_id"])), "size_bytes"].sum()) / 1e9
    plan = {
        "run_name": args.run_name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": args.splits, "videos": int(len(todo)), "by_split": todo["split"].value_counts().to_dict(),
        "shards": args.shards, "shard_sizes": [int(len(p)) for p in shard(todo, args.shards)],
        "parity_videos": args.parity, "limit_per_split": args.limit_per_split,
        "dataset_revision": c1["dataset"]["revision"], "step2_output_config": c2["output"],
        "fingerprint": code_fingerprint(PROJECT_ROOT), "planned_on": host_info(),
        "estimates": {"source_frames": est_frames,
                      "worker_hours_at_m2_rate_15_5_fps": est_frames / 15.5 / 3600,
                      "download_gb": download_gb,
                      "output_gb_at_93_5pct_acceptance": len(todo) * 0.935 * 0.316 / 1e3},
    }
    (run / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: plan[k] for k in ("run_name", "videos", "by_split", "shards", "shard_sizes", "estimates")}, indent=2))


if __name__ == "__main__":
    main()
