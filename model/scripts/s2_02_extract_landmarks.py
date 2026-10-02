"""Step 2.2: resumable MediaPipe Holistic landmark extraction.

Inputs: a sample CSV (default: the benchmark sample) or `--from-split` (frozen split,
for a future approved full run on a cloud machine). Re-run to resume. Runs larger
than output.max_videos_without_approval are refused unless --approved-large-run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.landmarks.pipeline import hf_fetcher, run_pipeline


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    b = c2["benchmark"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(PROJECT_ROOT / b["out_dir"] / "benchmark_sample.csv"))
    ap.add_argument("--from-split", choices=["train", "val", "test", "all"], default=None)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / b["out_dir"]))
    ap.add_argument("--workers", type=int, default=b["workers"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--approved-large-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.from_split:
        samples = load_frozen_split(c1.path("splits"), c1["split"]["version"])  # checksum-verified
        if args.from_split != "all":
            samples = samples[samples["split"] == args.from_split]
    else:
        samples = pd.read_csv(args.samples, dtype=str, keep_default_na=False)
    n = len(samples) if args.limit is None else min(args.limit, len(samples))
    cap = c2["output"]["max_videos_without_approval"]
    if n > cap and not args.approved_large_run:
        raise SystemExit(f"refusing to process {n:,} videos (> {cap:,}) without --approved-large-run")

    ds, p = c1["dataset"], c1["probe"]
    summary = run_pipeline(
        samples[["sample_id", "repo_path", "word", "split", "sha256"]], c2, Path(args.out_dir),
        c1.path("raw_cache"), hf_fetcher(ds["repo_id"], ds["revision"], p["download_retries"]),
        workers=args.workers, download_workers=b["download_workers"], batch_size=b["batch_size"],
        limit=args.limit, min_free_disk_gb=p["min_free_disk_gb"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
