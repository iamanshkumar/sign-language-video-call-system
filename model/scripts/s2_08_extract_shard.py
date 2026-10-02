"""Step 2.8: extract ONE shard of a planned run with the unchanged Step 2 pipeline.

Refuses to run if the code/config fingerprint differs from the plan (so a remote run is
provably identical code). Resumable: re-run the same command after an interruption.
Set HF_TOKEN in the environment to avoid Hugging Face rate limits (sent only to huggingface.co).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.landmarks.cloud import code_fingerprint, host_info, install_hf_auth
from asl.landmarks.pipeline import hf_fetcher, run_pipeline


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--shard", type=int, help="shard index (omit with --parity)")
    ap.add_argument("--parity", action="store_true", help="extract the parity list instead of a shard")
    ap.add_argument("--workers", type=int, default=c2["benchmark"]["workers"])
    ap.add_argument("--download-workers", type=int, default=c2["benchmark"]["download_workers"])
    ap.add_argument("--approved-large-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    run = PROJECT_ROOT / "data" / "landmarks" / "runs" / args.run_name
    plan = json.loads((run / "plan.json").read_text())
    now = code_fingerprint(PROJECT_ROOT)
    diff = sorted(k for k in set(now) | set(plan["fingerprint"]) if now.get(k) != plan["fingerprint"].get(k))
    if diff:
        raise SystemExit(f"code/config differs from the plan: {diff}")
    if args.parity:
        csv, out = run / "parity.csv", run / "parity"
    else:
        csv = run / "shards" / f"shard_{args.shard:02d}_of_{plan['shards']:02d}.csv"
        out = run / "shards" / f"shard_{args.shard:02d}"
    samples = pd.read_csv(csv, dtype=str, keep_default_na=False)
    cap = c2["output"]["max_videos_without_approval"]
    if len(samples) > cap and not args.approved_large_run:
        raise SystemExit(f"refusing {len(samples):,} videos (> {cap:,}) without --approved-large-run")

    token = install_hf_auth(os.environ.get("HF_TOKEN"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"run_name": args.run_name, "shard": args.shard, "parity": args.parity, "videos": len(samples),
                "workers": args.workers, "hf_token_used": token, "host": host_info(), "fingerprint_matches_plan": True,
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ds, p = c1["dataset"], c1["probe"]
    summary = run_pipeline(samples, c2, out, c1.path("raw_cache"), hf_fetcher(ds["repo_id"], ds["revision"],
                                                                            p["download_retries"]),
                           workers=args.workers, download_workers=args.download_workers,
                           batch_size=c2["benchmark"]["batch_size"], min_free_disk_gb=p["min_free_disk_gb"])
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), run_summary=summary)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"run_manifest_{stamp}.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
