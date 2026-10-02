"""Step 1.2b: probe ONLY the EDA sample (batch download -> SHA-256 check -> PyAV decode -> delete).

Resumable: re-run to continue and to retry failed (e.g. HTTP 429) downloads.
Measurements that the earlier, abandoned full-dataset probe already made for a
sampled file (same bytes, SHA-256 verified) are reused instead of re-downloaded
and tagged probe_origin = "legacy_full_probe".
"""
from __future__ import annotations

import argparse

import pandas as pd

from asl.config import PROJECT_ROOT, load_config
from asl.data.probe import load_probe_results, run_probe

LEGACY_PART = "part_00000_legacy_reuse.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--reprobe-decode-failures", action="store_true",
                    help="probe again files that downloaded but failed to decode (after a prober fix)")
    args = ap.parse_args()

    cfg = load_config()
    p, ds = cfg["probe"], cfg["dataset"]
    sdir = PROJECT_ROOT / cfg["eda_sample"]["dir"]
    parts = sdir / "probe_parts"
    parts.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sdir / "sample.csv", dtype=str, keep_default_na=False)

    if not (parts / LEGACY_PART).exists():
        legacy = load_probe_results(cfg.path("metadata") / "probe_parts")
        reuse = legacy[legacy["repo_path"].isin(set(sample["repo_path"]))
                       & legacy["download_ok"].astype(bool) & legacy["sha256_match"].astype(bool)]
        reuse.to_parquet(parts / LEGACY_PART, index=False)
        print(f"[sample-probe] reused {len(reuse)} legacy measurements")

    force = None
    if args.reprobe_decode_failures:
        prev = load_probe_results(parts)
        force = set(prev.loc[prev["download_ok"].astype(bool) & ~prev["decode_ok"].fillna(False).astype(bool),
                             "repo_path"])
        print(f"[sample-probe] re-probing {len(force)} decode failures: {sorted(force)}")

    run_probe(
        sample[["repo_path", "sha256"]],
        repo_id=ds["repo_id"], revision=ds["revision"],
        cache_dir=cfg.path("raw_cache"), parts_dir=parts,
        batch_size=args.batch_size, download_workers=p["download_workers"],
        decode_workers=p["decode_workers"], min_free_disk_gb=p["min_free_disk_gb"],
        retries=p["download_retries"], force=force,
    )

    res = load_probe_results(parts)
    res["probe_origin"] = "sample_probe"
    legacy_paths = set(pd.read_parquet(parts / LEGACY_PART)["repo_path"])
    res.loc[res["repo_path"].isin(legacy_paths), "probe_origin"] = "legacy_full_probe"
    missing = set(sample["repo_path"]) - set(res["repo_path"])
    failed = int((~res["download_ok"].astype(bool)).sum())
    print(f"[sample-probe] recorded {len(res)}/{len(sample)}; unprobed {len(missing)}; download failures {failed}")
    res.to_parquet(sdir / "sample_probe_results.parquet", index=False)


if __name__ == "__main__":
    main()
