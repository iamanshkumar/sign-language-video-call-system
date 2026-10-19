"""Step 2.2: resumable MediaPipe Holistic landmark extraction.

Inputs: a sample CSV (default: the benchmark sample) or `--from-split` (frozen split,
for a future approved full run on a cloud machine). Re-run to resume. Runs larger
than output.max_videos_without_approval are refused unless --approved-large-run.

Step 5 options: `--vocabulary configs/vocabulary_100.json` restricts the frozen split to
the vocabulary's classes (implies --from-split all unless --from-split is given);
`--classes N` and `--per-split train=4,val=1,test=1` take a deterministic subset (pipeline
tests); `--profile hands_pose_f32` stores hands + pose only, all float32 (no face).
`--estimate-only` prints the run estimate and exits.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.data.vocabulary import load_vocabulary, select_samples, vocabulary_words
from asl.landmarks.layout import GROUPS
from asl.landmarks.pipeline import DOWNLOAD_FAILED, completed_ids, hf_fetcher, load_metadata, run_pipeline
from asl.landmarks.storage import apply_profile

# Fallbacks when the store has no measurements yet (Step 1 EDA sample / Step 2 benchmark).
EDA_MEAN_FRAMES_30FPS = 86
M2_FRAMES_PER_S_PER_WORKER = 15.5


def parse_per_split(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        if k not in ("train", "val", "test"):
            raise SystemExit(f"--per-split: unknown split {k!r}")
        out[k] = int(v)
    return out


def bytes_per_frame(out_cfg: dict) -> int:
    """Uncompressed bytes per 30 FPS frame of a stored v2 array set."""
    import numpy as np

    b = sum(GROUPS[g] * 3 * np.dtype(out_cfg["coord_dtype"][g]).itemsize for g in out_cfg["groups_to_store"])
    return b + 33 * np.dtype(out_cfg["coord_dtype"]["pose"]).itemsize + 4 + 12 + 4 + 1  # vis, flags, center, scale, sm


def estimate(todo: pd.DataFrame, sizes: pd.Series, out_dir: Path, cfg: dict, workers: int) -> dict:
    meta = load_metadata(out_dir) if (out_dir / "meta_parts").exists() else pd.DataFrame()
    meta = meta[meta["rejection_reason"].ne(DOWNLOAD_FAILED)] if not meta.empty else meta
    n = len(todo)
    dl = float(sizes.reindex(todo["sample_id"]).sum())
    if len(meta) >= 20:  # measured on this store
        acc = meta[meta["bytes"] > 0]
        per_video_s = float(meta["total_s"].mean())
        out_b = float(acc["bytes"].mean()) * n * len(acc) / len(meta)
        basis = f"measured on {len(meta)} videos already in this store"
    else:
        per_video_s = EDA_MEAN_FRAMES_30FPS / M2_FRAMES_PER_S_PER_WORKER
        out_b = n * 0.935 * (EDA_MEAN_FRAMES_30FPS * bytes_per_frame(cfg["output"]) + 3000)
        basis = (f"theoretical: {EDA_MEAN_FRAMES_30FPS} frames/video (EDA mean), "
                 f"{M2_FRAMES_PER_S_PER_WORKER} frames/s/worker (M2 benchmark), 93.5% acceptance")
    cpu_s = per_video_s * n
    return {"videos_to_process": n, "download_gb": dl / 1e9, "output_gb": out_b / 1e9, "cpu_hours": cpu_s / 3600,
            "wall_hours_at_workers": cpu_s / 3600 / max(workers, 1), "workers": workers, "basis": basis}


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    b = c2["benchmark"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(PROJECT_ROOT / b["out_dir"] / "benchmark_sample.csv"))
    ap.add_argument("--from-split", choices=["train", "val", "test", "all"], default=None)
    ap.add_argument("--vocabulary", default=None, help="vocabulary JSON (Step 5); restricts the frozen split")
    ap.add_argument("--classes", type=int, default=None, help="with --vocabulary: deterministic class subset")
    ap.add_argument("--per-split", default=None, help="with --vocabulary: e.g. train=4,val=1,test=1 per class")
    ap.add_argument("--profile", default=None, help="storage profile from step2_landmarks.yaml")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / b["out_dir"]))
    ap.add_argument("--workers", type=int, default=b["workers"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--approved-large-run", action="store_true")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = apply_profile(c2, args.profile)
    out_dir = Path(args.out_dir)

    if args.vocabulary or args.from_split:
        samples = load_frozen_split(c1.path("splits"), c1["split"]["version"])  # checksum-verified
        if args.vocabulary:
            lock = json.loads((c1.path("splits") / f"split_{c1['split']['version']}.lock.json").read_text())
            v = load_vocabulary(PROJECT_ROOT / args.vocabulary, lock["sha256"])
            samples = select_samples(samples, vocabulary_words(v), parse_per_split(args.per_split), args.classes)
        elif args.classes or args.per_split:
            raise SystemExit("--classes / --per-split need --vocabulary")
        if args.from_split and args.from_split != "all":
            samples = samples[samples["split"] == args.from_split]
    else:
        samples = pd.read_csv(args.samples, dtype=str, keep_default_na=False)

    done = completed_ids(out_dir) if (out_dir / "meta_parts").exists() else set()
    todo = samples[~samples["sample_id"].isin(done)]
    if args.limit is not None:
        todo = todo.head(args.limit)
    man = pd.read_parquet(c1.path("metadata") / "manifest.parquet")
    sizes = man[man["eligible_pre_probe"]].set_index("sample_id")["size_bytes"]  # issue rows (dup/missing) dropped
    est = estimate(todo, sizes, out_dir, cfg, args.workers)
    print(f"Selected videos:     {len(samples):,} ({samples['word'].nunique()} classes; "
          f"{samples['split'].value_counts().to_dict()})")
    print(f"Already processed:   {len(samples) - len(samples[~samples['sample_id'].isin(done)]):,}")
    print(f"Videos to process:   {est['videos_to_process']:,}")
    print(f"Estimated download:  {est['download_gb']:.2f} GB (streamed per batch, deleted after processing)")
    print(f"Estimated output:    {est['output_gb']:.3f} GB")
    print(f"Estimated CPU time:  {est['cpu_hours']:.2f} worker-hours (~{est['wall_hours_at_workers']:.2f} h wall)")
    print(f"Workers:             {args.workers}")
    print(f"Storage profile:     {args.profile or 'default'} -> groups {cfg['output']['groups_to_store']}, "
          f"configurations {list(cfg['configurations'])}")
    print(f"Estimate basis:      {est['basis']}")
    if args.estimate_only:
        return

    cap = cfg["output"]["max_videos_without_approval"]
    if len(todo) > cap and not args.approved_large_run:
        raise SystemExit(f"refusing to process {len(todo):,} videos (> {cap:,}) without --approved-large-run")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.vocabulary:  # record exactly what was selected
        sel = out_dir / "selections"
        sel.mkdir(exist_ok=True)
        tag = f"{Path(args.vocabulary).stem}_c{args.classes or 'all'}_{(args.per_split or 'all').replace(',', '_')}"
        samples[["sample_id", "repo_path", "word", "split", "sha256"]].to_csv(sel / f"{tag}.csv", index=False)

    ds, p = c1["dataset"], c1["probe"]
    summary = run_pipeline(
        todo[["sample_id", "repo_path", "word", "split", "sha256"]], cfg, out_dir,
        c1.path("raw_cache"), hf_fetcher(ds["repo_id"], ds["revision"], p["download_retries"]),
        workers=args.workers, download_workers=b["download_workers"], batch_size=b["batch_size"],
        limit=None, min_free_disk_gb=p["min_free_disk_gb"],
    )
    summary["storage_profile"] = args.profile
    summary["vocabulary"] = args.vocabulary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
