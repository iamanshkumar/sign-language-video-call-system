"""Resumable, bounded batch runner for landmark extraction.

Per batch: download (threads, SHA-256 verified) -> process in worker processes
(one video per worker at a time; frames are streamed) -> delete the video ->
append one metadata Parquet part. Finished videos (processed OR rejected) are
never redone; failed downloads are retried on the next run.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import psutil

from asl.data.probe import _download, _file_url
from asl.landmarks.process import process_video

log = logging.getLogger("asl.landmarks")
DOWNLOAD_FAILED = "download_failed"
Fetch = Callable[[str, str, Path], dict]  # (repo_path, sha256, dest) -> {"download_ok": bool, ...}


def hf_fetcher(repo_id: str, revision: str, retries: int) -> Fetch:
    def fetch(repo_path: str, sha256: str, dest: Path) -> dict:
        r = _download(_file_url(repo_id, revision, repo_path), dest, sha256, retries)
        if r["download_ok"] and not r["sha256_match"]:
            return {"download_ok": False, "download_error": "sha256 mismatch"}
        return r
    return fetch


def load_metadata(out_dir: Path) -> pd.DataFrame:
    parts = sorted((out_dir / "meta_parts").glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["_failed_dl"] = df["rejection_reason"].eq(DOWNLOAD_FAILED)
    # A later record supersedes an earlier one; a real result beats a failed download.
    df = df.sort_values("_failed_dl", ascending=False, kind="stable").drop_duplicates("sample_id", keep="last")
    return df.drop(columns="_failed_dl").sort_values("sample_id", ignore_index=True)


def completed_ids(out_dir: Path) -> set[str]:
    df = load_metadata(out_dir)
    return set() if df.empty else set(df.loc[df["rejection_reason"].ne(DOWNLOAD_FAILED), "sample_id"])


class ResourceMonitor:
    """Samples RSS of this process + children and system CPU % once per second."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval, self.rss, self.cpu = interval, [], []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        me = psutil.Process()
        psutil.cpu_percent(None)
        while not self._stop.wait(self.interval):
            procs = [me] + me.children(recursive=True)
            rss = 0
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            self.rss.append(rss)
            self.cpu.append(psutil.cpu_percent(None))

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        if not self.rss:
            return {}
        return {"peak_rss_mb": max(self.rss) / 2**20, "mean_rss_mb": sum(self.rss) / len(self.rss) / 2**20,
                "mean_cpu_pct": sum(self.cpu) / len(self.cpu), "peak_cpu_pct": max(self.cpu),
                "logical_cpus": psutil.cpu_count()}


def run_pipeline(
    samples: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    cache_dir: Path,
    fetch: Fetch,
    *,
    workers: int,
    download_workers: int,
    batch_size: int,
    limit: int | None = None,
    min_free_disk_gb: float = 5.0,
) -> dict:
    """samples: sample_id, repo_path, word, split, sha256. Returns a run summary."""
    arrays_dir, parts_dir = out_dir / "arrays", out_dir / "meta_parts"
    for d in (arrays_dir, parts_dir, cache_dir, out_dir / "runs"):
        d.mkdir(parents=True, exist_ok=True)
    done = completed_ids(out_dir)
    todo = samples[~samples["sample_id"].isin(done)].sort_values("sample_id")
    if limit is not None:
        todo = todo.head(limit)
    log.info("already done %d, to process %d", len(done), len(todo))

    t_start = time.perf_counter()
    n_done = 0
    with ResourceMonitor() as mon, ThreadPoolExecutor(download_workers) as dl, ProcessPoolExecutor(workers) as pool:
        for start in range(0, len(todo), batch_size):
            if shutil.disk_usage(out_dir).free / 1e9 < min_free_disk_gb:
                raise RuntimeError("free disk below limit; stopping")
            batch = todo.iloc[start : start + batch_size]
            part_idx = len(list(parts_dir.glob("part_*.parquet")))
            bdir = cache_dir / f"lm_batch_{part_idx:05d}"
            bdir.mkdir(parents=True, exist_ok=True)
            records: list[dict] = []
            dl_futs, proc_futs = {}, {}
            for i, row in enumerate(batch.itertuples(index=False)):
                info = {"sample_id": row.sample_id, "repo_path": row.repo_path, "word": row.word, "split": row.split}
                dest = bdir / f"{i:05d}.mp4"
                dl_futs[dl.submit(fetch, row.repo_path, row.sha256, dest)] = (info, dest)
            waiting = set(dl_futs)
            while waiting or proc_futs:
                finished, _ = wait(waiting | set(proc_futs), return_when=FIRST_COMPLETED)
                for fut in finished:
                    if fut in dl_futs:
                        waiting.discard(fut)
                        info, dest = dl_futs[fut]
                        r = fut.result()
                        if r["download_ok"]:
                            proc_futs[pool.submit(process_video, str(dest), info, cfg, arrays_dir)] = dest
                        else:
                            records.append({**info, "rejected": None, "rejection_reason": DOWNLOAD_FAILED,
                                            "error": r.get("download_error", "")})
                            dest.unlink(missing_ok=True)
                    else:
                        dest = proc_futs.pop(fut)
                        records.append(fut.result())
                        dest.unlink(missing_ok=True)
            df = pd.DataFrame(records)
            tmp = parts_dir / f"part_{part_idx:05d}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.rename(parts_dir / f"part_{part_idx:05d}.parquet")
            shutil.rmtree(bdir, ignore_errors=True)
            n_done += len(batch)
            el = time.perf_counter() - t_start
            log.info("part %05d: %d/%d videos, %.1f videos/min, rejected %d, download failures %d",
                     part_idx, n_done, len(todo), 60 * n_done / el,
                     int(df["rejected"].eq(True).sum()), int(df["rejection_reason"].eq(DOWNLOAD_FAILED).sum()))
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "videos_attempted": int(len(todo)), "wall_s": time.perf_counter() - t_start,
               "workers": workers, "download_workers": download_workers, "batch_size": batch_size,
               **mon.summary()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / "runs" / f"run_{stamp}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
