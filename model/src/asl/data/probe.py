"""Resumable batch download -> full PyAV decode -> delete.

The dataset (~54 GB) does not fit on disk, so videos are processed in batches:
each batch is downloaded, every file's SHA-256 is checked against the repo
index, every file is fully decoded, and the batch directory is deleted. Results
are appended as one Parquet part per batch, so an interrupted run resumes where
it stopped. Downloads that fail (network problems) are retried on the next run
instead of being counted as corrupt videos.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import av
import pandas as pd

PROBE_COLUMNS = [
    "repo_path", "download_ok", "download_error", "sha256_match", "decode_ok", "decode_error",
    "container_format", "container_duration_s", "n_video_streams", "has_audio", "codec",
    "width", "height", "pix_fmt", "rotate", "avg_fps", "base_fps", "reported_frames",
    "stream_duration_s", "decoded_frames", "first_pts_s", "last_pts_s", "decoded_fps",
]


def probe_video(path: str) -> dict:
    """Fully decode the first video stream and collect metadata. Never raises."""
    rec: dict = {"decode_ok": False, "decode_error": "", "decoded_frames": 0}
    try:
        # Some files carry non-UTF-8 metadata tags; they must not be mistaken for
        # undecodable video, so metadata decoding errors are ignored.
        with av.open(path, metadata_errors="ignore") as container:
            rec["container_format"] = container.format.name
            rec["container_duration_s"] = container.duration / 1e6 if container.duration else None
            videos = [s for s in container.streams if s.type == "video"]
            rec["n_video_streams"] = len(videos)
            rec["has_audio"] = any(s.type == "audio" for s in container.streams)
            if not videos:
                raise ValueError("no video stream")
            s = videos[0]
            s.thread_type = "AUTO"
            cc = s.codec_context
            rec["codec"] = cc.name
            rec["width"], rec["height"] = cc.width, cc.height
            rec["pix_fmt"] = cc.pix_fmt
            rec["rotate"] = s.metadata.get("rotate")
            rec["avg_fps"] = float(s.average_rate) if s.average_rate else None
            rec["base_fps"] = float(s.base_rate) if s.base_rate else None
            rec["reported_frames"] = int(s.frames) if s.frames else None
            rec["stream_duration_s"] = float(s.duration * s.time_base) if s.duration else None

            n, first, last = 0, None, None
            for frame in container.decode(s):
                n += 1
                if frame.time is not None:
                    first = frame.time if first is None else min(first, frame.time)
                    last = frame.time if last is None else max(last, frame.time)
            rec["decoded_frames"] = n
            rec["first_pts_s"], rec["last_pts_s"] = first, last
            if n > 1 and first is not None and last > first:
                rec["decoded_fps"] = (n - 1) / (last - first)
            rec["decode_ok"] = True
    except Exception as e:  # corrupt files raise a variety of av/ValueError types
        rec["decode_error"] = f"{type(e).__name__}: {e}"[:300]
        rec["decoded_frames"] = locals().get("n", 0)
    return rec


def _download(url: str, dest: Path, expected_sha: str, retries: int) -> dict:
    last_err = ""
    for attempt in range(retries):
        try:
            h = hashlib.sha256()
            req = urllib.request.Request(url, headers={"User-Agent": "asl-step1-probe"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            return {"download_ok": True, "download_error": "", "sha256_match": h.hexdigest() == expected_sha}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"[:300]
            wait_s = min(2**attempt, 30)
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                # Rate limited: honour Retry-After, otherwise back off for longer.
                try:
                    wait_s = max(float(e.headers.get("Retry-After", 0)), 30.0)
                except (TypeError, ValueError):
                    wait_s = 30.0
                wait_s = min(wait_s * (1 + attempt), 300.0)
            time.sleep(wait_s)
    return {"download_ok": False, "download_error": last_err, "sha256_match": None}


def _file_url(repo_id: str, revision: str, repo_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
        + urllib.parse.quote(repo_path)
    )


def completed_paths(parts_dir: Path) -> set[str]:
    """Samples already downloaded + decoded (download failures are retried)."""
    df = load_probe_results(parts_dir)
    return set() if df.empty else set(df.loc[df["download_ok"], "repo_path"])


def load_probe_results(parts_dir: Path) -> pd.DataFrame:
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame(columns=PROBE_COLUMNS)
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # Later parts supersede earlier ones; a successful download beats a failed one.
    df = df.sort_values("download_ok", kind="stable").drop_duplicates("repo_path", keep="last")
    return df.sort_values("repo_path", ignore_index=True)


def run_probe(
    samples: pd.DataFrame,
    *,
    repo_id: str,
    revision: str,
    cache_dir: Path,
    parts_dir: Path,
    batch_size: int,
    download_workers: int,
    decode_workers: int,
    min_free_disk_gb: float,
    retries: int,
    limit: int | None = None,
    force: set[str] | None = None,
) -> None:
    """`samples` needs columns repo_path, sha256. Writes parts_dir/part_NNNNN.parquet.

    `force`: paths to probe again even if already done (a later part supersedes earlier ones).
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = completed_paths(parts_dir) - (force or set())
    todo = samples[~samples["repo_path"].isin(done)].sort_values("repo_path")
    if limit is not None:
        todo = todo.head(limit)
    total, n_done_start = len(todo), len(done)
    print(f"[probe] {n_done_start} already done, {total} to process", flush=True)
    t0, processed = time.time(), 0

    with ThreadPoolExecutor(download_workers) as dl_pool, ProcessPoolExecutor(decode_workers) as dec_pool:
        for start in range(0, total, batch_size):
            batch = todo.iloc[start : start + batch_size]
            free_gb = shutil.disk_usage(cache_dir).free / 1e9
            if free_gb < min_free_disk_gb:
                sys.exit(f"[probe] stopping: only {free_gb:.1f} GB free (< {min_free_disk_gb} GB)")

            part_idx = len(list(parts_dir.glob("part_*.parquet")))
            bdir = cache_dir / f"batch_{part_idx:05d}"
            bdir.mkdir(parents=True, exist_ok=True)
            records: dict[str, dict] = {}
            pending = {}
            dl_futs = {}
            for i, row in enumerate(batch.itertuples(index=False)):
                dest = bdir / f"{i:05d}.mp4"
                url = _file_url(repo_id, revision, row.repo_path)
                dl_futs[dl_pool.submit(_download, url, dest, row.sha256, retries)] = (row.repo_path, dest)

            # Decode each file as soon as its download finishes, then delete it.
            waiting = set(dl_futs)
            while waiting or pending:
                done_set, _ = wait(waiting | set(pending), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    if fut in dl_futs:
                        waiting.discard(fut)
                        repo_path, dest = dl_futs[fut]
                        rec = {"repo_path": repo_path, **fut.result()}
                        records[repo_path] = rec
                        if rec["download_ok"]:
                            pending[dec_pool.submit(probe_video, str(dest))] = (repo_path, dest)
                        else:
                            dest.unlink(missing_ok=True)
                    else:
                        repo_path, dest = pending.pop(fut)
                        records[repo_path].update(fut.result())
                        dest.unlink(missing_ok=True)

            df = pd.DataFrame(list(records.values())).reindex(columns=PROBE_COLUMNS)
            tmp = parts_dir / f"part_{part_idx:05d}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.rename(parts_dir / f"part_{part_idx:05d}.parquet")
            shutil.rmtree(bdir, ignore_errors=True)

            processed += len(batch)
            rate = processed / (time.time() - t0)
            eta_min = (total - processed) / rate / 60 if rate else float("nan")
            print(
                f"[probe] part {part_idx:05d}: {processed}/{total} "
                f"(dl_fail={int((~df['download_ok']).sum())}, "
                f"decode_fail={int((df['download_ok'] & ~df['decode_ok'].fillna(False).astype(bool)).sum())}) "
                f"{rate:.1f} files/s, ETA {eta_min:.0f} min",
                flush=True,
            )
