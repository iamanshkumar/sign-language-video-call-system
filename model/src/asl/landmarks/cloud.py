"""Support for running the (unchanged) Step 2 pipeline on other machines.

- deterministic extraction lists and shards from the frozen split;
- a code/config fingerprint so a remote run can be proven to use identical code;
- optional Hugging Face token authentication (rate limits), sent ONLY to huggingface.co
  (never to redirect targets such as CDN hosts).
Nothing here changes how a video is processed.
"""
from __future__ import annotations

import hashlib
import platform
import sys
import urllib.request
from pathlib import Path

import pandas as pd

FINGERPRINT_PATHS = ["src/asl/landmarks", "src/asl/data", "src/asl/sequences", "src/asl/config.py",
                     "configs/step1_data.yaml", "configs/step2_landmarks.yaml", "configs/step3_sequences.yaml",
                     "data/splits/split_v1.csv", "data/splits/split_v1.lock.json", "data/sequences/label_mapping.json"]


def extraction_list(frozen: pd.DataFrame, splits: list[str], limit_per_split: int | None = None) -> pd.DataFrame:
    """Samples of the given frozen splits, sorted by sample_id (optionally the first N per split)."""
    df = frozen[frozen["split"].isin(splits)].sort_values("sample_id", ignore_index=True)
    if limit_per_split is not None:
        df = df.groupby("split", group_keys=False).head(limit_per_split).sort_values("sample_id", ignore_index=True)
    return df


def shard(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Round-robin over the sample_id order: deterministic, disjoint, sizes differ by <= 1."""
    df = df.sort_values("sample_id", ignore_index=True)
    return [df.iloc[i::n].reset_index(drop=True) for i in range(n)]


def code_fingerprint(root: Path, paths: list[str] = FINGERPRINT_PATHS) -> dict[str, str]:
    """SHA-256 per file (Python sources, configs, frozen split, label mapping)."""
    out = {}
    for p in paths:
        q = root / p
        files = sorted(x for x in q.rglob("*.py")) if q.is_dir() else [q]
        for f in files:
            out[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def host_info() -> dict:
    import os
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpus": os.cpu_count(), "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:12]}


class _HFAuth(urllib.request.BaseHandler):
    def __init__(self, token: str) -> None:
        self.token = token

    def https_request(self, req):
        if req.host == "huggingface.co":  # never forwarded to redirect targets
            req.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return req


def install_hf_auth(token: str | None) -> bool:
    """Make urllib send the token to huggingface.co only. Returns whether a token is used."""
    if not token:
        return False
    urllib.request.install_opener(urllib.request.build_opener(_HFAuth(token)))
    return True
