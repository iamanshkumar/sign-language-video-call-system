"""Index the Hugging Face dataset repository at a pinned revision.

The Hugging Face dataset viewer / `datasets.load_dataset()` misinterprets this
repository (it reports ~47 rows), so we never use it. Instead we list the raw
repository tree, which gives every file's path, size and LFS SHA-256 without
downloading any video.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


def list_video_files(repo_id: str, revision: str, video_dirs: list[str]) -> pd.DataFrame:
    """Return one row per file under `video_dirs`: repo_path, folder, filename, size, sha256."""
    api = HfApi()
    rows = []
    for folder in video_dirs:
        for entry in api.list_repo_tree(
            repo_id, repo_type="dataset", path_in_repo=folder, revision=revision, recursive=True
        ):
            if not hasattr(entry, "size"):  # RepoFolder
                continue
            lfs = getattr(entry, "lfs", None)
            rows.append(
                {
                    "repo_path": entry.path,
                    "folder": folder,
                    "filename": entry.path.split("/", 1)[1],
                    "size_bytes": int(entry.size),
                    # Every video is stored in LFS; a missing hash is recorded, not guessed.
                    "sha256": lfs.sha256 if lfs is not None else None,
                }
            )
    return pd.DataFrame(rows).sort_values("repo_path", ignore_index=True)


def download_csv(repo_id: str, revision: str, csv_filename: str, out_dir: Path) -> Path:
    """Download the metadata CSV at the pinned revision into `out_dir`."""
    return Path(
        hf_hub_download(
            repo_id, csv_filename, repo_type="dataset", revision=revision, local_dir=out_dir
        )
    )
