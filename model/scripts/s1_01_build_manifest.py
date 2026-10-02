"""Step 1.1: index the repo, reconcile it with the CSV, audit duplicate content and labels.

No videos are downloaded here; everything comes from Hugging Face file metadata.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from asl.config import load_config
from asl.data import hf_index, labels, manifest as mf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-index", action="store_true", help="re-list the repo tree")
    args = ap.parse_args()

    cfg = load_config()
    ds, meta_dir = cfg["dataset"], cfg.path("metadata")

    csv_path = hf_index.download_csv(ds["repo_id"], ds["revision"], ds["csv_filename"], meta_dir / "hf")
    index_path = meta_dir / "repo_index.parquet"
    if args.refresh_index or not index_path.exists():
        repo = hf_index.list_video_files(ds["repo_id"], ds["revision"], ds["video_dirs"])
        repo.to_parquet(index_path, index=False)
    repo = pd.read_parquet(index_path)

    csv_df = mf.read_metadata_csv(csv_path)
    m = mf.annotate_duplicates(mf.build_manifest(csv_df, repo))
    m.to_parquet(meta_dir / "manifest.parquet", index=False)

    groups = mf.duplicate_content_groups(m)
    groups.to_csv(meta_dir / "duplicate_content_groups.csv", index=False)

    elig = m[m["eligible_pre_probe"]]
    audit = labels.audit_labels(elig["word"])
    audit.to_csv(meta_dir / "label_audit.csv", index=False)

    ext = repo["filename"].str.rsplit(".", n=1).str[-1].str.lower().value_counts().to_dict()
    summary = {
        "revision": ds["revision"],
        "csv_rows": int(len(csv_df)),
        "csv_unique_words": int(csv_df["word"].nunique()),
        "repo_video_files": int(len(repo)),
        "repo_files_per_folder": repo["folder"].value_counts().sort_index().to_dict(),
        "repo_total_bytes": int(repo["size_bytes"].sum()),
        "repo_extensions": ext,
        "repo_files_missing_sha256": int(repo["sha256"].isna().sum()),
        "issues": m.loc[m["issue"] != "", "issue"].value_counts().to_dict(),
        "issue_rows": m.loc[m["issue"] != "", ["csv_row", "word", "filename", "issue"]].to_dict("records"),
        "eligible_pre_probe_samples": int(len(elig)),
        "eligible_pre_probe_classes": int(elig["word"].nunique()),
        "duplicate_content": {
            "groups": int(len(groups)),
            "files_in_groups": int(groups["n_files"].sum()) if len(groups) else 0,
            "redundant_copies": int((groups["n_files"] - 1).sum()) if len(groups) else 0,
            "cross_label_groups": int((groups["n_labels"] > 1).sum()) if len(groups) else 0,
            "files_in_cross_label_groups": int(groups.loc[groups["n_labels"] > 1, "n_files"].sum()) if len(groups) else 0,
        },
        "source_proxy_counts": elig["source_proxy"].value_counts().to_dict(),
        "label_anomaly_counts": audit.groupby("anomaly")["label"].nunique().to_dict() if len(audit) else {},
    }
    (meta_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
