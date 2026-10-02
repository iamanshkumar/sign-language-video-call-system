"""Step 1.2a: select the reproducible class-balanced sample for Representative Sample EDA."""
from __future__ import annotations

import json

import pandas as pd

from asl.config import PROJECT_ROOT, load_config
from asl.data.sampling import select_sample


def main() -> None:
    cfg = load_config()
    es = cfg["eda_sample"]
    out = PROJECT_ROOT / es["dir"]
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_parquet(cfg.path("metadata") / "manifest.parquet")
    pool = m[m["eligible_pre_probe"]]
    sample, table = select_sample(pool, seed=es["seed"], per_class=es["per_class"], target_total=es["target_total"])

    cols = ["sample_id", "repo_path", "word", "sha256", "size_bytes", "source_proxy", "dup_group_size", "dup_cross_label"]
    sample[cols].to_csv(out / "sample.csv", index=False)
    table.to_csv(out / "sample_class_counts.csv")
    summary = {
        "seed": es["seed"], "per_class": es["per_class"], "target_total": es["target_total"],
        "pool_files": int(len(pool)), "sampled_files": int(len(sample)),
        "classes_in_pool": int(pool["word"].nunique()), "classes_sampled": int(sample["word"].nunique()),
        "samples_per_class_counts": table["sampled"].value_counts().sort_index().to_dict(),
        "selection_counts": table["selection"].value_counts().to_dict(),
        "sampled_in_duplicate_groups": int((sample["dup_group_size"] > 1).sum()),
        "sampled_in_cross_label_groups": int(sample["dup_cross_label"].sum()),
        "sampled_distinct_sha256": int(sample["sha256"].nunique()),
        "source_proxy_sample": sample["source_proxy"].value_counts().to_dict(),
        "source_proxy_pool": pool["source_proxy"].value_counts().to_dict(),
    }
    (out / "sample_summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
