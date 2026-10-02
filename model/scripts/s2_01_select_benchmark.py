"""Step 2.1: select the reproducible benchmark videos from the FROZEN train split."""
from __future__ import annotations

import json

import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.landmarks.benchmark import select_benchmark


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    b = c2["benchmark"]
    out = PROJECT_ROOT / b["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "benchmark_sample.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; the benchmark sample is fixed. Delete it explicitly to reselect.")

    split = load_frozen_split(c1.path("splits"), c1["split"]["version"])  # checksum-verified, read-only
    probed = pd.read_parquet(PROJECT_ROOT / c1["eda_sample"]["dir"] / "sample_validated.parquet")
    probed = probed[probed["decode_ok"].fillna(False).astype(bool)]
    sel = select_benchmark(split, probed, seed=b["seed"], n=b["n_videos"], quotas=b["quotas"], which=b["split"])
    sel.to_csv(path, index=False)
    summary = {
        "seed": b["seed"], "videos": int(len(sel)), "classes": int(sel["word"].nunique()),
        "split": b["split"], "by_reason": sel["benchmark_reason"].value_counts().to_dict(),
        "decoded_frames_known": int(sel["decoded_frames"].notna().sum()),
        "decoded_frames": sel["decoded_frames"].describe().round(1).to_dict(),
        "resolutions": sel["resolution"].value_counts().to_dict(),
        "in_duplicate_groups": int((sel["dup_group_size"].astype(int) > 1).sum()),
        "augmented": int(sel["aug_source"].fillna("").ne("").sum()),
    }
    (out / "benchmark_sample_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
