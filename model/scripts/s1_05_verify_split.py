"""Step 1.5: verify the frozen split (integrity, leakage, reproducibility, stratification)."""
from __future__ import annotations

import json
import sys

import pandas as pd

from asl.config import load_config
from asl.data.leakage import run_checks, split_class_ratio_spread
import hashlib

from asl.data.groups import build_split_groups
from asl.data.split import SPLITS, assign_splits, expected_classes, load_frozen_split


def main() -> None:
    cfg = load_config()
    sp, rep = cfg["split"], cfg.path("reports")
    frozen = load_frozen_split(cfg.path("splits"), sp["version"])  # verifies checksum
    m = pd.read_parquet(cfg.path("metadata") / "manifest.parquet")
    elig = build_split_groups(m[m["eligible_pre_probe"]])
    ratios = {s: sp[s] for s in SPLITS}

    expected = expected_classes(elig)
    infeasible = sorted(set(elig["word"]) - expected)
    checks = run_checks(frozen, set(elig["sample_id"]), ratios, expected=expected)
    checks.insert(0, ("split file checksum matches lock", True, "verified on load"))

    splits_dir = cfg.path("splits")
    lock = json.loads((splits_dir / f"split_{sp['version']}.lock.json").read_text())
    cc_path = splits_dir / lock["class_counts_file"]
    cc_ok = hashlib.sha256(cc_path.read_bytes()).hexdigest() == lock["class_counts_sha256"]
    checks.insert(1, ("class-count file checksum matches lock", cc_ok, ""))
    cc = pd.read_csv(cc_path, index_col=0, keep_default_na=False)
    live = pd.crosstab(frozen["word"], frozen["split"]).reindex(columns=list(SPLITS), fill_value=0)
    checks.append(("class-count file matches split", bool(cc.sort_index().equals(live.sort_index())), ""))
    for k, f in lock["per_split_files"].items():
        pth = splits_dir / f["file"]
        part = pd.read_csv(pth, dtype=str, keep_default_na=False)
        ok = (hashlib.sha256(pth.read_bytes()).hexdigest() == f["sha256"]
              and set(part["sample_id"]) == set(frozen.loc[frozen["split"] == k, "sample_id"]))
        checks.append((f"{k} file checksum matches lock and equals {k} rows of split", ok, f["file"]))
    per_path = frozen.groupby("repo_path")["split"].nunique()
    checks.append(("no path assigned to multiple splits", bool((per_path == 1).all()), ""))
    g = frozen.merge(elig[["sample_id", "split_group"]], on="sample_id", suffixes=("", "_recomputed"))
    checks.append(("split groups identical to recomputed grouping",
                   bool((g["split_group"] == g["split_group_recomputed"]).all() and len(g) == len(frozen)), ""))
    checks.append(("lock records dataset revision and seed",
                   lock.get("dataset_revision") == cfg["dataset"]["revision"] and lock.get("seed") == sp["seed"], ""))

    # Reproducibility: regenerating in memory with the same seed must give the identical assignment.
    again = assign_splits(elig, seed=sp["seed"], val=sp["val"], test=sp["test"])
    same = again.set_index("sample_id")["split"].sort_index().equals(
        frozen.set_index("sample_id")["split"].sort_index()
    )
    checks.append(("deterministic regeneration matches frozen split", same, ""))

    # Label consistency with the validated manifest.
    lab = frozen.merge(elig[["sample_id", "word"]], on="sample_id", suffixes=("", "_m"))
    ok = len(lab) == len(frozen) and (lab["word"] == lab["word_m"]).all()
    checks.append(("labels identical to manifest", bool(ok), ""))

    counts = pd.crosstab(frozen["word"], frozen["split"]).reindex(columns=list(SPLITS))
    spread = split_class_ratio_spread(frozen)
    summary = {
        "checks": [{"check": c, "passed": bool(p), "detail": d} for c, p, d in checks],
        "all_passed": all(p for _, p, _ in checks),
        "counts": frozen["split"].value_counts().reindex(SPLITS).astype(int).to_dict(),
        "fractions": (frozen["split"].value_counts(normalize=True).reindex(SPLITS)).round(4).to_dict(),
        "classes": int(frozen["word"].nunique()),
        "classes_per_split": {k: int(frozen.loc[frozen["split"] == k, "word"].nunique()) for k in SPLITS},
        "classes_expected_in_all_splits": len(expected),
        "classes_infeasible_for_all_splits": infeasible,
        "per_class_min_samples": counts.min().astype(int).to_dict(),
        "per_class_median_samples": counts.median().to_dict(),
        "per_class_train_fraction_range": [float(spread["train"].min()), float(spread["train"].max())],
        "source_proxy_by_split": pd.crosstab(frozen["source_proxy"], frozen["split"])
        .reindex(columns=list(SPLITS)).to_dict(orient="index"),
        "limitation": "No signer IDs: the split is not signer-independent.",
    }
    (rep / "split_verification.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    counts.to_csv(rep / "tables" / "split_class_counts.csv")

    for c, p, d in checks:
        print(f"[{'PASS' if p else 'FAIL'}] {c} {('- ' + d) if d else ''}")
    print(json.dumps({k: summary[k] for k in ("counts", "fractions", "classes", "classes_per_split", "classes_infeasible_for_all_splits", "per_class_min_samples")}, indent=2))
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
