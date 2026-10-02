"""Step 3.3: end-to-end sequence preparation check + report (no training, no MediaPipe).

For every configuration x window x split of a store: build the Dataset, generate every
sequence twice (bit-identical?), check shapes / dtype / finiteness / presence flags,
iterate the DataLoaders (batch shapes, deterministic order), and write
reports/step3/sequence_preparation_report.md + sequence_preparation_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.layout import GROUPS, feature_dim
from asl.sequences.dataset import SequenceDataset, make_dataloaders
from asl.sequences.index import INTERVAL_BINS, SPLITS, check_index_leakage
from asl.sequences.labels import load_label_mapping

STEP3_CONFIG = PROJECT_ROOT / "configs" / "step3_sequences.yaml"


def presence_columns(groups: list[str]) -> list[int]:
    cols, off = [], 0
    for g in groups:
        off += GROUPS[g] * 3
        cols.append(off)
        off += 1 + (GROUPS[g] if g == "pose" else 0)
    return cols


def main() -> None:
    c2, c3 = load_yaml(STEP2_CONFIG), load_yaml(STEP3_CONFIG)
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="benchmark_v2_mixed", choices=sorted(c3["stores"]))
    args = ap.parse_args()
    store = c3["stores"][args.store]
    arrays_dir = PROJECT_ROOT / store["landmarks_dir"] / "arrays"
    idx_dir = PROJECT_ROOT / store["index_dir"]
    rep = PROJECT_ROOT / "reports" / "step3"
    rep.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(idx_dir / "index.parquet")
    excluded = pd.read_csv(idx_dir / "excluded.csv", keep_default_na=False)
    isum = json.loads((idx_dir / "index_summary.json").read_text())
    mapping = load_label_mapping(PROJECT_ROOT / c3["labels"]["mapping_path"])
    configs, windows, dl = c2["configurations"], c3["sequences"]["windows"], c3["dataloader"]

    checks: dict[str, bool] = {"index_leakage_free": not check_index_leakage(idx)}
    gen: dict = {}
    t0 = time.perf_counter()
    for cname in configs:
        for w in windows:
            for split in SPLITS:
                ds = SequenceDataset(idx, arrays_dir, split, w, cname, configs, mapping)
                key = f"{cname}/{w}/{split}"
                if len(ds) == 0:
                    gen[key] = {"sequences": 0}
                    continue
                h1, h2, ok_shape, ok_presence, min_v, max_v = [], [], True, True, np.inf, -np.inf
                pcols = presence_columns(ds.groups)
                for i in range(len(ds)):
                    x, y = ds[i]
                    x2, _ = ds[i]
                    a = x.numpy()
                    ok_shape &= x.dtype == torch.float32 and tuple(x.shape) == (w, feature_dim(ds.groups)) \
                        and y.dtype == torch.long and bool(torch.isfinite(x).all())
                    ok_presence &= bool(np.isin(a[:, pcols], (0.0, 1.0)).all())
                    h1.append(hashlib.sha256(a.tobytes()).hexdigest())
                    h2.append(hashlib.sha256(x2.numpy().tobytes()).hexdigest())
                    min_v, max_v = min(min_v, float(a.min())), max(max_v, float(a.max()))
                gen[key] = {"sequences": len(ds), "feature_dim": ds.feature_dim, "shape_ok": bool(ok_shape),
                            "presence_flags_binary": bool(ok_presence), "regeneration_identical": h1 == h2,
                            "value_range": [min_v, max_v],
                            "sequence_set_sha256": hashlib.sha256("".join(h1).encode()).hexdigest()}
    gen_s = time.perf_counter() - t0
    checks["all_shapes_ok"] = all(v.get("shape_ok", True) for v in gen.values())
    checks["presence_flags_binary"] = all(v.get("presence_flags_binary", True) for v in gen.values())
    checks["regeneration_identical"] = all(v.get("regeneration_identical", True) for v in gen.values())
    # identical generated sequences must not appear in different splits
    by_hash: dict[str, set] = {}
    for cname in configs:
        for w in windows:
            for split in SPLITS:
                ds = SequenceDataset(idx, arrays_dir, split, w, cname, configs, mapping)
                for i in range(len(ds)):
                    by_hash.setdefault(hashlib.sha256(ds[i][0].numpy().tobytes()).hexdigest(), set()).add(split)
    checks["no_identical_sequence_across_splits"] = all(len(s) == 1 for s in by_hash.values())

    # DataLoaders (largest configuration, every window): batch shapes + deterministic order
    loader_info = {}
    for w in windows:
        dsets = {s: SequenceDataset(idx, arrays_dir, s, w, "hands_pose_face", configs, mapping) for s in SPLITS}
        dsets = {s: d for s, d in dsets.items() if len(d)}
        order = []
        for _ in range(2):
            loaders = make_dataloaders(dsets, dl["batch_size"], dl["num_workers"], dl["seed"], dl["drop_last_train"])
            xb, yb = next(iter(loaders["train"]))
            order.append(yb.tolist())
        loader_info[w] = {"train_batches": len(loaders["train"]), "first_batch_x": list(xb.shape),
                          "first_batch_y": list(yb.shape), "x_dtype": str(xb.dtype), "y_dtype": str(yb.dtype),
                          "seeded_shuffle_reproducible": order[0] == order[1]}
    checks["dataloader_shuffle_reproducible"] = all(v["seeded_shuffle_reproducible"] for v in loader_info.values())
    checks["batch_size_from_config"] = all(v["first_batch_x"][0] == min(dl["batch_size"], len(idx[idx.split == "train"]))
                                           for v in loader_info.values())

    # counts
    bins = {name: int((idx["active_bin"] == name).sum()) for _, _, name in INTERVAL_BINS}
    by_split = {s: {"videos": int((idx["split"] == s).sum()), "classes": int(idx.loc[idx["split"] == s, "word"].nunique()),
                    "short_active": int(idx.loc[idx["split"] == s, "short_active_interval"].sum())} for s in SPLITS}
    per_label = idx.groupby("word").size()
    stretch = {w: {"upsampled": int((idx[f"stretch_{w}"] > 1).sum()), "identity": int((idx[f"stretch_{w}"] == 1).sum()),
                   "downsampled": int((idx[f"stretch_{w}"] < 1).sum()), "max_stretch": float(idx[f"stretch_{w}"].max()),
                   "short_interval_median_stretch": float(idx.loc[idx["short_active_interval"], f"stretch_{w}"].median())
                   if idx["short_active_interval"].any() else None}
               for w in windows}
    per_label_table = idx.groupby("word").agg(videos=("sample_id", "size"), label=("label", "first"),
                                              short=("short_active_interval", "sum")).sort_values("label")
    per_label_table.to_csv(rep / "sequences_per_label.csv")
    summary = {
        "store": args.store, "index_sha256": isum["index_sha256"],
        "label_mapping": {"num_classes": mapping["num_classes"], "vocabulary_sha256": mapping["vocabulary_sha256"],
                          "frozen_split_sha256": mapping["frozen_split_sha256"]},
        "landmark_videos": isum["landmark_videos"], "accepted_videos": int(len(idx)),
        "excluded_videos": int(len(excluded)), "excluded_by_reason": isum["excluded_by_reason"],
        "excluded_by_split": isum["excluded_by_split"], "active_interval_bins": bins,
        "by_split": by_split, "windows": windows,
        "sequences_per_window": {w: int(len(idx)) for w in windows},
        "unable_to_produce_window": {w: int((idx["active_frames"] < 2).sum()) for w in windows},
        "resampling": stretch, "labels_with_sequences": int(per_label.size),
        "sequences_per_label": {"min": int(per_label.min()), "median": float(per_label.median()), "max": int(per_label.max())},
        "generation": gen, "generation_seconds": gen_s, "dataloaders": loader_info, "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    (rep / "sequence_preparation_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_report(rep / "sequence_preparation_report.md", summary, configs)
    print(json.dumps({k: summary[k] for k in ("accepted_videos", "excluded_videos", "active_interval_bins", "by_split",
                                               "sequences_per_window", "unable_to_produce_window", "resampling",
                                               "dataloaders", "checks", "all_checks_pass", "generation_seconds")},
                     indent=2, default=str))


def write_report(path, s: dict, configs: dict) -> None:
    L = ["# Step 3 - Sequence preparation report\n",
         f"Store: `{s['store']}` (Phase A benchmark extractions converted to the approved v2 mixed-precision format, "
         "`scripts/s2_06_convert_benchmark_to_v2.py`). Generated by `scripts/s3_03_sequence_report.py`; "
         "machine-readable: `reports/step3/sequence_preparation_summary.json`. Methodology: `docs/step3_methodology.md`.\n",
         "> **Scope:** the benchmark contains only TRAIN-split videos (by Step 2 design), so validation/test "
         "Datasets are empty here; val/test behaviour is covered by the synthetic-fixture tests "
         "(`tests/test_sequences.py`). Counts below are benchmark counts, not dataset counts.\n",
         "## Methodology (summary)\n",
         "- Content region: the Step 2 **active signing interval** `[start, end]` (inclusive, 30 FPS grid); rest frames are not used.",
         "- One sequence per video × window × configuration: the active interval is **uniformly resampled** to exactly "
         "W frames, positions `p_k = start + k (L-1)/(W-1)`; linear interpolation per group when present in both "
         "neighbouring frames, otherwise nearest frame (value + presence). No padding, no random selection, no frame copying.",
         "- Short active intervals (< 30 frames) are kept and upsampled (temporal resampling, not duplication).",
         "- Rejected Step 2 videos are excluded (no arrays) and listed below; they remain in the frozen split.",
         "- Labels: one mapping built from the frozen split vocabulary (sorted by code point), shared by all splits.",
         "- Storage: no sequence copies; `index.parquet` + lazy generation from the landmark `.npz` files.",
         "- Splits: inherited from the frozen split file; the index build fails on any split/group leakage.\n",
         "## Label mapping\n",
         f"{s['label_mapping']['num_classes']} classes; vocabulary SHA-256 `{s['label_mapping']['vocabulary_sha256'][:16]}…`; "
         f"frozen split SHA-256 `{s['label_mapping']['frozen_split_sha256'][:16]}…` (`data/sequences/label_mapping.json`).\n",
         "## Samples\n",
         f"- Landmark videos: {s['landmark_videos']}; accepted (indexed): **{s['accepted_videos']}**; excluded: "
         f"{s['excluded_videos']} {s['excluded_by_reason']} (by split: {s['excluded_by_split']}).",
         "\n| active interval | videos |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["active_interval_bins"].items()]
    L += ["\n| split | videos | classes | short active |", "|---|---|---|---|"]
    L += [f"| {k} | {v['videos']} | {v['classes']} | {v['short_active']} |" for k, v in s["by_split"].items()]
    L += ["\n| window | sequences per configuration | unable to produce | upsampled | identity | downsampled | "
          "max stretch | median stretch (short intervals) |", "|---|---|---|---|---|---|---|---|"]
    for w in s["windows"]:
        r = s["resampling"][w]
        L.append(f"| {w} | {s['sequences_per_window'][w]} | {s['unable_to_produce_window'][w]} | {r['upsampled']} | "
                 f"{r['identity']} | {r['downsampled']} | {r['max_stretch']:.2f}× | {r['short_interval_median_stretch']:.2f}× |")
    L += [f"\nLabels with ≥1 sequence: {s['labels_with_sequences']} (per label min {s['sequences_per_label']['min']}, "
          f"median {s['sequences_per_label']['median']}, max {s['sequences_per_label']['max']}); per-label table: "
          "`reports/step3/sequences_per_label.csv`.\n",
          "## Feature configurations × windows (generated and checked)\n",
          "| configuration | window | split | sequences | F | shape ok | presence binary | regeneration identical |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in s["generation"].items():
        c, w, sp = k.split("/")
        if v["sequences"]:
            L.append(f"| {c} | {w} | {sp} | {v['sequences']} | {v['feature_dim']} | {v['shape_ok']} | "
                     f"{v['presence_flags_binary']} | {v['regeneration_identical']} |")
        else:
            L.append(f"| {c} | {w} | {sp} | 0 | – | – | – | – |")
    L += [f"\nGeneration time for all of the above (each sequence twice): {s['generation_seconds']:.1f} s.\n",
          "## DataLoaders (hands_pose_face)\n", "| window | train batches | first batch x | y | dtypes | seeded shuffle reproducible |",
          "|---|---|---|---|---|---|"]
    for w, v in s["dataloaders"].items():
        L.append(f"| {w} | {v['train_batches']} | {v['first_batch_x']} | {v['first_batch_y']} | {v['x_dtype']}, "
                 f"{v['y_dtype']} | {v['seeded_shuffle_reproducible']} |")
    L += ["\n## Checks\n", "| check | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in s["checks"].items()]
    L += ["\n## Limitations and open decisions\n",
          "1. **Time normalization:** each sequence spans the whole active interval, so W is a temporal resolution, "
          "not a real-time duration; this differs from planned real-time streaming windows (needs approval; "
          "alternative in `docs/step3_methodology.md` §8).",
          "2. **Short intervals** are upsampled up to "
          f"{max(v['max_stretch'] for v in s['resampling'].values()):.0f}× (60-frame window); most of their frames are "
          "interpolated. Kept as decided in Step 2.",
          "3. Benchmark only: TRAIN split, ~1 video per class; dataset-level counts need the full extraction.",
          "4. The benchmark store was converted from Phase A v1 extractions (same MediaPipe output, re-packed), not "
          "produced by a fresh v2 extraction run."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
