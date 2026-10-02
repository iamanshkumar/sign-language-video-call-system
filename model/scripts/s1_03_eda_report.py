"""Step 1.3: Representative Sample EDA.

Exact statistics come from the complete metadata manifest (all 108,616 files):
class distribution, duplicate content, labels, filename conventions, file sizes.
Video statistics (FPS, duration, frames, resolution, codec, decodability) come
ONLY from the probed class-balanced sample and are labelled as sample estimates.

Outputs:
  data/metadata/eda_sample/sample_validated.parquet
  reports/step1/stats.json      every number quoted in the report
  reports/step1/eda_report.md   human-readable report
  reports/step1/figures/*.png, reports/step1/tables/*.csv
"""
from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, load_config
from asl.data.validate import validate

BAR = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WINDOWS = (30, 45, 60)


# ---------------------------------------------------------------- plotting
def _axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _hist(values, bins, title, xlabel, path, vlines=()):
    fig, ax = _axes(title, xlabel, "videos")
    ax.hist(values, bins=bins, color=BAR, edgecolor="white", linewidth=0.5)
    for x in vlines:
        ax.axvline(x, color=INK2, linestyle="--", linewidth=1)
        ax.text(x, ax.get_ylim()[1] * 0.97, f" {x}", color=INK2, va="top", fontsize=9)
    _save(fig, path)


def _barh(labels, values, title, xlabel, path):
    fig, ax = _axes(title, xlabel, "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=BAR, height=0.7)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:,}", va="center", fontsize=8, color=INK2)
    _save(fig, path)


# ---------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(x):
        if isinstance(x, float):
            return floatfmt.format(x)
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        return str(x)

    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def describe(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    if s.empty:
        return {}
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(s.size), "mean": float(s.mean()), "std": float(s.std()),
        "min": float(s.min()), "p01": float(q[0.01]), "p05": float(q[0.05]), "p25": float(q[0.25]),
        "median": float(q[0.5]), "p75": float(q[0.75]), "p95": float(q[0.95]), "p99": float(q[0.99]),
        "max": float(s.max()),
    }


def class_overlap(elig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classes that share byte-identical videos with other classes."""
    groups = elig.groupby("sha256")["word"].agg(lambda s: sorted(set(s)))
    cross = groups[groups.map(len) > 1]
    pairs: Counter = Counter()
    for labels in cross:
        pairs.update(itertools.combinations(labels, 2))
    pair_df = pd.DataFrame(
        [(a, b, n) for (a, b), n in pairs.most_common()], columns=["label_a", "label_b", "shared_videos"]
    )
    in_cross = elig["sha256"].isin(cross.index)
    per_class = pd.DataFrame(
        {"n_samples": elig.groupby("word").size(), "n_shared_with_other_class": elig[in_cross].groupby("word").size()}
    ).fillna(0).astype(int)
    per_class["shared_fraction"] = per_class["n_shared_with_other_class"] / per_class["n_samples"]
    return per_class.sort_values("shared_fraction", ascending=False), pair_df


AUG = r"_aug_\d+\.mp4$"


def augmentation_stats(full: pd.DataFrame) -> dict:
    """Files named `<source>_<word>_aug_<n>.mp4`: augmented copies of another file."""
    a = full[full["filename"].str.contains(AUG, case=False, regex=True)]
    stem = [re.sub(AUG, "", f, flags=re.I) for f in a["filename"]]
    src = [re.sub("_" + re.escape(w) + "$", "", st, flags=re.I) + ".mp4" for st, w in zip(stem, a["word"])]
    names = set(full["filename"])
    in_cls = full[full["word"].isin(set(a["word"]))]
    return {
        "files": int(len(a)),
        "classes": int(a["word"].nunique()),
        "classes_list": sorted(a["word"].unique().tolist()),
        "source_file_identifiable": int(sum(x in names for x in src)),
        "share_of_files_in_those_classes": float(len(a) / max(len(in_cls), 1)),
    }


# ---------------------------------------------------------------- main
def wprop(mask: pd.Series, w: pd.Series) -> float:
    """Class-size-weighted proportion (inverse sampling probability)."""
    return float((w * mask.astype(float)).sum() / w.sum())


def main() -> None:
    cfg = load_config()
    meta, rep = cfg.path("metadata"), cfg.path("reports")
    sdir = PROJECT_ROOT / cfg["eda_sample"]["dir"]
    fig_dir, tab_dir = rep / "figures", rep / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)
    min_frames = cfg["validation"]["min_decoded_frames"]

    manifest = pd.read_parquet(meta / "manifest.parquet")
    summary = json.loads((meta / "manifest_summary.json").read_text())
    sample_summary = json.loads((sdir / "sample_summary.json").read_text())
    sample = pd.read_csv(sdir / "sample.csv", dtype=str, keep_default_na=False)
    sample_counts = pd.read_csv(sdir / "sample_class_counts.csv", keep_default_na=False, index_col=0)
    probe = pd.read_parquet(sdir / "sample_probe_results.parquet")
    audit = pd.read_csv(meta / "label_audit.csv", keep_default_na=False)

    full = manifest[manifest["eligible_pre_probe"]]
    s_man = manifest[manifest["sample_id"].isin(set(sample["sample_id"]))]
    v = validate(s_man, probe, min_frames)
    v["weight"] = v["word"].map(sample_counts["class_size"] / sample_counts["sampled"])
    v.to_parquet(sdir / "sample_validated.parquet", index=False)

    unprobed = int((v["exclusion_reason"] == "not_probed").sum())
    dl_failed = int((v["exclusion_reason"] == "download_failed").sum())
    ok_dl = v[v["download_ok"].fillna(False).astype(bool)]
    dec = v[v["decode_ok"].fillna(False).astype(bool)].copy()
    w = dec["weight"]

    # --- derived per-video quantities (sample)
    dec["duration_s"] = dec["decoded_frames"] / dec["decoded_fps"].fillna(dec["avg_fps"])
    dec["frames_at_30fps"] = np.floor(dec["duration_s"] * 30 + 1e-6)
    dec["resolution"] = dec["width"].astype("Int64").astype(str) + "x" + dec["height"].astype("Int64").astype(str)
    fps_invalid = ~np.isfinite(dec["avg_fps"].astype(float)) | (dec["avg_fps"].fillna(0) <= 0)
    ts_fps_invalid = ~np.isfinite(dec["decoded_fps"].astype(float)) | (dec["decoded_fps"].fillna(0) <= 0)
    fps = dec["decoded_fps"].fillna(dec["avg_fps"])  # timestamp-derived FPS is what resampling sees
    up, down = fps < 29.5, fps > 30.5
    vfr = (dec["avg_fps"] - dec["decoded_fps"]).abs() > 0.5
    frame_mismatch = dec["reported_frames"].notna() & (dec["reported_frames"] != dec["decoded_frames"])
    decode_fail = ok_dl["sha256_match"].fillna(False).astype(bool) & ~ok_dl["decode_ok"].fillna(False).astype(bool)

    # --- exact metadata statistics (full manifest)
    counts_all = full.groupby("word").size()
    per_class, pair_df = class_overlap(full)
    per_class.to_csv(tab_dir / "class_content_overlap.csv")
    pair_df.to_csv(tab_dir / "shared_content_label_pairs.csv", index=False)
    counts_all.rename("n_files").to_csv(tab_dir / "class_counts_full.csv")
    v.loc[~v["eligible"], ["word", "repo_path", "exclusion_reason", "decode_error", "decoded_frames", "avg_fps"]] \
        .to_csv(tab_dir / "sample_rejected_videos.csv", index=False)
    dec[["repo_path", "word", "avg_fps", "decoded_fps", "decoded_frames", "duration_s", "frames_at_30fps",
         "resolution", "codec", "pix_fmt", "has_audio", "weight"]].to_csv(tab_dir / "sample_video_properties.csv", index=False)

    def below(col, wn):
        m = dec[col] < wn
        return {"count": int(m.sum()), "sample_pct": 100 * float(m.mean()), "weighted_est_pct": 100 * wprop(m, w)}

    stats = {
        "full_metadata": {
            "manifest": summary,
            "files": int(len(full)),
            "classes": int(full["word"].nunique()),
            "class_counts": describe(counts_all),
            "imbalance_ratio_max_min": float(counts_all.max() / counts_all.min()),
            "file_size_bytes": describe(full["size_bytes"]),
            "files_under_10kb": int((full["size_bytes"] < 10_000).sum()),
            "metadata_checks": {
                "csv_rows_with_empty_word_or_filename": int((manifest["issue"] == "empty_word_or_filename").sum()),
                "repo_files_missing_sha256": int(summary["repo_files_missing_sha256"]),
                "csv_duplicate_filenames": int(manifest.loc[manifest["csv_row"] >= 0, "filename"].duplicated().sum()),
                "duplicate_repo_paths_among_eligible": int(full["repo_path"].duplicated().sum()),
                "filenames_in_multiple_folders": int((manifest["issue"] == "ambiguous_filename").sum()),
                "repo_files_not_in_csv": int((manifest["issue"] == "repo_file_not_in_csv").sum()),
                "files_listed_under_conflicting_labels": int((manifest["issue"] == "conflicting_label_for_file").sum()),
            },
            "content_overlap": {
                "classes_sharing_any_video_with_another_class": int((per_class["n_shared_with_other_class"] > 0).sum()),
                "classes_with_>=50pct_shared": int((per_class["shared_fraction"] >= 0.5).sum()),
                "classes_with_100pct_shared": int((per_class["shared_fraction"] >= 1.0).sum()),
                "label_pairs_sharing_videos": int(len(pair_df)),
            },
            "source_proxy": full["source_proxy"].value_counts().to_dict(),
            "augmented_files": augmentation_stats(full),
        },
        "sample": {
            "selection": sample_summary,
            "sampled": int(len(v)),
            "classes_represented": int(v["word"].nunique()),
            "unprobed": unprobed,
            "download_failed_after_retries": dl_failed,
            "downloaded": int(len(ok_dl)),
            "sha256_mismatch": int((~ok_dl["sha256_match"].fillna(False).astype(bool)).sum()),
            "decode_ok": int(len(dec)),
            "decode_failed": int(decode_fail.sum()),
            "probe_origin": v["probe_origin"].value_counts().to_dict(),
            "rejections_step1_rules": v.loc[~v["eligible"], "exclusion_reason"].value_counts().to_dict(),
            "avg_fps": describe(dec["avg_fps"]),
            "decoded_fps": describe(dec["decoded_fps"]),
            "fps_rounded_counts": {str(k): int(n) for k, n in fps.round().astype(int).value_counts().sort_index().items()},
            "invalid_container_fps": int(fps_invalid.sum()),
            "invalid_timestamp_fps": int(ts_fps_invalid.sum()),
            "needs_upsampling_fps_below_29.5": {"count": int(up.sum()), "sample_pct": 100 * float(up.mean()),
                                               "weighted_est_pct": 100 * wprop(up, w)},
            "needs_downsampling_fps_above_30.5": {"count": int(down.sum()), "sample_pct": 100 * float(down.mean()),
                                                 "weighted_est_pct": 100 * wprop(down, w)},
            "fps_within_0.5_of_30": int((~up & ~down).sum()),
            "variable_frame_rate_suspected": int(vfr.sum()),
            "duration_s": describe(dec["duration_s"]),
            "decoded_frames": describe(dec["decoded_frames"]),
            "frames_at_30fps": describe(dec["frames_at_30fps"]),
            "decoded_frames_below": {str(wn): below("decoded_frames", wn) for wn in WINDOWS},
            "frames_at_30fps_below": {str(wn): below("frames_at_30fps", wn) for wn in WINDOWS},
            "reported_vs_decoded_frame_mismatch": int(frame_mismatch.sum()),
            "resolution_counts": dec["resolution"].value_counts().head(15).to_dict(),
            "n_resolutions": int(dec["resolution"].nunique()),
            "codec_counts": dec["codec"].value_counts().to_dict(),
            "pix_fmt_counts": dec["pix_fmt"].value_counts().to_dict(),
            "container_counts": dec["container_format"].value_counts().to_dict(),
            "has_audio": int(dec["has_audio"].fillna(False).astype(bool).sum()),
            "rotation_metadata": int(dec["rotate"].notna().sum()),
            "multi_video_stream": int((dec["n_video_streams"] > 1).sum()),
            "decode_error_types": v.loc[v["decode_error"].fillna("") != "", "decode_error"].value_counts().to_dict(),
            "in_duplicate_groups": int((v["dup_group_size"] > 1).sum()),
            "in_cross_label_groups": int(v["dup_cross_label"].sum()),
        },
    }
    (rep / "stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")

    # --- figures (full metadata)
    fig, ax = _axes("Videos per class (full manifest, 2,207 classes)", "videos in class", "classes")
    ax.hist(counts_all, bins=range(0, int(counts_all.max()) + 5, 2), color=BAR, edgecolor="white", linewidth=0.5)
    _save(fig, fig_dir / "class_distribution.png")
    fig, ax = _axes("Classes ranked by size (full manifest)", "class rank", "videos")
    ax.plot(np.arange(1, len(counts_all) + 1), np.sort(counts_all.values)[::-1], color=BAR, linewidth=2)
    _save(fig, fig_dir / "class_rank.png")
    fig, ax = _axes("Share of each class's videos also labelled as another class (full manifest)",
                    "shared fraction", "classes")
    ax.hist(per_class["shared_fraction"], bins=np.linspace(0, 1, 21), color=BAR, edgecolor="white", linewidth=0.5)
    _save(fig, fig_dir / "class_content_overlap.png")
    sp = full["source_proxy"].value_counts()
    _barh(sp.index.tolist(), sp.values.tolist(), "Filename convention / source proxy (full manifest)", "videos",
          fig_dir / "source_proxy.png")
    _hist(full["size_bytes"].astype(float) / 1e6, 100, "File size (full manifest)", "MB", fig_dir / "file_size_hist.png")

    # --- figures (sample)
    n = len(dec)
    _hist(fps.clip(upper=65), np.arange(0, 66, 1), f"FPS from frame timestamps (sample, n={n:,})",
          "frames per second", fig_dir / "fps_hist.png", (30,))
    _hist(dec["duration_s"].clip(upper=10), np.arange(0, 10.1, 0.1), f"Clip duration, clipped at 10 s (sample, n={n:,})",
          "seconds", fig_dir / "duration_hist.png", (1, 1.5, 2))
    _hist(dec["decoded_frames"].clip(upper=300), np.arange(0, 304, 3),
          f"Decoded frames per video, clipped at 300 (sample, n={n:,})", "decoded frames",
          fig_dir / "decoded_frames_hist.png", WINDOWS)
    _hist(dec["frames_at_30fps"].clip(upper=300), np.arange(0, 304, 3),
          f"Frames after 30 FPS standardisation, clipped at 300 (sample, n={n:,})", "frames at 30 FPS",
          fig_dir / "frames_at_30fps_hist.png", WINDOWS)
    res = dec["resolution"].value_counts().head(12)
    _barh(res.index.tolist(), res.values.tolist(), f"Most common resolutions (sample, n={n:,})", "videos",
          fig_dir / "resolutions.png")

    write_report(rep / "eda_report.md", stats, audit, pair_df, counts_all, v)
    print(json.dumps({k: stats["sample"][k] for k in (
        "sampled", "classes_represented", "unprobed", "download_failed_after_retries", "downloaded",
        "sha256_mismatch", "decode_ok", "decode_failed", "rejections_step1_rules", "probe_origin")}, indent=2))


def write_report(path, st, audit, pair_df, counts_all, v) -> None:
    F, S = st["full_metadata"], st["sample"]
    ms, sel = F["manifest"], S["selection"]
    L = []
    w = L.append
    w("# Step 1 - Dataset validation and Representative Sample EDA\n")
    w(f"Dataset: `akasheroor/American-Sign-Language-Dataset` @ `{ms['revision']}`. All numbers are generated by "
      "`scripts/s1_03_eda_report.py` (`reports/step1/stats.json`).\n")
    w("> **How to read this report.** Part I is **exact**: computed from the complete metadata manifest "
      "(CSV + repository file list + SHA-256 of every file). Part II is **Representative Sample EDA**: video "
      f"properties measured on a class-balanced sample of {S['sampled']:,} videos. Part II numbers are sample "
      "statistics / estimates, **not** exact statistics for the whole dataset.\n")
    if S["unprobed"] or S["download_failed_after_retries"]:
        w(f"> **Incomplete sample probe:** {S['unprobed']} unprobed, {S['download_failed_after_retries']} "
          "download failures.\n")

    w("# Part I - Complete metadata (exact)\n")
    w("## 1. Repository vs metadata CSV\n")
    w(md_table(pd.DataFrame([
        ("CSV rows", ms["csv_rows"]), ("Unique words in CSV", ms["csv_unique_words"]),
        ("Video files in repository", ms["repo_video_files"]),
        ("Total size (GB)", round(ms["repo_total_bytes"] / 1e9, 2)),
        ("Samples after reconciliation", F["files"]), ("Classes", F["classes"]),
    ], columns=["item", "value"]), floatfmt="{:.2f}"))
    w("\nREADME claims 108,618 videos / 2,208 words. The CSV columns are `word`, `videos` (bare filenames, no "
      "`part_N/` folder), not `word`, `video_path` as documented; files are resolved by filename.\n")
    w("Metadata checks:\n")
    w(md_table(pd.DataFrame(list(F["metadata_checks"].items()), columns=["check", "count"])))
    w("\nMetadata issues (rows excluded from the manifest):\n")
    w(md_table(pd.DataFrame(ms["issue_rows"])))

    w("\n## 2. Class distribution\n")
    c = F["class_counts"]
    w(md_table(pd.DataFrame([("videos per class", c["min"], c["p05"], c["median"], c["mean"], c["p95"], c["max"])],
                            columns=["", "min", "p5", "median", "mean", "p95", "max"]), floatfmt="{:.1f}"))
    w(f"\nImbalance ratio (largest / smallest class): **{F['imbalance_ratio_max_min']:.2f}**\n")
    top = counts_all.sort_values(ascending=False)
    w(md_table(pd.DataFrame({"largest": top.index[:10], "n": top.values[:10],
                             "smallest": top.index[::-1][:10], "n ": top.values[::-1][:10]})))

    w("\n## 3. Exact duplicate content (SHA-256)\n")
    dc, co = ms["duplicate_content"], F["content_overlap"]
    w(md_table(pd.DataFrame([
        ("Groups of byte-identical files", dc["groups"]), ("Files in those groups", dc["files_in_groups"]),
        ("Extra copies", dc["redundant_copies"]), ("Groups spanning >1 label", dc["cross_label_groups"]),
        ("Files in cross-label groups", dc["files_in_cross_label_groups"]),
        ("Classes sharing ≥1 identical video with another class", co["classes_sharing_any_video_with_another_class"]),
        ("Classes with ≥50% of videos shared", co["classes_with_>=50pct_shared"]),
        ("Classes with 100% of videos shared", co["classes_with_100pct_shared"]),
        ("Label pairs sharing videos", co["label_pairs_sharing_videos"]),
    ], columns=["item", "value"])))
    w("\nThe README says duplicates were removed; at the byte level they were not. Most duplicate groups carry "
      "*different* labels: the same clip is filed under synonym words (one clip under 20 labels). Labels are kept "
      "unchanged; such classes are partly or fully indistinguishable from video alone. Top label pairs:\n")
    w(md_table(pair_df.head(15)))
    w("\nIn the split, each SHA-256 group is one indivisible unit. Only byte-identical copies are detectable; "
      "re-encoded or trimmed near-duplicates are not.\n")

    w("## 4. Label audit (report only - no labels changed)\n")
    ac = audit.groupby("anomaly")["label"].nunique().reset_index(name="labels")
    w(md_table(ac))
    for kind in ("near_duplicate_label", "repeated_word", "classifier_suffix", "uppercase", "contains_digit",
                 "unusual_characters", "long_unseparated"):
        sub = audit[audit["anomaly"] == kind]
        if len(sub):
            w(f"\n`{kind}`: " + ", ".join(f"`{x}`" for x in sorted(sub["label"].unique())))
    sub = audit[audit["anomaly"] == "concatenation_of_labels"]
    w(f"\n`concatenation_of_labels` ({len(sub)} labels; heuristic, many are legitimate multi-word glosses) - e.g. "
      + ", ".join(f"`{r.label}` = {r.detail}" for r in sub.head(10).itertuples()))
    w("\nFull list: `data/metadata/label_audit.csv`.\n")

    w("## 5. Filename conventions (source proxy)\n")
    w(md_table(pd.DataFrame(list(F["source_proxy"].items()), columns=["filename pattern", "files"])))
    w("\nThese patterns suggest different source collections. They are **not** signer IDs; no signer metadata "
      "exists, so a signer-independent split cannot be guaranteed.\n")
    ag = F["augmented_files"]
    w("### Augmented copies (near-duplicates not caught by SHA-256)\n")
    w(f"**{ag['files']:,}** files are named `<source>_<word>_aug_<n>.mp4` across **{ag['classes']}** classes "
      f"({', '.join(ag['classes_list'])}); for {ag['source_file_identifiable']:,} of them the source file exists in the "
      f"dataset. They make up {100 * ag['share_of_files_in_those_classes']:.0f}% of the files in those classes. They "
      "are not byte-identical to their source, so SHA-256 grouping does not keep them with it (see the split lock "
      "for how this was handled).\n")
    fs = F["file_size_bytes"]
    w(f"File size: median {fs['median']/1e6:.2f} MB, max {fs['max']/1e6:.2f} MB; files under 10 KB: {F['files_under_10kb']}.\n")

    w("# Part II - Representative Sample EDA (video properties)\n")
    w("## 6. Sampling methodology\n")
    w(f"- Pool: all {sel['pool_files']:,} reconciled files, {sel['classes_in_pool']:,} classes.")
    w(f"- Seed {sel['seed']}; every class gets {sel['per_class']} videos drawn without replacement; the remaining "
      f"slots up to {sel['target_total']:,} give +1 video to the {sel['selection_counts'].get('extra_smallest_class', 0)} "
      f"smallest and {sel['selection_counts'].get('extra_largest_class', 0)} largest classes (seeded tie-break).")
    w(f"- Result: **{S['sampled']:,} videos, {S['classes_represented']:,} classes**; per-class sample sizes "
      f"{sel['samples_per_class_counts']} (size: number of classes). Table: `data/metadata/eda_sample/sample_class_counts.csv`.")
    w(f"- Duplicate groups were not collapsed: {S['in_duplicate_groups']} sampled files belong to a duplicate group "
      f"({S['in_cross_label_groups']} cross-label); {sel['sampled_distinct_sha256']:,} distinct contents.")
    w("- The sample is class-balanced, not proportional: small classes are over-represented. Rates are therefore "
      "given both as raw sample percentages and as **class-size-weighted estimates** (weight = class size / "
      "sampled in class), which estimate the dataset-level proportion.")
    w(f"- Probe origin: {S['probe_origin']} (legacy = measured by the abandoned full probe; same verified bytes).\n")
    w("Source-proxy mix, sample vs full pool:\n")
    w(md_table(pd.DataFrame([(k, sel["source_proxy_sample"].get(k, 0), n) for k, n in sel["source_proxy_pool"].items()],
                            columns=["pattern", "sample", "full pool"])))

    w("\n## 7. Decodability (sample)\n")
    w(md_table(pd.DataFrame([
        ("sampled", S["sampled"]), ("downloaded", S["downloaded"]),
        ("download failed after retries (not a video defect)", S["download_failed_after_retries"]),
        ("unprobed", S["unprobed"]), ("SHA-256 mismatch", S["sha256_mismatch"]),
        ("decoded successfully", S["decode_ok"]), ("decode failures (real)", S["decode_failed"]),
    ], columns=["item", "videos"])))
    if S["decode_error_types"]:
        w("\nDecode errors:\n")
        w(md_table(pd.DataFrame(list(S["decode_error_types"].items()), columns=["error", "videos"])))
    rej = v[~v["eligible"]]
    if len(rej):
        w("\nSampled videos failing a Step 1 rule:\n")
        w(md_table(rej[["word", "repo_path", "exclusion_reason", "decoded_frames"]].fillna("").astype(str)))

    w("\n## 8. FPS, duration, frames (sample, decodable videos)\n")
    def row(name, d, fmt="{:.2f}"):
        return (name, *(fmt.format(d[k]) for k in ("min", "p05", "median", "mean", "p95", "max")))
    w(md_table(pd.DataFrame([
        row("container avg FPS", S["avg_fps"]), row("FPS from frame timestamps", S["decoded_fps"]),
        row("duration (s)", S["duration_s"]), row("decoded frames", S["decoded_frames"], "{:.0f}"),
        row("frames at 30 FPS", S["frames_at_30fps"], "{:.0f}"),
    ], columns=["measure", "min", "p5", "median", "mean", "p95", "max"])))
    up, dn = S["needs_upsampling_fps_below_29.5"], S["needs_downsampling_fps_above_30.5"]
    w("\n| 30 FPS standardisation | videos in sample | sample % | weighted estimate % |\n|---|---|---|---|")
    w(f"| temporal upsampling (FPS < 29.5) | {up['count']:,} | {up['sample_pct']:.1f} | {up['weighted_est_pct']:.1f} |")
    w(f"| temporal downsampling (FPS > 30.5) | {dn['count']:,} | {dn['sample_pct']:.1f} | {dn['weighted_est_pct']:.1f} |")
    w(f"| already ~30 FPS (±0.5) | {S['fps_within_0.5_of_30']:,} | | |")
    w(f"\n- Invalid/missing FPS: container {S['invalid_container_fps']}, timestamp-derived {S['invalid_timestamp_fps']}")
    w(f"- Suspected variable frame rate (container vs timestamp FPS differ >0.5): {S['variable_frame_rate_suspected']}")
    w(f"- Container frame count ≠ decoded frame count: {S['reported_vs_decoded_frame_mismatch']}")
    w("- Most common FPS values (rounded, sample): " + ", ".join(
        f"{k}: {n}" for k, n in sorted(S["fps_rounded_counts"].items(), key=lambda kv: -kv[1])[:8]) + "\n")
    w("Videos shorter than each window:\n")
    w("| window | decoded frames < window | sample % | weighted est. % | frames at 30 FPS < window | sample % | weighted est. % |")
    w("|---|---|---|---|---|---|---|")
    for wn in WINDOWS:
        a, b = S["decoded_frames_below"][str(wn)], S["frames_at_30fps_below"][str(wn)]
        w(f"| {wn} ({wn/30:.1f} s) | {a['count']:,} | {a['sample_pct']:.1f} | {a['weighted_est_pct']:.1f} | "
          f"{b['count']:,} | {b['sample_pct']:.1f} | {b['weighted_est_pct']:.1f} |")
    w("\nThe Step 1 rule rejects < 30 *decoded* frames. A low-FPS clip can have < 30 decoded frames yet last > 1 s "
      "(and vice versa), which is why both columns are shown. Clips shorter than 45/60 frames will be upsampled by "
      "uniform temporal resampling in Step 3.\n")

    w("## 9. Resolution, codec, container (sample)\n")
    w(md_table(pd.DataFrame(list(S["resolution_counts"].items()), columns=["resolution", "videos"])))
    w(f"\n- Distinct resolutions: {S['n_resolutions']}; codecs: {S['codec_counts']}; pixel formats: {S['pix_fmt_counts']}")
    w(f"- Containers: {S['container_counts']}; audio track: {S['has_audio']}; rotation metadata: "
      f"{S['rotation_metadata']}; multiple video streams: {S['multi_video_stream']}\n")

    w("## 10. Figures\n")
    w("Full manifest (exact): class_distribution, class_rank, class_content_overlap, source_proxy, file_size_hist. "
      "Sample: fps_hist, duration_hist, decoded_frames_hist, frames_at_30fps_hist, resolutions.\n")
    for f in ("class_distribution", "class_rank", "class_content_overlap", "source_proxy", "file_size_hist",
              "fps_hist", "duration_hist", "decoded_frames_hist", "frames_at_30fps_hist", "resolutions"):
        w(f"![{f}](figures/{f}.png)")

    w("\n## 11. Limitations\n")
    w("1. **Video properties are sample estimates** (n above), not exact dataset statistics. Rare anomalies "
      "(e.g. corrupt files) occurring in a few files may be missed or their rate poorly estimated. Real decode "
      "failures across the full dataset will only be known when Step 2 processes every video.\n"
      "2. The split is built from the complete metadata manifest without decoding every video; undecodable or "
      "too-short videos will be rejected in Step 2 and flagged within their frozen split (never re-split).\n"
      "3. Isolated-word dataset; not continuous signing.\n"
      "4. No signer IDs → the split is not signer-independent; test scores may be optimistic for unseen signers.\n"
      "5. Many labels are synonyms sharing identical videos (Section 3); the class set is not a set of distinct signs.\n"
      "6. Only byte-identical duplicates are detected; near-duplicates may still cross splits.\n")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
