"""Step 2.3: Phase A benchmark report (methodology v1 outputs: throughput, rejections, storage).

Historical: it reads the v1-format Phase A run. The methodology v2 evaluation of the same
run is scripts/s2_04_reevaluate_benchmark.py.

Reads the benchmark run and, if present, the worker-count sweep runs under
data/landmarks/benchmark_sweep/w*/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.landmarks.layout import assemble_features, feature_dim
from asl.landmarks.outputs import check_outputs, load_arrays, unreferenced_arrays
from asl.landmarks.pipeline import DOWNLOAD_FAILED, load_metadata

FULL_DATASET = 108_616


def run_totals(out_dir: Path) -> dict:
    runs = [json.loads(p.read_text()) for p in sorted((out_dir / "runs").glob("run_*.json"))]
    runs = [r for r in runs if r["videos_attempted"] > 0]
    return {
        "runs": len(runs), "wall_s": sum(r["wall_s"] for r in runs),
        "workers": sorted({r["workers"] for r in runs}),
        "peak_rss_mb": max((r["peak_rss_mb"] for r in runs if "peak_rss_mb" in r), default=None),
        "mean_cpu_pct": (float(np.mean([r["mean_cpu_pct"] for r in runs if "mean_cpu_pct" in r]))
                         if any("mean_cpu_pct" in r for r in runs) else None),
        "logical_cpus": next((r["logical_cpus"] for r in runs if "logical_cpus" in r), None),
        "sources": sorted({r.get("source", "run summary") for r in runs}),
    }


def throughput(meta: pd.DataFrame, wall_s: float) -> dict:
    m = meta[meta["rejection_reason"].ne(DOWNLOAD_FAILED)]
    frames = float(m["original_frame_count"].fillna(0).sum())
    return {"videos": int(len(m)), "wall_s": wall_s, "videos_per_min": 60 * len(m) / wall_s,
            "frames_per_s": frames / wall_s, "mediapipe_frames": int(frames)}


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    b = c2["benchmark"]
    out_dir = PROJECT_ROOT / b["out_dir"]
    rep = PROJECT_ROOT / "reports" / "step2"
    rep.mkdir(parents=True, exist_ok=True)
    configs = c2["configurations"]

    meta = load_metadata(out_dir)
    meta.to_csv(out_dir / "benchmark_metadata.csv", index=False)
    sample = pd.read_csv(out_dir / "benchmark_sample.csv", dtype=str, keep_default_na=False)
    # The v2 consistency checker does not apply to v1-format arrays.
    problems = (check_outputs(meta, out_dir / "arrays", configs) if "format_version" in meta.columns
                else ["skipped: v1-format outputs (see s2_04 for v2)"])
    rt = run_totals(out_dir)
    tp = throughput(meta, rt["wall_s"])

    att = meta[meta["rejection_reason"].ne(DOWNLOAD_FAILED)]
    ok = att[att["rejected"].eq(False)]
    rej = att[att["rejected"].eq(True)]
    reasons = rej["rejection_reason"].value_counts().to_dict()
    errors = rej.loc[rej["error"].fillna("") != "", ["sample_id", "rejection_reason", "error"]]

    # --- per-configuration: feature dims, assembly time, storage if stored alone.
    # Uses every saved array (accepted + hand-rule rejections kept for analysis) so the
    # per-video size is measured on as many videos as possible.
    per_cfg = {}
    saved = att[att["array_file"].fillna("").ne("")]
    arrays = {r.array_file: load_arrays(out_dir / "arrays" / r.array_file) for r in saved.itertuples()}
    for name, groups in configs.items():
        t0 = time.perf_counter()
        n_bytes = 0
        for a in arrays.values():
            n_bytes += assemble_features(a["coords"], a["present"], a["pose_vis"], groups).nbytes
        per_cfg[name] = {"groups": groups, "feature_dim": feature_dim(groups),
                         "assembly_s_total": time.perf_counter() - t0,
                         "float32_bytes_if_stored_alone": int(n_bytes),
                         "bytes_per_processed_video": n_bytes / max(len(saved), 1)}

    # --- missing-landmark statistics (attempted videos with landmarks, incl. hand-rule rejections)
    lm = att[att["std_frame_count"].notna() & att["both_hands_missing_pct"].notna()]
    missing = {g: {"mean_pct": float(lm[f"{g}_missing_pct"].mean()),
                   "median_pct": float(lm[f"{g}_missing_pct"].median())}
               for g in ("left_hand", "right_hand", "pose", "face")}
    missing["both_hands"] = {"mean_pct": float(lm["both_hands_missing_pct"].mean()),
                             "median_pct": float(lm["both_hands_missing_pct"].median())}
    runs_all = [ln for v in lm["both_hands_missing_runs"] for _, ln in json.loads(v)]
    run_stats = {"intervals": len(runs_all), "le_5": int(sum(x <= 5 for x in runs_all)),
                 "6_to_15": int(sum(5 < x <= 15 for x in runs_all)), "gt_15": int(sum(x > 15 for x in runs_all)),
                 "videos_with_run_gt_15": int((lm["longest_both_hands_missing_run"] > 15).sum())}
    frac_rest = (lm["leading_no_hands_frames"] + lm["trailing_no_hands_frames"]) / lm["std_frame_count"]
    whatif = lm["whatif_active_segment_rejection"].fillna("accepted").value_counts().to_dict()
    rest = {
        "leading_no_hands_frames": {"mean": float(lm["leading_no_hands_frames"].mean()),
                                    "median": float(lm["leading_no_hands_frames"].median())},
        "trailing_no_hands_frames": {"mean": float(lm["trailing_no_hands_frames"].mean()),
                                     "median": float(lm["trailing_no_hands_frames"].median())},
        "rest_share_of_clip_pct": {"mean": 100 * float(frac_rest.mean()), "median": 100 * float(frac_rest.median())},
        "active_segment_frames": {"mean": float(lm["active_segment_frames"].mean()),
                                  "median": float(lm["active_segment_frames"].median()),
                                  "below_30": int((lm["active_segment_frames"] < 30).sum())},
        "hand_rule_rejections_where_active_segment_passes": int(
            (lm["rejection_reason"].isin(["both_hands_missing_fraction", "both_hands_missing_run"])
             & lm["whatif_active_segment_rejection"].isna()).sum()),
    }
    interp = {g: int(ok[f"{g}_interpolated_frames"].sum()) for g in ("left_hand", "right_hand", "pose", "face")}
    zero = {g: int(ok[f"{g}_zero_filled_frames"].sum()) for g in ("left_hand", "right_hand", "pose", "face")}

    # --- timing per video
    tcols = {k: att[k].dropna() for k in ("total_s", "mediapipe_s", "decode_s", "post_s")}
    timing = {k: {"mean": float(v.mean()), "median": float(v.median()), "p95": float(v.quantile(0.95))}
              for k, v in tcols.items() if len(v)}
    mp_fps_per_worker = float(att["original_frame_count"].sum() / att["mediapipe_s"].sum())

    # --- worker sweep
    sweep = {}
    for d in sorted((PROJECT_ROOT / "data" / "landmarks" / "benchmark_sweep").glob("w*")):
        sm = load_metadata(d)
        if not sm.empty:
            sweep[d.name] = {**throughput(sm, run_totals(d)["wall_s"]), **run_totals(d)}

    # --- full-dataset estimates
    sv = pd.read_parquet(PROJECT_ROOT / c1["eda_sample"]["dir"] / "sample_validated.parquet")
    sv = sv[sv["decode_ok"].fillna(False).astype(bool)]
    mean_frames_weighted = float((sv["decoded_frames"] * sv["weight"]).sum() / sv["weight"].sum())
    est_frames = mean_frames_weighted * FULL_DATASET
    acc_rate = len(ok) / max(len(att), 1)
    acc_rate_whatif = float(lm["whatif_active_segment_rejection"].isna().sum() / max(len(att), 1))

    def estimate(videos_per_min: float, frames_per_s: float) -> dict:
        return {"hours_by_videos": FULL_DATASET / videos_per_min / 60,
                "hours_by_frames": est_frames / frames_per_s / 3600}

    est = {"benchmark_workers": estimate(tp["videos_per_min"], tp["frames_per_s"])}
    for k, v in sweep.items():
        est[f"sweep_{k}"] = estimate(v["videos_per_min"], v["frames_per_s"])
    per_video = float(saved["bytes"].sum()) / max(len(saved), 1)
    storage = {
        "stored_npz_all_groups_bytes_per_saved_video": per_video,
        "saved_videos_measured": int(len(saved)),
        "est_full_dataset_npz_gb_at_specified_rules": per_video * FULL_DATASET * acc_rate / 1e9,
        "est_full_dataset_npz_gb_at_whatif_rule": per_video * FULL_DATASET * acc_rate_whatif / 1e9,
        "est_full_dataset_npz_gb_if_all_saved": per_video * FULL_DATASET / 1e9,
        "per_configuration_float32_gb_if_all_saved_alone": {
            k: v["bytes_per_processed_video"] * FULL_DATASET / 1e9 for k, v in per_cfg.items()},
        "acceptance_rate_specified_rules": acc_rate,
        "acceptance_rate_whatif_rule": acc_rate_whatif,
        "benchmark_mean_frames_30fps": float(ok["std_frame_count"].mean()),
        "full_dataset_mean_decoded_frames_weighted_est": mean_frames_weighted,
    }

    stats = {
        "benchmark": {"videos_selected": int(len(sample)), "classes": int(sample["word"].nunique()),
                      "classes_attempted": int(att["word"].nunique()),
                      "attempted_by_selection_reason": sample[sample["sample_id"].isin(set(att["sample_id"]))]
                      ["benchmark_reason"].value_counts().to_dict(),
                      "attempted": int(len(att)), "download_failed": int(meta["rejection_reason"].eq(DOWNLOAD_FAILED).sum()),
                      "processed": int(len(ok)), "rejected": int(len(rej)), "rejection_reasons": reasons,
                      "rejection_rate_pct": 100 * len(rej) / max(len(att), 1),
                      "consistency_problems": problems[:50], "n_consistency_problems": len(problems),
                      "unreferenced_array_files": len(unreferenced_arrays(meta, out_dir / "arrays"))},
        "run": rt, "throughput": tp, "timing_per_video_s": timing,
        "mediapipe_frames_per_s_per_worker": mp_fps_per_worker,
        "configurations": per_cfg, "missing": missing, "rest_frames": rest,
        "whatif_active_segment_rule": whatif, "both_hands_missing_runs": run_stats,
        "interpolated_frames": interp, "zero_filled_frames": zero,
        "shoulder_frames_carried": int(ok["shoulder_frames_carried"].sum()),
        "nonfinite_detections_repaired": int(att["nonfinite_detections_repaired"].fillna(0).sum()),
        "sweep": sweep, "estimates": est, "storage": storage,
        "errors": errors.to_dict("records"),
        "free_disk_gb_now": __import__("shutil").disk_usage(out_dir).free / 1e9,
    }
    (rep / "benchmark_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    write_report(rep / "benchmark_report.md", stats)
    print(json.dumps({k: stats[k] for k in ("benchmark", "throughput", "estimates", "storage", "sweep")},
                     indent=2, default=str))


def write_report(path: Path, s: dict) -> None:
    bm, tp, rt = s["benchmark"], s["throughput"], s["run"]
    L = [
        "# Step 2 Phase A - MediaPipe Holistic benchmark\n",
        "All numbers are measured by `scripts/s2_03_benchmark_report.py` (`reports/step2/benchmark_stats.json`). "
        "Full-dataset figures are extrapolations from this benchmark.\n",
        "## Benchmark set\n",
        f"- {bm['videos_selected']} videos from the frozen TRAIN split, {bm['classes']} classes "
        "(`data/landmarks/benchmark/benchmark_sample.csv`).",
        f"- Stopped early at user request (Mac heating): attempted {bm['attempted']} videos "
        f"({bm['classes_attempted']} classes), first by sample_id order; by selection reason: "
        f"{bm['attempted_by_selection_reason']}.",
        f"- Attempted {bm['attempted']}, download failures {bm['download_failed']}, processed {bm['processed']}, "
        f"rejected {bm['rejected']} ({bm['rejection_rate_pct']:.1f}%).",
        f"- Rejection reasons: {bm['rejection_reasons']}",
        f"- Output consistency problems: {bm['n_consistency_problems']}; unreferenced array files (from the "
        f"interrupted batch, ignored): {bm['unreferenced_array_files']}\n",
        "## Throughput\n",
        f"- Workers {rt['workers']}, wall time {tp['wall_s'] / 60:.1f} min (includes download, overlapped).",
        f"- **{tp['videos_per_min']:.1f} videos/min**, **{tp['frames_per_s']:.1f} source frames/s** "
        f"({tp['mediapipe_frames']:,} frames through Holistic).",
        f"- Holistic alone: {s['mediapipe_frames_per_s_per_worker']:.1f} frames/s per worker.",
        (f"- Peak RSS (all processes) {rt['peak_rss_mb']:.0f} MB; mean system CPU {rt['mean_cpu_pct']:.0f}% of "
         f"{rt['logical_cpus']} logical CPUs.\n" if rt["peak_rss_mb"] is not None else
         f"- RAM/CPU: not available (run stopped at user request before the resource summary was written). "
         f"Wall-time source: {rt['sources']}.\n"),
        "| per video (s) | mean | median | p95 |", "|---|---|---|---|",
    ]
    for k, v in s["timing_per_video_s"].items():
        L.append(f"| {k} | {v['mean']:.2f} | {v['median']:.2f} | {v['p95']:.2f} |")
    if s["sweep"]:
        L += ["\nWorker sweep (same first videos of the benchmark):\n",
              "| run | workers | videos | videos/min | frames/s | peak RSS MB | mean CPU % |", "|---|---|---|---|---|---|---|"]
        for k, v in s["sweep"].items():
            L.append(f"| {k} | {v['workers']} | {v['videos']} | {v['videos_per_min']:.1f} | {v['frames_per_s']:.1f} | "
                     f"{v['peak_rss_mb']:.0f} | {v['mean_cpu_pct']:.0f} |")
    L += ["\n## Configurations\n",
          "One Holistic pass produces all groups; configurations are feature selections of the same stored "
          "arrays, so extraction time is identical for all three. Differences are feature size and storage.\n",
          "| configuration | groups | feature dim | float32 MB per video if stored alone |", "|---|---|---|---|"]
    for k, v in s["configurations"].items():
        L.append(f"| {k} | {', '.join(v['groups'])} | {v['feature_dim']} | {v['bytes_per_processed_video'] / 1e6:.3f} |")
    L += ["\n## Missing landmarks (30 FPS grid, before gap filling)\n", "| group | mean % frames missing | median % |",
          "|---|---|---|"]
    for g, v in s["missing"].items():
        L.append(f"| {g} | {v['mean_pct']:.1f} | {v['median_pct']:.1f} |")
    r = s["both_hands_missing_runs"]
    rf = s["rest_frames"]
    L += ["\n### Rest frames (hands outside the frame before/after the sign)\n",
          f"- Leading frames without hands: mean {rf['leading_no_hands_frames']['mean']:.1f}, median "
          f"{rf['leading_no_hands_frames']['median']:.0f}; trailing: mean {rf['trailing_no_hands_frames']['mean']:.1f}, "
          f"median {rf['trailing_no_hands_frames']['median']:.0f}.",
          f"- Rest share of clip: mean {rf['rest_share_of_clip_pct']['mean']:.1f}%, median "
          f"{rf['rest_share_of_clip_pct']['median']:.1f}%.",
          f"- Active signing segment (first→last hand detection): mean {rf['active_segment_frames']['mean']:.1f} "
          f"frames, median {rf['active_segment_frames']['median']:.0f}; {rf['active_segment_frames']['below_30']} "
          "segments shorter than 30 frames.",
          f"- WHAT-IF (not applied): hand rules evaluated only on the active segment → {s['whatif_active_segment_rule']}; "
          f"{rf['hand_rule_rejections_where_active_segment_passes']} current hand-rule rejections would pass.",
          ]
    L += [f"\nBoth-hands-missing intervals: {r['intervals']} (≤5 frames: {r['le_5']}, 6-15: {r['6_to_15']}, "
          f">15: {r['gt_15']}); videos with a run >15: {r['videos_with_run_gt_15']}.",
          f"Interpolated frames (processed videos): {s['interpolated_frames']}; zero-filled: {s['zero_filled_frames']}.",
          f"Frames using carried-over shoulder frame: {s['shoulder_frames_carried']}; non-finite detections repaired: "
          f"{s['nonfinite_detections_repaired']}.\n",
          "## Full-dataset estimates (108,616 videos)\n", "| basis | hours (per-video rate) | hours (per-frame rate) |",
          "|---|---|---|"]
    for k, v in s["estimates"].items():
        L.append(f"| {k} | {v['hours_by_videos']:.1f} | {v['hours_by_frames']:.1f} |")
    st = s["storage"]
    L += [f"\nStorage: stored NPZ (all groups, float32) {st['stored_npz_all_groups_bytes_per_saved_video'] / 1e6:.3f} MB "
          f"per saved video (measured on {st['saved_videos_measured']}). Full dataset: "
          f"**{st['est_full_dataset_npz_gb_at_specified_rules']:.1f} GB** at the specified rules "
          f"(acceptance {100 * st['acceptance_rate_specified_rules']:.1f}%), "
          f"**{st['est_full_dataset_npz_gb_at_whatif_rule']:.1f} GB** at the what-if rule "
          f"(acceptance {100 * st['acceptance_rate_whatif_rule']:.1f}%), "
          f"{st['est_full_dataset_npz_gb_if_all_saved']:.1f} GB if every video were saved. Per configuration stored "
          "alone (float32, all videos): "
          + ", ".join(f"{k} {v:.1f} GB" for k, v in st["per_configuration_float32_gb_if_all_saved_alone"].items())
          + f". Free disk now: {s['free_disk_gb_now']:.1f} GB.\n",
          "The per-frame estimate uses the class-size-weighted mean decoded frame count from the Step 1 sample "
          f"({st['full_dataset_mean_decoded_frames_weighted_est']:.1f}); the benchmark over-represents some "
          "properties by design (quotas), so the per-frame estimate is the more representative one.\n"]
    if s["errors"]:
        L += ["## Errors\n", "| sample | reason | error |", "|---|---|---|"]
        L += [f"| {e['sample_id']} | {e['rejection_reason']} | {e['error']} |" for e in s["errors"]]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
