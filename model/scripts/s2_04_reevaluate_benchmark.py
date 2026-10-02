"""Step 2.4: re-evaluate the Phase A benchmark under methodology v2 WITHOUT re-running MediaPipe.

The Phase A run stored, for 391 of 400 attempted videos, the full landmark arrays
(v1 format, float32) including the raw per-frame detections (`observed`). Because
the v2 rules only change which frames the hand rules look at, and v2 storage only
changes dtype/layout, both can be evaluated exactly from those arrays:

1. hand rules on the active signing interval (the same functions the pipeline uses);
2. v2 storage size: float32 vs float16, compressed vs not, per configuration;
3. float16 precision validation against the pre-registered criteria below.

Outputs: reports/step2/methodology_v2_report.md, reports/step2/methodology_v2_stats.json,
data/landmarks/benchmark/benchmark_metadata_v2_reevaluated.csv
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.landmarks import preprocess as pp
from asl.landmarks.layout import GROUP_INDEX, GROUP_ORDER, SLICES, assemble_features
from asl.landmarks.outputs import load_arrays
from asl.landmarks.pipeline import DOWNLOAD_FAILED, load_metadata
from asl.landmarks.convert import v2_from_v1
from asl.landmarks.process import HANDS

FULL_DATASET = 108_616
# Pre-registered float16 acceptance criteria (fixed before looking at the results).
CRITERIA = {
    "max_abs_value_below": 60000.0,        # float16 max is 65504: no overflow
    "p999_error_px_below": 0.1,            # 99.9th percentile rounding error, in source-video pixels
    "max_error_px_below": 0.5,             # worst case stays sub-pixel (MediaPipe input resolution is 1 px)
    "max_zscore_feature_error_below": 1e-2,  # model-input features after z-scoring (in feature std units)
    "max_stat_feature_rel_error_below": 1e-2,  # Random-Forest statistics (mean/std/min/max), relative to feature std
}


VARIANTS = {
    "float32": "float32",
    "float16": "float16",
    "float16_pose_float32": {"pose": "float32", "face": "float16", "left_hand": "float16", "right_hand": "float16"},
}


def precision_report(stored, reference, packed, configs, ref_feats, ref_sd) -> dict:
    """Rounding error of a storage variant vs the float32 reference, on present landmarks only."""
    per_group = {}
    all_su, all_px, max_abs, zeros_ok = [], [], 0.0, True
    for g in GROUP_ORDER:
        su, px = [], []
        for sid, a in stored.items():
            ref = reference[sid][g].astype(np.float64)
            got = packed[sid][g].astype(np.float64)
            pres = a["present"][:, GROUP_INDEX[g]].astype(bool)
            zeros_ok &= bool((got[~pres] == 0).all())
            if pres.any():
                e = np.abs(got - ref)[pres]
                su.append(e.ravel())
                px.append((e * a["scale"].astype(np.float64)[pres][:, None, None]).ravel())
                max_abs = max(max_abs, float(np.abs(ref[pres]).max()))
        su, px = np.concatenate(su), np.concatenate(px)
        all_su.append(su)
        all_px.append(px)
        per_group[g] = {"max_abs_value": float(np.abs(np.concatenate(
            [reference[s][g][stored[s]["present"][:, GROUP_INDEX[g]].astype(bool)].ravel() for s in stored])).max()),
            "p999_px": float(np.quantile(px, 0.999)), "max_px": float(px.max())}
    su, px = np.concatenate(all_su), np.concatenate(all_px)
    downstream = {}
    for k, groups in configs.items():
        sd = ref_sd[k]
        f16 = [assemble_features(packed[sid], groups) for sid in stored]
        stat = lambda x: np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])
        downstream[k] = {
            "max_zscore_feature_error": max(float(np.abs((x - r) / sd).max()) for x, r in zip(f16, ref_feats[k])),
            "max_stat_feature_rel_error": max(float(np.abs((stat(x) - stat(r)) / np.tile(sd, 4)).max())
                                              for x, r in zip(f16, ref_feats[k]))}
    checks = {
        "no_overflow": max_abs < CRITERIA["max_abs_value_below"],
        "zero_fill_preserved": zeros_ok,
        "p999_error_px": float(np.quantile(px, 0.999)) < CRITERIA["p999_error_px_below"],
        "max_error_px": float(px.max()) < CRITERIA["max_error_px_below"],
        "zscore_features": all(d["max_zscore_feature_error"] < CRITERIA["max_zscore_feature_error_below"]
                               for d in downstream.values()),
        "rf_statistics": all(d["max_stat_feature_rel_error"] < CRITERIA["max_stat_feature_rel_error_below"]
                             for d in downstream.values()),
    }
    return {"values_checked": int(su.size), "max_abs_value_shoulder_units": max_abs,
            "error_shoulder_units": {"mean": float(su.mean()), "p999": float(np.quantile(su, 0.999)), "max": float(su.max())},
            "error_px": {"mean": float(px.mean()), "p99": float(np.quantile(px, 0.99)),
                         "p999": float(np.quantile(px, 0.999)), "max": float(px.max())},
            "per_group": per_group, "downstream": downstream, "checks": checks, "passes_all": all(checks.values())}


def npz_size(arrays: dict, compress: bool) -> int:
    buf = io.BytesIO()
    (np.savez_compressed if compress else np.savez)(buf, **arrays)
    return buf.getbuffer().nbytes


def config_keys(groups: list[str]) -> list[str]:
    return groups + ["flags", "center", "scale", "shoulder_measured", "active_interval", "version"] + (
        ["pose_vis"] if "pose" in groups else [])


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    rules, out_cfg, configs = c2["rejection"], c2["output"], c2["configurations"]
    bdir = PROJECT_ROOT / c2["benchmark"]["out_dir"]
    rep = PROJECT_ROOT / "reports" / "step2"
    meta = load_metadata(bdir)
    meta = meta[meta["rejection_reason"].ne(DOWNLOAD_FAILED)]

    # ---------------- 1. rules
    rows = []
    stored: dict[str, dict] = {}
    for r in meta.itertuples(index=False):
        rec = {"sample_id": r.sample_id, "word": r.word, "split": r.split,
               "original_frame_count": r.original_frame_count, "std_frame_count": r.std_frame_count,
               "v1_rejection_reason": r.rejection_reason}
        if isinstance(r.array_file, str) and r.array_file:
            a = load_arrays(bdir / "arrays" / r.array_file)
            stored[r.sample_id] = a
            hands = a["observed"][:, HANDS].astype(bool)
            iv = pp.active_interval(hands, rules["active_interval_min_hand_run"])
            reason = pp.rejection_reason(int(r.original_frame_count), hands, rules)
            if reason is None and not a["shoulder_measured"].any():
                reason = "normalization_no_valid_shoulders"
            rec.update(v2_rejection_reason=reason, fullclip_rule=pp.fullclip_rejection_reason(
                int(r.original_frame_count), hands, rules), active_start_frame=iv.start if iv else None,
                active_end_frame=iv.end if iv else None, active_frames=iv.frames if iv else 0,
                short_active_interval=bool(iv and iv.frames < rules["short_active_interval_frames"]))
            if iv:
                q = pp.hand_quality(hands[iv.start : iv.end + 1])
                rec.update(active_both_hands_missing_pct=100 * q["both_missing_frac"],
                           active_longest_both_hands_missing_run=q["longest_both_missing_run"],
                           rest_frames_before=iv.start, rest_frames_after=len(hands) - 1 - iv.end)
        else:  # no landmarks stored: v1 rejected before extraction output (too few frames)
            rec["v2_rejection_reason"] = r.rejection_reason
        rows.append(rec)
    v2 = pd.DataFrame(rows)
    v2.to_csv(bdir / "benchmark_metadata_v2_reevaluated.csv", index=False)
    acc = v2[v2["v2_rejection_reason"].isna()]

    # ---------------- 2./3. storage and precision per storage variant
    ivs = {sid: pp.active_interval(a["observed"][:, HANDS].astype(bool), rules["active_interval_min_hand_run"])
           for sid, a in stored.items()}
    variants = {name: {**out_cfg, "coord_dtype": dt} for name, dt in VARIANTS.items()}
    reference = {sid: v2_from_v1(a, ivs[sid], variants["float32"]) for sid, a in stored.items()}
    ref_feats = {k: [assemble_features(reference[sid], g) for sid in stored] for k, g in configs.items()}
    ref_sd = {}
    for k, fs in ref_feats.items():
        sd = np.concatenate(fs).std(0)
        ref_sd[k] = np.where(sd > 1e-8, sd, 1.0)
    acc_rate = len(acc) / len(v2)
    n_saved = len(stored)
    results = {}
    for name, vcfg in variants.items():
        packed = {sid: v2_from_v1(a, ivs[sid], vcfg) for sid, a in stored.items()}
        size = {cz: sum(npz_size(x, cz) for x in packed.values()) / n_saved for cz in (False, True)}
        cfg_size = {k: {cz: sum(npz_size({x: arr[x] for x in config_keys(g)}, cz) for arr in packed.values()) / n_saved
                        for cz in (False, True)} for k, g in configs.items()}
        res = {"bytes_per_video": {"uncompressed": size[False], "compressed": size[True]},
               "est_full_dataset_gb": {"uncompressed": size[False] * FULL_DATASET * acc_rate / 1e9,
                                       "compressed": size[True] * FULL_DATASET * acc_rate / 1e9},
               "est_full_dataset_gb_per_config_alone": {
                   k: {"uncompressed": v[False] * FULL_DATASET * acc_rate / 1e9,
                       "compressed": v[True] * FULL_DATASET * acc_rate / 1e9} for k, v in cfg_size.items()}}
        if name != "float32":
            res["precision"] = precision_report(stored, reference, packed, configs, ref_feats, ref_sd)
        results[name] = res

    stats = {
        "videos_attempted": int(len(v2)), "videos_with_stored_landmarks": n_saved,
        "v1_decisions": v2["v1_rejection_reason"].fillna("accepted").value_counts().to_dict(),
        "v2_decisions": v2["v2_rejection_reason"].fillna("accepted").value_counts().to_dict(),
        "v2_acceptance_rate": acc_rate,
        "v2_accepted_short_active_interval": int(acc["short_active_interval"].sum()),
        "active_frames_accepted": acc["active_frames"].describe().round(2).to_dict(),
        "active_both_hands_missing_pct_accepted": acc["active_both_hands_missing_pct"].describe().round(2).to_dict(),
        "rest_frames_before": v2["rest_frames_before"].describe().round(2).to_dict(),
        "rest_frames_after": v2["rest_frames_after"].describe().round(2).to_dict(),
        "acceptance_rate_used_for_storage": acc_rate, "variants": results, "criteria": CRITERIA,
    }
    (rep / "methodology_v2_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    write_report(rep / "methodology_v2_report.md", stats)
    print(json.dumps({k: stats[k] for k in ("v1_decisions", "v2_decisions", "v2_acceptance_rate",
                                             "v2_accepted_short_active_interval", "variants")}, indent=2, default=str))


def write_report(path: Path, s: dict) -> None:
    L = ["# Step 2 - methodology v2 re-evaluation of the Phase A benchmark\n",
         "Re-evaluated from the stored Phase A extractions (no MediaPipe re-run): "
         f"{s['videos_attempted']} attempted videos, {s['videos_with_stored_landmarks']} with stored landmarks.\n",
         "## Decisions\n", "| rule | outcome counts |", "|---|---|",
         f"| v1 (whole clip) | {s['v1_decisions']} |", f"| v2 (active signing interval) | {s['v2_decisions']} |",
         f"\nv2 acceptance: **{100 * s['v2_acceptance_rate']:.1f}%**; accepted clips with a short active interval "
         f"(< 30 frames; flagged `short_active_interval`, not rejected): {s['v2_accepted_short_active_interval']}.",
         f"\nActive frames (accepted): {s['active_frames_accepted']}",
         f"\nBoth hands missing inside the interval, % (accepted): {s['active_both_hands_missing_pct_accepted']}",
         f"\nRest frames before: {s['rest_frames_before']}\n\nRest frames after: {s['rest_frames_after']}\n",
         "## Storage variants (measured on the stored benchmark extractions)\n",
         f"Full-dataset estimates use the v2 acceptance rate ({100 * s['acceptance_rate_used_for_storage']:.1f}%).\n",
         "| variant | MB/video | MB/video compressed | full GB | full GB compressed | hands GB | hands+pose GB | "
         "hands+pose+face GB | precision |", "|---|---|---|---|---|---|---|---|---|"]
    for name, v in s["variants"].items():
        c = v["est_full_dataset_gb_per_config_alone"]
        prec = "reference" if "precision" not in v else ("PASS" if v["precision"]["passes_all"] else "FAIL")
        L.append(f"| {name} | {v['bytes_per_video']['uncompressed'] / 1e6:.3f} | {v['bytes_per_video']['compressed'] / 1e6:.3f} "
                 f"| {v['est_full_dataset_gb']['uncompressed']:.1f} | {v['est_full_dataset_gb']['compressed']:.1f} "
                 f"| {c['hands']['uncompressed']:.1f} | {c['hands_pose']['uncompressed']:.1f} "
                 f"| {c['hands_pose_face']['uncompressed']:.1f} | {prec} |")
    L += ["\n## Precision validation (pre-registered criteria)\n", f"Criteria: {s['criteria']}\n"]
    for name, v in s["variants"].items():
        if "precision" not in v:
            continue
        p = v["precision"]
        L += [f"### {name}\n", f"- values checked {p['values_checked']:,}; error px {p['error_px']}",
              f"- per group: {p['per_group']}", f"- downstream: {p['downstream']}",
              f"- checks: {p['checks']} → **{'PASS' if p['passes_all'] else 'FAIL'}**\n"]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
