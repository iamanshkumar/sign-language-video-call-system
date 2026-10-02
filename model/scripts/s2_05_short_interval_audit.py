"""Step 2.5: class-level audit of short active signing intervals (benchmark only, no MediaPipe).

Question: are clips with an active interval < 30 frames (flag `short_active_interval`)
concentrated in particular classes or clip types, and which classes would lose all
their benchmark clips if short intervals were rejected?

Inputs: data/landmarks/benchmark/benchmark_metadata_v2_reevaluated.csv (s2_04),
benchmark_sample.csv, Phase A metadata. Outputs: reports/step2/short_interval_audit.md,
short_interval_audit.json, tables/short_interval_by_class.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.pipeline import load_metadata


def label_type(word: str) -> str:
    if len(word) == 1 and word.isalpha():
        return "fingerspelled_letter"
    if any(ch.isdigit() for ch in word):
        return "contains_digit"
    if " " in word or "_" in word:
        return "multi_word"
    return "single_word"


def rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = df.groupby(col, observed=True)["short_active_interval"].agg(accepted="size", short="sum")
    t["short_pct"] = 100 * t["short"] / t["accepted"]
    return t.sort_values("accepted", ascending=False)


def chi2(df: pd.DataFrame, col: str) -> dict:
    ct = pd.crosstab(df[col], df["short_active_interval"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"p_value": None, "note": "not testable"}
    stat, p, dof, expected = chi2_contingency(ct)
    return {"chi2": float(stat), "dof": int(dof), "p_value": float(p),
            "min_expected_count": float(expected.min()),
            "note": "approximate: expected counts < 5" if expected.min() < 5 else ""}


def md(t: pd.DataFrame) -> str:
    t = t.reset_index()
    out = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    for r in t.itertuples(index=False):
        out.append("| " + " | ".join(f"{x:.1f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> None:
    c2 = load_yaml(STEP2_CONFIG)
    thr = c2["rejection"]["short_active_interval_frames"]
    bdir = PROJECT_ROOT / c2["benchmark"]["out_dir"]
    rep = PROJECT_ROOT / "reports" / "step2"
    (rep / "tables").mkdir(parents=True, exist_ok=True)

    v2 = pd.read_csv(bdir / "benchmark_metadata_v2_reevaluated.csv", keep_default_na=False, na_values=[""])
    sample = pd.read_csv(bdir / "benchmark_sample.csv", dtype=str, keep_default_na=False)
    v1 = load_metadata(bdir)[["sample_id", "original_fps", "original_duration_s"]]
    df = (v2.merge(sample[["sample_id", "source_proxy", "dup_group_size", "aug_source", "benchmark_reason"]],
                   on="sample_id", how="left")
          .merge(v1, on="sample_id", how="left"))
    acc = df[df["v2_rejection_reason"].isna()].copy()
    acc["short_active_interval"] = acc["short_active_interval"].astype(str).str.lower().eq("true")
    acc["label_type"] = acc["word"].map(label_type)
    acc["augmented"] = acc["aug_source"].fillna("").ne("")
    acc["in_duplicate_group"] = acc["dup_group_size"].astype(int) > 1
    acc["clip_frames_30fps"] = pd.cut(acc["std_frame_count"], [0, 45, 60, 90, 10_000],
                                      labels=["<45", "45-59", "60-89", ">=90"], right=False)
    acc["fps_band"] = pd.cut(acc["original_fps"], [0, 29.5, 30.5, 1_000], labels=["<29.5", "~30", ">30.5"])
    acc["active_share_of_clip_pct"] = 100 * acc["active_frames"] / acc["std_frame_count"]
    short = acc[acc["short_active_interval"]]

    # ---- per class
    by_class = acc.groupby("word").agg(accepted=("sample_id", "size"), short=("short_active_interval", "sum"),
                                       min_active_frames=("active_frames", "min"),
                                       median_active_frames=("active_frames", "median"))
    by_class["short_pct"] = 100 * by_class["short"] / by_class["accepted"]
    by_class = by_class.sort_values(["short_pct", "accepted"], ascending=[False, False])
    by_class.to_csv(rep / "tables" / "short_interval_by_class.csv")
    all_short = by_class[(by_class["short"] == by_class["accepted"])]
    multi = by_class[by_class["accepted"] >= 2]

    rho, p_rho = spearmanr(acc["std_frame_count"], acc["active_frames"])
    stats = {
        "threshold_frames": thr,
        "accepted_clips": int(len(acc)), "short_clips": int(len(short)),
        "short_pct": 100 * len(short) / len(acc),
        "short_active_frames": short["active_frames"].describe().round(2).to_dict(),
        "short_active_seconds_median": float(short["active_frames"].median() / 30),
        "short_under_10_frames": int((short["active_frames"] < 10).sum()),
        "classes_with_accepted_clips": int(len(by_class)),
        "classes_with_any_short": int((by_class["short"] > 0).sum()),
        "classes_all_accepted_clips_short": int(len(all_short)),
        "classes_with_2plus_accepted": int(len(multi)),
        "classes_2plus_with_any_short": int((multi["short"] > 0).sum()),
        "classes_2plus_all_short": int((multi["short"] == multi["accepted"]).sum()),
        "spearman_clip_length_vs_active_frames": {"rho": float(rho), "p_value": float(p_rho)},
        "active_share_of_clip_pct": {"short": short["active_share_of_clip_pct"].describe().round(1).to_dict(),
                                     "not_short": acc.loc[~acc["short_active_interval"], "active_share_of_clip_pct"]
                                     .describe().round(1).to_dict()},
        "chi2": {c: chi2(acc, c) for c in ("label_type", "source_proxy", "clip_frames_30fps", "fps_band")},
    }
    tables = {c: rate_table(acc, c) for c in ("label_type", "source_proxy", "clip_frames_30fps", "fps_band",
                                                "augmented", "in_duplicate_group")}
    stats["rates"] = {c: t.reset_index().astype({c: str}).to_dict("records") for c, t in tables.items()}
    (rep / "short_interval_audit.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")

    shortest = short.sort_values("active_frames")[["word", "sample_id", "active_frames", "std_frame_count",
                                                   "rest_frames_before", "rest_frames_after",
                                                   "active_both_hands_missing_pct"]].head(15)
    L = [
        "# Step 2 - class-level short-active-interval audit (benchmark)\n",
        f"Short = active signing interval < {thr} frames (flag `short_active_interval`, not a rejection). "
        f"Population: the {len(acc)} benchmark clips accepted under methodology v2 (re-evaluated from stored "
        "Phase A detections; no MediaPipe re-run).\n",
        "> **Limitation:** the benchmark has ~1 accepted clip per class (see below) and is not proportional "
        "to the dataset, so per-class rates are indicative only; category-level rates are more reliable. "
        "A dataset-wide class audit needs the full extraction.\n",
        "## Overall\n",
        f"- Short clips: **{len(short)} / {len(acc)} ({stats['short_pct']:.1f}%)**; active frames among short clips: "
        f"{stats['short_active_frames']} (median {stats['short_active_seconds_median']:.2f} s); "
        f"{stats['short_under_10_frames']} have < 10 active frames.",
        f"- Clip length vs active frames: Spearman ρ = {rho:.2f} (p = {p_rho:.1e}).",
        f"- Active share of clip (%): short {stats['active_share_of_clip_pct']['short']}; "
        f"not short {stats['active_share_of_clip_pct']['not_short']}.\n",
        "## Classes\n",
        f"- Classes with accepted clips: {len(by_class)}; with ≥1 short clip: {stats['classes_with_any_short']}; "
        f"**all accepted clips short: {len(all_short)}** (these classes would have no benchmark clip left if short "
        "intervals were rejected).",
        f"- Classes with ≥2 accepted clips: {len(multi)}; of these, with any short: "
        f"{stats['classes_2plus_with_any_short']}, all short: {stats['classes_2plus_all_short']}.\n",
        "Classes with ≥2 accepted clips and at least one short clip:\n",
        md(multi[multi["short"] > 0].round(1)),
        f"\nClasses whose every accepted benchmark clip is short ({len(all_short)}): "
        + ", ".join(f"`{w}`" for w in all_short.index) + "\n",
        "Full per-class table: `reports/step2/tables/short_interval_by_class.csv`.\n",
        "## Clip categories\n",
    ]
    for c, t in tables.items():
        test = stats["chi2"].get(c)
        L += [f"### {c}\n", md(t.round(1))]
        if test:
            L.append(f"\nχ² test (short vs not short): p = {test['p_value']:.3g} {test.get('note', '')}\n"
                     if test["p_value"] is not None else "\nχ² test: not testable\n")
    L += ["## Shortest active intervals (accepted)\n", md(shortest.set_index("word").round(1)),
          "\nVery short intervals are worth a visual spot-check: they may be quick signs, or clips where the "
          "hands are mostly outside the frame/undetected even during the sign.\n"]
    (rep / "short_interval_audit.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: stats[k] for k in ("accepted_clips", "short_clips", "short_pct", "short_active_frames",
                                             "short_under_10_frames", "classes_with_accepted_clips",
                                             "classes_with_any_short", "classes_all_accepted_clips_short",
                                             "classes_with_2plus_accepted", "classes_2plus_with_any_short",
                                             "classes_2plus_all_short", "spearman_clip_length_vs_active_frames",
                                             "active_share_of_clip_pct", "chi2")}, indent=2, default=str))
    for c, t in tables.items():
        print(f"\n{c}\n{t.round(1).to_string()}")


if __name__ == "__main__":
    main()
