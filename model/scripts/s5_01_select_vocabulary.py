"""Step 5.1: evidence-driven subset vocabularies (metadata only; no video is downloaded).

Inputs: frozen split v1 (checksum-verified), manifest, duplicate-content groups, label audit.
Outputs: configs/vocabulary_100.json, configs/vocabulary_expanded.json,
reports/step5/vocabulary_report.md, reports/step5/summary.json, reports/step5/tables/*.csv.
Existing vocabulary files are never replaced with different content unless --overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from asl.config import PROJECT_ROOT, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.data.vocabulary import (SPLITS, add_file_columns, class_features, components, exclusion_reasons, label_pairs,
                                 pair_treatment, rank_candidates, summary_counts)

CFG = PROJECT_ROOT / "configs" / "step5_vocabulary.yaml"
FEATURE_COLS = ["clips", "train_clips", "val_clips", "test_clips", "unique_clips", "units", "train_units",
                "val_units", "test_units", "cross_label_clips", "cross_label_fraction", "augmented_clips",
                "augmented_fraction", "within_dup_clips", "within_dup_fraction", "source_patterns",
                "max_batch_share", "numericid_clips", "train_unit_fraction"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    df = df[cols] if cols else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    return head + "\n" + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))


def vocabulary_doc(name: str, words: list[str], f: pd.DataFrame, ranked: pd.DataFrame, full_map: dict, meta: dict) -> dict:
    words = sorted(words)  # index = position in Unicode code-point order (same rule as the full mapping)
    classes = []
    for i, w in enumerate(words):
        r = f.loc[w]
        classes.append({
            "index": i, "class": w, "dataset_label": w, "full_vocabulary_index": full_map["word_to_index"][w],
            "selection_rank": int(ranked.loc[w, "rank"]),
            "reason": (f"passes every Step 5 rule (no shared/augmented clips, single-token label without audit "
                       f"anomaly, >= 6 independent val and test units, balanced split, <= 50% from one batch, "
                       f">= 2 source collections); rank {int(ranked.loc[w, 'rank'])} by min(val,test) units "
                       f"= {int(r.min_eval_units)}, source collections = {int(r.source_patterns)}, "
                       f"units = {int(r.units)}"),
            **{k: int(r[k]) for k in ("clips", "train_clips", "val_clips", "test_clips", "unique_clips", "units",
                                      "train_units", "val_units", "test_units", "source_patterns")},
            "cross_label_shared_fraction": float(r.cross_label_fraction),
            "augmented_fraction": float(r.augmented_fraction),
        })
    body = {"name": name, "num_classes": len(words), "merged_labels": [], **meta,
            "ordering": "class names sorted by Unicode code point; index = position",
            "summary": summary_counts(f, words), "classes": classes}
    body["vocabulary_sha256"] = hashlib.sha256("\n".join(words).encode()).hexdigest()
    return body


def write_vocab(path, doc: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        old = json.loads(path.read_text())
        if [c["class"] for c in old["classes"]] != [c["class"] for c in doc["classes"]]:
            raise FileExistsError(f"{path} exists with a different class list; pass --overwrite to replace it")
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg, c1 = load_yaml(CFG), load_config()
    frozen = load_frozen_split(c1.path("splits"), cfg["split_version"])  # verifies the checksum
    lock = json.loads((c1.path("splits") / f"split_{cfg['split_version']}.lock.json").read_text())
    audit = pd.read_csv(c1.path("metadata") / "label_audit.csv")
    dup = pd.read_csv(c1.path("metadata") / "duplicate_content_groups.csv")
    full_map = json.loads((PROJECT_ROOT / "data" / "sequences" / "label_mapping.json").read_text())
    rep = PROJECT_ROOT / cfg["reports_dir"]
    tables = rep / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    s = add_file_columns(frozen)
    f = class_features(s, audit)
    f["exclusion_reasons"] = exclusion_reasons(f, cfg["rules"], cfg["heavy_share_threshold"])
    deferred = cfg["deferred_pending_inspection"]
    for w in deferred:
        if w in f.index:
            f.loc[w, "exclusion_reasons"] = "|".join(filter(None, [f.loc[w, "exclusion_reasons"],
                                                                   "deferred_pending_manual_inspection"]))
    f["primary_reason"] = f["exclusion_reasons"].str.split("|").str[0]
    pool = f[f["exclusion_reasons"] == ""]
    ranked = rank_candidates(pool, cfg["ranking"]["keys"], cfg["ranking"]["tie_break_seed"])
    n_dev, n_exp = cfg["sets"]["dev"]["size"], cfg["sets"]["expanded"]["size"]
    if len(ranked) < n_exp:
        raise ValueError(f"candidate pool has {len(ranked)} classes < expanded size {n_exp}")
    dev, exp = list(ranked.index[:n_dev]), list(ranked.index[:n_exp])
    ranked["in_dev_100"] = ranked.index.isin(dev)
    ranked["in_expanded"] = ranked.index.isin(exp)

    # ---------------- duplicate analysis
    pairs = label_pairs(s, f["clips"])
    ev = pairs.apply(pair_treatment, axis=1, result_type="expand")
    pairs["evidence"], pairs["recommended_treatment"] = ev[0], ev[1]
    comps = components(pairs)
    comp_df = pd.DataFrame([{"component": i, "labels": len(c), "clips_in_labels": int(f.loc[sorted(c), "clips"].sum()),
                             "members": "|".join(sorted(c))} for i, c in enumerate(comps)])
    shared_cls = f[f["cross_label_clips"] > 0].sort_values("cross_label_fraction", ascending=False)
    fully = shared_cls[shared_cls["cross_label_fraction"] >= 1.0]
    n_partners = pd.concat([pairs[["label_a", "label_b"]].rename(columns={"label_a": "w", "label_b": "o"}),
                            pairs[["label_b", "label_a"]].rename(columns={"label_b": "w", "label_a": "o"})]
                           ).groupby("w")["o"].nunique()
    shared_cls = shared_cls.assign(labels_shared_with=n_partners.reindex(shared_cls.index).fillna(0).astype(int))
    fully = shared_cls.loc[fully.index]

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    meta = {"created_utc": stamp, "created_by": "scripts/s5_01_select_vocabulary.py",
            "config": "configs/step5_vocabulary.yaml", "config_sha256": cfg_sha,
            "frozen_split": {"version": cfg["split_version"], "sha256": lock["sha256"]},
            "full_label_mapping_vocabulary_sha256": full_map["vocabulary_sha256"],
            "selection_rules": cfg["rules"], "ranking": cfg["ranking"],
            "note": "Subset of the frozen split; no re-splitting, no merged labels. Metadata-only selection: "
                    "it does not establish that the classes are visually distinguishable."}

    # ---------------- tables
    f.reset_index().to_csv(tables / "class_features.csv", index=False)
    f[f["exclusion_reasons"] != ""][FEATURE_COLS + ["audit_anomalies", "primary_reason", "exclusion_reasons"]
                                    ].reset_index().to_csv(tables / "excluded_classes.csv", index=False)
    ranked[["rank", "in_dev_100", "in_expanded"] + FEATURE_COLS].reset_index().to_csv(tables / "candidate_pool.csv",
                                                                                     index=False)
    pairs.to_csv(tables / "shared_label_pairs.csv", index=False)
    comp_df.to_csv(tables / "sharing_components.csv", index=False)
    shared_cls[FEATURE_COLS + ["labels_shared_with"]].reset_index().to_csv(tables / "classes_sharing_clips.csv",
                                                                          index=False)

    # synonym / merge candidates
    syn = []
    for r in pairs[pairs["min_fraction"] >= 0.95].itertuples():
        syn.append({"label_a": r.label_a, "label_b": r.label_b, "evidence": f"{r.shared_clips} identical clips = "
                    f"{pct(r.fraction_of_a)} of A, {pct(r.fraction_of_b)} of B", "recommended_treatment":
                    "merge if ever included; excluded from both vocabularies"})
    for r in pairs[(pairs["max_fraction"] >= 0.5) & (pairs["min_fraction"] < 0.95)].itertuples():
        syn.append({"label_a": r.label_a, "label_b": r.label_b, "evidence": f"{r.shared_clips} identical clips = "
                    f"{pct(r.fraction_of_a)} of A, {pct(r.fraction_of_b)} of B", "recommended_treatment":
                    "investigate manually; excluded from both vocabularies"})
    su = f.loc[["set up", "setup"]]
    syn.append({"label_a": "set up", "label_b": "setup", "evidence":
                f"near-identical label strings; 0 shared clips; 'set up' {su.loc['set up', 'clips']} clips (all "
                f"word_video_n, {su.loc['set up', 'cross_label_clips']} shared with other labels), 'setup' "
                f"{su.loc['setup', 'clips']} clips (all numericid-WORD, {su.loc['setup', 'val_units']} val / "
                f"{su.loc['setup', 'test_units']} test units)",
                "recommended_treatment": "investigate manually (likely one sign, two source collections); excluded"})
    for w, why in deferred.items():
        syn.append({"label_a": w, "label_b": "(homonym senses)", "evidence": f"lexical only: {why}",
                    "recommended_treatment": "investigate manually; deferred from both vocabularies"})
    for a, b in cfg["related_pairs_kept_separate"]:
        where = "dev_100" if a in dev and b in dev else ("expanded" if a in exp and b in exp else None)
        if where:
            syn.append({"label_a": a, "label_b": b, "evidence": f"lexically related; 0 shared clips; both in {where}",
                        "recommended_treatment": "keep separate; check their confusions after training"})
    syn_df = pd.DataFrame(syn)
    syn_df.to_csv(tables / "synonym_merge_candidates.csv", index=False)

    # ---------------- vocabulary files
    vdev = vocabulary_doc("dev_100", dev, f, ranked, full_map, meta)
    vexp = vocabulary_doc("expanded_300", exp, f, ranked, full_map, {**meta, "superset_of": "dev_100"})
    write_vocab(PROJECT_ROOT / cfg["sets"]["dev"]["output"], vdev, args.overwrite)
    write_vocab(PROJECT_ROOT / cfg["sets"]["expanded"]["output"], vexp, args.overwrite)

    # ---------------- summary + report
    reason_counts = f.loc[f["exclusion_reasons"] != "", "exclusion_reasons"].str.split("|").explode().value_counts()
    primary_counts = f.loc[f["exclusion_reasons"] != "", "primary_reason"].value_counts()
    summ = {
        "dataset": {"classes": len(f), "videos": len(s), "unique_contents": int(s["sha256"].nunique()),
                    "independent_split_units": int(s["split_group"].nunique())},
        "duplicates": {"multi_file_groups": int(len(dup)), "files_in_multi_file_groups": int(dup["n_files"].sum()),
                       "cross_label_groups": int((dup["n_labels"] > 1).sum()),
                       "files_in_cross_label_groups": int(dup.loc[dup["n_labels"] > 1, "n_files"].sum()),
                       "classes_fully_shared": len(fully), "classes_sharing_any_clip": len(shared_cls),
                       "classes_heavily_shared_ge_50pct": int((shared_cls["cross_label_fraction"] >= 0.5).sum()),
                       "label_pairs": len(pairs), "label_pairs_sharing_1_clip": int((pairs["shared_clips"] == 1).sum()),
                       "label_pairs_near_identical_ge_95pct_both": int((pairs["min_fraction"] >= 0.95).sum()),
                       "sharing_components": len(comps), "largest_component_labels": len(comps[0]),
                       "shared_files_by_source_pattern": s[s["cross_label"]]["source_proxy"].value_counts().to_dict(),
                       "within_class_duplicate_files": int(s["within_dup"].sum()),
                       "classes_with_within_class_duplicates": int((f["within_dup_clips"] > 0).sum())},
        "batches": {"word_timestamp_n_files": int(s["batch"].notna().sum()), "batches": int(s["batch"].nunique()),
                    "batches_spanning_splits": int((s[s["batch"].notna()].groupby("batch")["split"].nunique() > 1).sum()),
                    "classes_with_batches": int((f["max_batch_clips"] > 0).sum()),
                    "classes_over_50pct_one_batch": int((f["max_batch_share"] > 0.5).sum())},
        "exclusion_reason_counts": reason_counts.to_dict(), "primary_reason_counts": primary_counts.to_dict(),
        "candidate_pool": len(ranked), "dev_100": vdev["summary"], "expanded_300": vexp["summary"],
        "dev_vocabulary_sha256": vdev["vocabulary_sha256"], "expanded_vocabulary_sha256": vexp["vocabulary_sha256"],
        "config_sha256": cfg_sha, "created_utc": stamp}
    (rep / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")

    def cand_table(words):
        t = f.loc[sorted(words)].reset_index()
        t["shared %"] = (100 * t["cross_label_fraction"]).round(1)
        t["aug %"] = (100 * t["augmented_fraction"]).round(1)
        t["rank"] = t["word"].map(ranked["rank"])
        return md_table(t, ["word", "rank", "clips", "train_clips", "val_clips", "test_clips", "unique_clips", "units",
                            "shared %", "aug %", "source_patterns"])

    def set_block(title, v):
        m = v["summary"]
        return [f"## {title}\n", md_table(pd.DataFrame([{
            "classes": m["classes"], "clips": m["clips"], "unique clips": m["unique_clips"],
            "independent units": m["independent_units"], "train / val / test clips":
            f"{m['train_clips']} / {m['val_clips']} / {m['test_clips']}", "train / val / test units":
            f"{m['train_units']} / {m['val_units']} / {m['test_units']}", "shared clips": m["cross_label_shared_clips"],
            "augmented clips": m["augmented_clips"], "within-class dup clips": m["within_class_duplicate_clips"],
            "clips/class min-median-max": f"{m['clips_per_class']['min']} / {m['clips_per_class']['median']:.0f} / "
                                          f"{m['clips_per_class']['max']}",
            "imbalance (max/min)": round(m["class_imbalance_ratio"], 2),
            "min val / test units per class": f"{m['min_val_units_per_class']} / {m['min_test_units_per_class']}"}])),
            "", f"Classes ({m['classes']}): " + ", ".join(f"`{c['class']}`" for c in v["classes"]), ""]

    d, b = summ["duplicates"], summ["batches"]
    top_shared = shared_cls.head(40).reset_index()
    top_shared["shared %"] = (100 * top_shared["cross_label_fraction"]).round(1)
    L = [
        "# Step 5 - evidence-driven subset vocabulary\n",
        f"Generated by `scripts/s5_01_select_vocabulary.py` ({stamp}) from metadata only: frozen split "
        f"`{cfg['split_version']}` (checksum verified), manifest, duplicate-content groups, label audit. "
        "No video was downloaded. Config: `configs/step5_vocabulary.yaml`. Tables: `reports/step5/tables/`.\n",
        "> Metadata can show that clips are byte-identical, that classes are balanced across the frozen split and "
        "where their files come from. It **cannot** show that two classes look different, that a label is used "
        "consistently, or which signer made a clip. Nothing below claims visual distinguishability.\n",
        "# 1. Duplicate analysis\n",
        md_table(pd.DataFrame([{"classes": summ["dataset"]["classes"], "videos": summ["dataset"]["videos"],
                                "distinct contents (SHA-256)": summ["dataset"]["unique_contents"],
                                "independent split units": summ["dataset"]["independent_split_units"],
                                "multi-file groups": d["multi_file_groups"], "files in them": d["files_in_multi_file_groups"],
                                "cross-label groups": d["cross_label_groups"],
                                "files in cross-label groups": d["files_in_cross_label_groups"]}])), "",
        f"- Classes sharing >= 1 clip with another label: **{d['classes_sharing_any_clip']}**; ≥ 50 % of their "
        f"clips: **{d['classes_heavily_shared_ge_50pct']}**; 100 %: **{d['classes_fully_shared']}**.",
        f"- {d['label_pairs']} label pairs share clips; {d['label_pairs_sharing_1_clip']} of them share exactly one "
        f"clip; {d['label_pairs_near_identical_ge_95pct_both']} pairs are ≥ 95 % identical in both directions.",
        f"- The sharing graph has {d['sharing_components']} connected components; the largest links "
        f"**{d['largest_component_labels']} labels**.",
        f"- Shared files by filename collection: {d['shared_files_by_source_pattern']}. The systematic sharing is "
        "the `word_video_n` collection: synonym labels hold the same `_video_<n>` file "
        "(e.g. `batch_video_29` = `deal_video_29` = `lot_video_29` …). The label sets match dictionary synonym sets "
        "(\"animate being / beast / brute / creature\"), consistent with the collection having been expanded with "
        "synonyms that reuse one recording - an inference from the label sets, not documented by the dataset.",
        "- One-clip pairs are mostly between unrelated words (`cake`/`hide`, `color`/`pink`, `busy`/`job`): the "
        "same file under two unrelated labels means at least one label is wrong for that clip (label noise).",
        f"- Within-class duplicates (same bytes twice under one label): {d['within_class_duplicate_files']} files "
        f"in {d['classes_with_within_class_duplicates']} classes (inflate counts; harmless for leakage because a "
        "SHA-256 group never spans splits).",
        "- Equal file size with different SHA-256 was checked as a near-duplicate signal and rejected: 5,893 sizes "
        "collide, but only 4 within one word - the rest is chance across unrelated words.\n",
        f"## 1.1 The {d['classes_fully_shared']} fully shared classes\n",
        md_table(fully.reset_index().assign(**{"shared %": lambda t: (100 * t["cross_label_fraction"]).round(1)}),
                 ["word", "clips", "cross_label_clips", "shared %", "labels_shared_with", "source_patterns"]), "",
        f"## 1.2 Classes sharing clips - 40 highest shares (all {len(shared_cls)}: "
        "`tables/classes_sharing_clips.csv`)\n",
        md_table(top_shared, ["word", "clips", "cross_label_clips", "shared %", "labels_shared_with"]), "",
        "## 1.3 Label pairs with the most overlap (all: `tables/shared_label_pairs.csv`)\n",
        md_table(pairs.head(25), ["label_a", "label_b", "shared_clips", "fraction_of_a", "fraction_of_b", "evidence"]),
        "", "## 1.4 Largest sharing components (all: `tables/sharing_components.csv`)\n",
        md_table(comp_df.head(10).assign(members=lambda t: t["members"].str.slice(0, 160) + "…"),
                 ["component", "labels", "clips_in_labels", "members"]), "",
        "## 1.5 Timestamp batches (a leakage risk metadata cannot resolve)\n",
        f"{b['word_timestamp_n_files']:,} files are named `<word>_<date>_<time>_<n>.mp4`. They form "
        f"{b['batches']} batches, each of one word; {b['batches_spanning_splits']} batches are spread over "
        f"train/val/test. {b['classes_over_50pct_one_batch']} classes take more than half their clips from a "
        "single batch. Neighbouring words carry consecutive timestamps (`comb_…172637`, `cool_…172638`), so the "
        "timestamp is most likely the time a script processed the word, not a recording session - but if a "
        "batch is one signer or one recording, its val/test clips are not independent of its train clips. "
        "**Requires manual inspection.** The vocabularies cap any one batch at 50 % of a class.\n",
        "# 2. Problem classes\n",
        "Every class gets all applicable reasons (`tables/excluded_classes.csv`, `tables/class_features.csv`).\n",
        md_table(pd.DataFrame({"reason": reason_counts.index, "classes (any)": reason_counts.values,
                               "classes (primary)": [int(primary_counts.get(k, 0)) for k in reason_counts.index]})),
        "",
        "| reason | rule | why it would mislead evaluation |\n|---|---|---|\n"
        "| augmentation_dominated / augmented_copies | any `*_aug_<n>` file | 26 letter classes: 74-90 % of files "
        "are augmented copies of 5-13 source clips; accuracy would measure copy recognition |\n"
        "| fully / heavily / shares_clips_with_other_labels | any byte-identical clip under another label | the same "
        "input carries two labels: identical clips cannot both be classified correctly, or one label is noise |\n"
        "| multiword_label | space, `_` or `-` in the label | phrases and synonym expansions (`whole slew`, `set up`) |\n"
        "| label_audit_anomaly | flagged in `label_audit.csv` | digits, classifier gloss, `TRUE`, repeated words, "
        "concatenations - unclear semantics |\n"
        "| too_few_val_units / too_few_test_units | < 6 independent units | per-class val/test metrics from < 6 "
        "clips are mostly noise |\n"
        "| imbalanced_split | train share outside 60-80 % | (no class violates this in split v1) |\n"
        "| within_class_duplicates | > 10 % same-bytes copies | inflated counts |\n"
        "| single_batch_dominated | > 50 % from one timestamp batch | possible same-recording leakage (§1.5) |\n"
        "| single_source_collection | 1 filename collection | no recording diversity evidence at all |\n"
        "| deferred_pending_manual_inspection | config list | English homonym; one label may hold two signs |\n",
        "`set up` / `setup`: `set up` fails multiword_label, label_audit_anomaly, shares_clips_with_other_labels "
        f"({su.loc['set up', 'cross_label_clips']} clips) and single_source_collection; `setup` fails "
        "label_audit_anomaly, too_few_val_units / too_few_test_units and single_source_collection. They share no "
        "clip and come from different collections (`set up`: word_video_n; `setup`: numericid-WORD), so the same "
        "sign is probably split over two labels. Both are excluded.\n",
        "# 3. Candidate pool\n",
        f"**{len(ranked)} classes** pass every rule. Ranking (descending, lexicographic): min(val units, test units) "
        "→ number of source collections → independent units → `sha256(\"42:<word>\")`. Full per-class table: "
        "`tables/candidate_pool.csv` (class, total/train/val/test clips, unique clips, shared %, aug %, …). Every "
        "pool class has 0 % shared and 0 % augmented clips by construction.\n",
        *set_block("4. Dev vocabulary - 100 classes (`configs/vocabulary_100.json`)", vdev),
        cand_table(dev), "",
        *set_block("5. Expanded vocabulary - 300 classes (`configs/vocabulary_expanded.json`, superset of dev)", vexp),
        "Per-class rows for the 200 added classes: `tables/candidate_pool.csv` (`in_expanded` and not "
        "`in_dev_100`).\n",
        "The two sets have different purposes: dev_100 is for getting a first real train → val → test result; "
        "expanded_300 is for checking how the same pipeline scales once dev_100 works. Expanded is a superset so "
        "dev results carry over.\n",
        "# 6. Synonym / merge candidates (`tables/synonym_merge_candidates.csv`)\n",
        "No label is merged in either vocabulary. Pairs below are treatments recommended for later, if those "
        "classes are ever included.\n",
        md_table(syn_df[syn_df["recommended_treatment"].str.startswith("merge")].head(30)), "",
        f"(+{max(0, int(syn_df['recommended_treatment'].str.startswith('merge').sum()) - 30)} more near-identical "
        "pairs in the CSV)\n",
        md_table(syn_df[~syn_df["recommended_treatment"].str.startswith("merge")
                        & ~syn_df["evidence"].str.contains("identical clips")]), "",
        f"Heavily overlapping (≥ 50 % of one class) but not near-identical pairs: "
        f"{int(syn_df['recommended_treatment'].str.startswith('investigate manually; excluded').sum())} in the CSV, "
        "all `investigate manually` and excluded.\n",
        "# 7. Manual inspection still required\n",
        "1. Timestamp batches (§1.5): watch a few clips of one batch (e.g. `peas_20241119_195754_*`) - same signer "
        "/ same recording? If yes, batch-aware splitting is needed for any class that relies on batches.\n",
        "2. Homonym labels deferred in the config (`sign`, `fly`, `lie`, `cold`, `charge`, `wave`, `major`): do "
        "their clips show one sign or several?\n",
        "3. `set up` vs `setup`, and the near-identical synonym clusters, only if they are ever wanted.\n",
        "4. `numericid-WORD` filenames (55,751 files; present in every dev class) look like the naming of the "
        "public ASL Citizen dataset, which publishes participant IDs. If confirmed, signer IDs could be recovered "
        "and a signer-independent split built. This is a hypothesis from the filename pattern only; checking it "
        "needs that dataset's metadata (not downloaded).\n",
    ]
    (rep / "vocabulary_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("dataset", "candidate_pool", "dev_100", "expanded_300")}, indent=1))
    print("primary exclusion reasons:", summ["primary_reason_counts"])
    print(f"report -> {(rep / 'vocabulary_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
