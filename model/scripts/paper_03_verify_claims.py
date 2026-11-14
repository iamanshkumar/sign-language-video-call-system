"""Verify the numbers in the paper draft against the saved source reports (no model is run).

  PYTHONPATH=src .venv/bin/python scripts/paper_03_verify_claims.py \
      --draft reports/paper/draft/sections.md --out reports/paper/draft/claims_verification.md

Each claim recomputes a value from its source file, formats it exactly as written in the draft, and checks
that this text occurs in the draft. Exit code 1 if any claim fails. Overwrites only its own --out file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from asl.config import PROJECT_ROOT

R = PROJECT_ROOT / "reports"


def j(p):
    return json.loads((R / p).read_text())


def c(p):
    return pd.read_csv(R / p)


def p1(x):
    return f"{100 * x:.1f}"


def f3(x):
    return f"{x:.3f}"


def sg(x, d=3):
    return f"{x:+.{d}f}"


def claims() -> list[tuple[str, str, str]]:
    """(description, expected text in the draft, source)."""
    out = []
    idx = pd.read_parquet(PROJECT_ROOT / "data/sequences/dev100/index.parquet")
    per = idx[idx.split == "train"].groupby("word").size()
    out += [("classes / clips", f"{idx.word.nunique()} ASL word classes and {len(idx):,} accepted video clips", "index.parquet"),
            ("split", "{:,} training,\n725 validation and 706 test".format((idx.split == "train").sum()).replace("\n", " "), "index.parquet"),
            ("clips per class", f"median of {int(per.median())} training clips per class (range {per.min()}–{per.max()})", "index.parquet")]
    sm = c("e0/bootstrap/seed_mean.csv").set_index(["split", "model"])
    b, l = sm.loc[("test", "bilstm")], sm.loc[("test", "lstm")]
    dd = c("e0/bootstrap/bilstm_minus_lstm.csv")
    dt = dd[(dd.split == "test") & (dd.seed == "mean")].iloc[0]
    out += [("BiLSTM test acc", f"{p1(b.accuracy_mean)} ± {p1(b.accuracy_sd)} %\n(95 % CI {p1(b.acc_clip_lo)}–{p1(b.acc_clip_hi)})", "e0/bootstrap/seed_mean.csv"),
            ("LSTM test acc", f"{p1(l.accuracy_mean)} ± {p1(l.accuracy_sd)} % for the LSTM", "e0/bootstrap/seed_mean.csv"),
            ("test macro-F1 both", f"{f3(b.macro_f1_mean)} ± {f3(b.macro_f1_sd)} and\n{f3(l.macro_f1_mean)} ± {f3(l.macro_f1_sd)}", "e0/bootstrap/seed_mean.csv"),
            ("paired test acc", f"{100 * dt.d_accuracy:+.1f} percentage points in accuracy (95 % CI {100 * dt.d_acc_lo:+.1f} to {100 * dt.d_acc_hi:+.1f})", "e0/bootstrap/bilstm_minus_lstm.csv"),
            ("paired test F1", f"{sg(dt.d_macro_f1)} in macro-F1 ({sg(dt.d_f1_lo)} to {sg(dt.d_f1_hi)})", "e0/bootstrap/bilstm_minus_lstm.csv")]
    hs = j("paper/hands_only/comparison/summary.json")
    out += [("hands-only means", f"from {f3(hs['mean_val_macro_f1']['hands_pose'])} to {f3(hs['mean_val_macro_f1']['hands'])}", "paper/hands_only/comparison/summary.json"),
            ("hands-only change", f"{sg(hs['mean_difference_hands_minus_hands_pose'])} (95 % CI {sg(hs['difference_ci'][0])} to {sg(hs['difference_ci'][1])})", "paper/hands_only/comparison/summary.json"),
            ("hands-only per class", f"{hs['per_class']['hands_better']} classes improved and {hs['per_class']['hands_worse']}\ndeclined", "paper/hands_only/comparison/summary.json"),
            ("hands-only seeds", "all five seeds higher" if hs["seeds_hands_lower"] == 0 else "MISMATCH", "paper/hands_only/comparison/summary.json")]
    hpc = c("paper/hands_only/comparison/per_class_val_f1.csv").nsmallest(3, "diff").word.str.upper().tolist()
    out += [("hands-only largest declines", f"{hpc[0]},\n{hpc[1]} and {hpc[2]}", "paper/hands_only/comparison/per_class_val_f1.csv")]
    e2 = j("e2/comparison/summary.json")["rule"]
    e2s = c("e2/comparison/per_seed.csv").groupby("model").mean(numeric_only=True)
    out += [("E2 val means", f"from {f3(e2s.loc['e1', 'val_macro_f1'])} to {f3(e2s.loc['e2', 'val_macro_f1'])}, a paired change of {sg(e2['mean_change'])} (95 % CI {sg(e2['mean_change_ci'][0])} to {sg(e2['mean_change_ci'][1])})", "e2/comparison/summary.json"),
            ("E2 non-inferiority met / improvement not met", "met the predefined non-inferiority rule" if e2["non_inferiority_met"] and not e2["improvement_rule_met"] else "MISMATCH", "e2/comparison/summary.json")]
    ch = c("e2/comparison/val_change_paired.csv").val_macro_f1_change
    out += [("E2 per-seed range", f"from {ch.min():+.3f} to {ch.max():+.3f}, with {int((ch > 0).sum())} of five seeds higher".replace(" 4 of", " four of"), "e2/comparison/val_change_paired.csv")]
    ex = c("step4/results/experiment_results.csv").set_index("experiment_id")
    out += [("w45 vs w30", f"{f3(ex.loc['hands_pose_w45_lstm_e100', 'best_val_macro_f1'])} against {f3(ex.loc['hands_pose_w30_lstm_e100', 'best_val_macro_f1'])}", "step4/results/experiment_results.csv")]
    e1 = j("e1/comparison/summary.json")["decision_rule"]
    e1s = c("e1/comparison/per_seed.csv").groupby("model").mean(numeric_only=True)
    out += [("E1 val", f"from {f3(e1s.loc['ref', 'val_macro_f1'])} to {f3(e1s.loc['e1', 'val_macro_f1'])}, a paired gain of {sg(e1['mean_gain'])} (95 % CI {sg(e1['mean_gain_ci'][0])} to {sg(e1['mean_gain_ci'][1])})", "e1/comparison/summary.json"),
            ("E1 all seeds / rule met", "all five seeds improved" if e1["seeds_better"] == 5 and e1["all_criteria_met"] else "MISMATCH", "e1/comparison/summary.json")]
    fm = c("paper/final_test/seed_mean.csv").set_index("model")
    fp = c("paper/final_test/paired.csv").set_index("comparison")
    out += [("E1 test acc", f"{p1(fm.loc['e1', 'accuracy_mean'])} ± {p1(fm.loc['e1', 'accuracy_sd'])} % for E1 (95 % CI {p1(fm.loc['e1', 'acc_lo'])}–{p1(fm.loc['e1', 'acc_hi'])})", "paper/final_test/seed_mean.csv"),
            ("E2 test acc", f"{p1(fm.loc['e2', 'accuracy_mean'])} ± {p1(fm.loc['e2', 'accuracy_sd'])} % for E2 ({p1(fm.loc['e2', 'acc_lo'])}–{p1(fm.loc['e2', 'acc_hi'])})", "paper/final_test/seed_mean.csv"),
            ("baseline test acc (reused)", f"against {p1(fm.loc['baseline', 'accuracy_mean'])} ± {p1(fm.loc['baseline', 'accuracy_sd'])} % for the\nbaseline", "paper/final_test/seed_mean.csv"),
            ("test macro-F1", f"{f3(fm.loc['e1', 'macro_f1_mean'])} ± {f3(fm.loc['e1', 'macro_f1_sd'])} for E1 and {f3(fm.loc['e2', 'macro_f1_mean'])} ± {f3(fm.loc['e2', 'macro_f1_sd'])} for E2, against {f3(fm.loc['baseline', 'macro_f1_mean'])} ± {f3(fm.loc['baseline', 'macro_f1_sd'])}", "paper/final_test/seed_mean.csv")]
    r = fp.loc["e2 - baseline"]
    out += [("E2-baseline", f"{100 * r.d_accuracy:+.1f} percentage points in accuracy (95 % CI {100 * r.d_acc_lo:+.1f} to {100 * r.d_acc_hi:+.1f}) and\n{sg(r.d_macro_f1)} in macro-F1 ({sg(r.d_f1_lo)} to {sg(r.d_f1_hi)})", "paper/final_test/paired.csv")]
    r1, r2 = fp.loc["e1 - baseline"], fp.loc["e2 - e1"]
    out += [("E1-baseline", f"{100 * r1.d_accuracy:+.1f} points ({100 * r1.d_acc_lo:+.1f} to {100 * r1.d_acc_hi:+.1f})", "paper/final_test/paired.csv"),
            ("E2-E1 test", f"{100 * r2.d_accuracy:+.1f} points ({100 * r2.d_acc_lo:+.1f} to {100 * r2.d_acc_hi:+.1f})", "paper/final_test/paired.csv"),
            ("E2-E1 discussion", f"{100 * r2.d_accuracy:+.1f} points of accuracy and {sg(r2.d_macro_f1)} macro-F1", "paper/final_test/paired.csv"),
            ("all seeds each step", "Each step was better in all five seeds" if (fp.seeds_better_macro_f1 == 5).all() else "MISMATCH", "paper/final_test/paired.csv")]
    pc = c("paper/final_test/per_class.csv")
    out += [("E2 per class test", f"for {int((pc.e2_f1 > pc.baseline_f1).sum())} classes and reduced it for {int((pc.e2_f1 < pc.baseline_f1).sum())}", "paper/final_test/per_class.csv")]
    gaps = {m: c(f).query("model == @m").train_minus_val_acc.mean() for m, f in (("ref", "e1/comparison/per_seed.csv"), ("e1", "e1/comparison/per_seed.csv"))}
    gaps["e2"] = c("e2/comparison/per_seed.csv").query("model == 'e2'").train_minus_val_acc.mean()
    out += [("gaps", f"from {p1(gaps['ref'])} points (baseline) to {p1(gaps['e1'])} (E1) and {p1(gaps['e2'])} (E2)", "e1/e2 per_seed.csv")]
    rb0 = c("e0/robustness/by_model.csv").query("model == 'bilstm'").set_index("perturbation").d_accuracy_mean
    rb1 = c("e1/comparison/robustness_e1_vs_ref.csv").set_index("perturbation")["mean"]
    rb2 = c("e2/comparison/robustness_e2.csv").set_index("perturbation")["mean"]
    out += [("hold_start", f"from {100 * rb0['hold_start']:.1f} to {100 * rb1['hold_start']:.1f}\npoints", "robustness"),
            ("repeat", f"from {100 * rb0['repeat_x2']:.1f} to {100 * rb1['repeat_x2']:.1f} points", "robustness"),
            ("hand loss / mirror", f"({100 * rb0['drop_minor_hand']:.1f} points for the baseline) and to mirroring ({100 * rb0['mirror']:.1f})", "robustness"),
            ("hips", f"({100 * rb0['hips_visible']:.1f} to {abs(100 * rb2['hips_visible']):.1f} points)", "robustness")]
    L = j("step6/latency_hands_pose_w30_bilstm_e100_s3.json")
    lm = ex.reset_index()
    lm = lm[lm.experiment_id.str.match(r"hands_pose_w30_(lstm|bilstm)_e100(_s\d)?$")]
    out += [("mediapipe", f"median of {L['mediapipe_per_frame']['median_ms']:.1f} ms per frame (p95 {L['mediapipe_per_frame']['p95_ms']:.1f} ms)", "latency json"),
            ("max fps", f"about {round(L['mediapipe_max_fps_single_process'])} frames per second", "latency json"),
            ("bilstm ms", f"median of\n{L['model_only_forward']['median_ms']:.2f} ms per sign", "latency json"),
            ("training-run latencies", f"LSTM took {lm[lm.model == 'lstm'].median_latency_ms.median():.2f} ms and the BiLSTM {lm[lm.model == 'bilstm'].median_latency_ms.median():.2f} ms", "experiment_results.csv"),
            ("segment close", f"median of {L['segment_close_preprocess_plus_model']['median_ms']:.1f} ms", "latency json"),
            ("push", f"bookkeeping {L['push_per_frame_no_segment']['median_ms']:.3f} ms", "latency json")]
    fps = {k: j(f"step6/webcam_eval/{v}_camera_S01/summary.json")["input_diagnostics"]["processed_fps"]
           for k, v in (("S2", "20261019T075328Z"), ("S3", "20261019T084115Z"), ("S4", "20261020T045903Z"))}
    out += [("fps", f"processing to {fps['S2']:.1f} frames per second", "session summaries"),
            ("fps threaded", f"raised this to {fps['S3']:.1f}–{fps['S4']:.1f}", "session summaries")]
    dl = []
    for v in ("20261019T084115Z", "20261020T045903Z"):
        a = c(f"step6/webcam_eval/{v}_camera_S01/attempts.csv")
        dl.append(a[a.forced_cut == False].delay_s.median())  # noqa: E712
    out += [("delay", f"median of\n{min(dl):.2f}–{max(dl):.2f} s", "attempts.csv")]
    sb = c("e0/webcam_replay_set/summary_by_model.csv").query("scope == 'core_valid'")
    w1, w2 = c("e1/comparison/webcam_replay_paired.csv"), c("e2/comparison/webcam_replay_paired.csv")
    l_, b_ = sb[sb.model.str.match(r"lstm_s")], sb[sb.model.str.match(r"bilstm_s")]
    out += [("webcam top1", f"{p1(l_.top1.mean())} % for the LSTM, {p1(b_.top1.mean())} % for the BiLSTM baseline, {p1(w1.e1_top1.mean())} % for E1 and {p1(w2.e2_top1.mean())} % for E2", "replay set / e1 / e2"),
            ("webcam top5", f"{p1(l_.top5.mean())} %, {p1(b_.top5.mean())} %, {p1(w1.e1_top5.mean())} % and {p1(w2.e2_top5.mean())} %", "replay set / e1 / e2"),
            ("no significant seed", "no\nstatistically significant difference" if (w1.exact_mcnemar_p.min() > 0.05 and w2.exact_mcnemar_p.min() > 0.05) else "MISMATCH", "webcam_replay_paired.csv"),
            ("webcam attempts", f"{int(b_.attempts.iloc[0])} webcam attempts", "replay set")]
    s3 = j("step6/webcam_eval/20261019T084115Z_camera_S01/replay_firsthand/summary.json")
    s4 = j("step6/webcam_eval/20261020T045903Z_camera_S01/summary.json")
    out += [("sessions", f"{s3['outcomes']['correct']} and {s4['outcomes']['correct']} of 20 attempts were correct", "session summaries"),
            ("session dups", f"{s3['attempts_with_duplicates']} and {s4['attempts_with_duplicates']} attempts\nproduced a duplicate", "session summaries"),
            ("session missed", "no sign was missed" if s3["outcomes"].get("missed", 0) == 0 and s4["outcomes"].get("missed", 0) == 0 else "MISMATCH", "session summaries")]
    P = c("paper/postprocessing/summary_by_setting.csv")
    d0 = P[(P.cooldown_s == 0) & (P.confidence_threshold == 0)].iloc[0]
    out += [("pp default", f"{int(d0.correct)} attempts were\ncorrect and {int(d0.attempts_with_duplicates)} attempts produced one duplicate emission each" if d0.attempts_with_duplicates == d0.duplicate_emissions else "MISMATCH", "postprocessing"),
            ("pp cooldown", "removed all 14 duplicate emissions" if (P[P.cooldown_s > 0].duplicate_emissions == 0).all() and (P[P.cooldown_s > 0].correct == d0.correct).all() else "MISMATCH", "postprocessing")]
    for t in (0.4, 0.57, 0.7):
        r = P[(P.cooldown_s == 0) & (P.confidence_threshold == t)].iloc[0]
        out.append((f"threshold {t}", f"{p1(r.coverage)} % of attempts were answered with {p1(r.precision_of_answers)} % precision" if t == 0.4
                    else f"{p1(r.coverage)} % with {p1(r.precision_of_answers)} %", "postprocessing"))
    out += [("pp no missed", "No sign was missed under any setting" if (P.no_segment == 0).all() else "MISMATCH", "postprocessing")]
    C = j("step6/calibration_hands_pose_w30_bilstm_e100_s3.json")["test_check_at_that_threshold"]
    out += [("calibration test", f"{p1(C['test_accuracy_of_accepted'])} % precision at {p1(C['test_coverage'])} % coverage", "calibration json")]
    wd = j("paper/draft/webcam_details/summary.json")
    nr = wd["never_or_rarely_recognised"]
    out += [("duplicate causes", f"of the {wd['duplicate_emissions']} duplicate emissions in the replay below, {wd['continuations_of_cut_sign']} were such\ncontinuations and one was a second burst of movement starting {wd['other'][0]['gap_after_answer_last_hand_s']:.2f} s"
             if len(wd["other"]) == 1 else "MISMATCH", "paper/draft/webcam_details/summary.json"),
            ("never recognised", "AFRICA and\nCOMPUTER were never recognised by any of the 20 models" if nr["africa"]["correct"] == 0 and nr["computer"]["correct"] == 0 else "MISMATCH",
             "paper/draft/webcam_details/summary.json"),
            ("forbid once", f"FORBID\nwas recognised once (by one E2 seed) in {nr['forbid']['model_attempts']} model-attempts" if nr["forbid"]["correct"] == 1 else "MISMATCH",
             "paper/draft/webcam_details/summary.json")]
    ex2 = c("step4/results/experiment_results.csv")
    out += [("test uses", f"{int(ex2[ex2.store.eq('dev100_hp')].test_accuracy.notna().sum())} training runs and three auxiliary checks", "experiment_results.csv; T12")]
    return out


def manuscript_claims() -> list[tuple[str, str, str]]:
    """Additional claims of the full manuscript (manuscript.md): data counts, models, precision / recall."""
    out = []
    S = json.loads((PROJECT_ROOT / "data/sequences/dev100/index_summary.json").read_text())
    rr = pd.DataFrame(S["excluded_by_split_reason"]).groupby("rejection_reason").n.sum()
    out += [("candidates / accepted / rejected", f"Of {S['landmark_videos']:,} candidate clips of the 100 classes, {S['accepted_indexed']:,} were accepted and {S['excluded']} rejected", "index_summary.json"),
            ("abstract rejected", f"({S['excluded']} of {S['landmark_videos']:,} candidate clips", "index_summary.json"),
            ("rejection reasons", f"{rr['active_both_hands_missing_fraction']} for both hands missing in more than 30 % of the active interval, {rr['too_few_decoded_frames']} for fewer than 30 decoded frames, {rr['active_both_hands_missing_run']} for both hands missing for more than 15 consecutive frames, and {rr['no_active_signing_interval']} for having no active signing interval", "index_summary.json")]
    idx = pd.read_parquet(PROJECT_ROOT / "data/sequences/dev100/index.parquet")
    mins = idx.groupby("split").word.value_counts().groupby("split").min()
    out += [("per-split minimum", f"at least {mins['train']} training, {mins['val']} validation and {mins['test']} test clips per class", "index.parquet")]
    st = j("step1/stats.json")["full_metadata"]
    out += [("full collection", f"{st['files']:,} video files labelled with {st['classes']:,} word classes", "step1/stats.json")]
    meta = {n: json.loads((PROJECT_ROOT / "experiments/dev100_hp" / n / "metadata.json").read_text()) for n in
            ("hands_pose_w30_lstm_e100", "hands_pose_w30_bilstm_e100", "hands_w30_bilstm_e100", "hands_pose_w30_bilstm_e100_augtemp_upperpose")}
    out += [("parameters", f"the LSTM has {meta['hands_pose_w30_lstm_e100']['params']:,} and the BiLSTM {meta['hands_pose_w30_bilstm_e100']['params']:,} trainable parameters; the hands-only BiLSTM has {meta['hands_w30_bilstm_e100']['params']:,} and the E2 BiLSTM {meta['hands_pose_w30_bilstm_e100_augtemp_upperpose']['params']:,}", "experiments metadata.json"),
            ("E2 features", f"leaving {meta['hands_pose_w30_bilstm_e100_augtemp_upperpose']['feature_dim']} features", "E2 metadata.json")]
    out += [("E1 unrounded", f"The unrounded mean gain is {j('e1/comparison/summary.json')['decision_rule']['mean_gain']:.5f}, i.e. {j('e1/comparison/summary.json')['decision_rule']['mean_gain']:.4f}", "e1/comparison/summary.json")]
    P = c("paper/final_test_prf/seed_mean.csv").set_index("model")
    ms = lambda m, k: f"{P.loc[m, k + '_mean']:.3f} ± {P.loc[m, k + '_sd']:.3f}"
    gap = P.macro_precision_mean - P.macro_recall_mean
    rec = c("paper/final_test_prf/reconciliation.csv")
    out += [("macro-precision", f"{ms('baseline', 'macro_precision')} for the baseline, {ms('e1', 'macro_precision')} for E1 and {ms('e2', 'macro_precision')} for E2", "paper/final_test_prf/seed_mean.csv"),
            ("macro-recall", f"mean macro-recall was {ms('baseline', 'macro_recall')}, {ms('e1', 'macro_recall')} and {ms('e2', 'macro_recall')}", "paper/final_test_prf/seed_mean.csv"),
            ("LSTM precision / recall", f"macro-precision was {ms('lstm', 'macro_precision')} and macro-recall {ms('lstm', 'macro_recall')}", "paper/final_test_prf/seed_mean.csv"),
            ("abstract E2 P/R", f"{P.loc['e2', 'macro_precision_mean']:.3f} macro-precision and {P.loc['e2', 'macro_recall_mean']:.3f} macro-recall", "paper/final_test_prf/seed_mean.csv"),
            ("conclusion E2 P/R", f"macro-precision {P.loc['e2', 'macro_precision_mean']:.3f} and macro-recall {P.loc['e2', 'macro_recall_mean']:.3f}", "paper/final_test_prf/seed_mean.csv"),
            ("P-R gap", f"Macro-precision exceeded macro-recall by {gap.min():.3f} to {gap.max():.3f} for every model" if (gap > 0).all() else "MISMATCH", "paper/final_test_prf/seed_mean.csv"),
            ("same ordering", "The models were ordered identically by precision, recall and F1" if
             all(list(P[k + "_mean"].sort_values().index) == list(P.macro_f1_mean.sort_values().index) for k in ("macro_precision", "macro_recall", "accuracy")) else "MISMATCH",
             "paper/final_test_prf/seed_mean.csv"),
            ("reconciliation", f"matched the existing final-test reports exactly for all {rec[rec.reference_source.str.contains('final_test')].groupby(['model', 'seed']).ngroups} final-evaluation runs, and for the {rec[rec.reference_source.str.contains('metrics_test')].groupby(['model', 'seed']).ngroups} runs that had saved development test metrics" if rec.match.all() else "MISMATCH",
             "paper/final_test_prf/reconciliation.csv")]
    return out


def table_checks(draft: str, tables_md: Path) -> list[tuple[str, bool]]:
    """Every table body of the generated TABLES.md that the manuscript uses must appear in it verbatim."""
    import re
    t = tables_md.read_text()
    res = []
    for m in re.finditer(r"^## (T\d+b?)\. .*?\n\n(\|.*?)(?=\n\n)", t, re.S | re.M):
        res.append((m.group(1), m.group(2) in draft))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", help="also check the manuscript claims and that every table of this TABLES.md "
                                     "appears verbatim in the draft (full manuscript only)")
    a = ap.parse_args()
    text = (PROJECT_ROOT / a.draft).read_text()
    flat = " ".join(text.split()).replace("\u2212", "-")          # typographic minus == hyphen-minus
    rows, bad = [], 0
    for tid, ok in (table_checks(text, PROJECT_ROOT / a.tables) if a.tables else []):
        bad += not ok
        rows.append(f"| {'PASS' if ok else 'FAIL'} | table {tid} | table body identical to {a.tables} | {a.tables} |")
    for desc, expected, src in claims() + (manuscript_claims() if a.tables else []):
        ok = "MISMATCH" not in expected and " ".join(expected.split()).replace("\u2212", "-") in flat
        bad += not ok
        rows.append(f"| {'PASS' if ok else 'FAIL'} | {desc} | {' '.join(expected.split())} | {src} |")
    report = ["# Claims verification", "", f"Draft: `{a.draft}`. {len(rows)} claims, {bad} failed.", "",
              "| result | claim | text expected in the draft (recomputed from source) | source (under reports/) |", "|---|---|---|---|"] + rows
    (PROJECT_ROOT / a.out).write_text("\n".join(report) + "\n")
    print("\n".join(report[2:3] + [r for r in rows if r.startswith("| FAIL")]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
