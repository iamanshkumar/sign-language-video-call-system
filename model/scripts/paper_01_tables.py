"""Paper tables and figures, generated ONLY from saved source reports (no model is run).

  PYTHONPATH=src .venv/bin/python scripts/paper_01_tables.py --out reports/paper/tables_figures

Every table lists its source files. Validation results, previously reported test results and the
single-signer webcam replay are kept in separate tables. --out must be new or empty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from asl.config import PROJECT_ROOT

R = PROJECT_ROOT / "reports"
EXP_TABLE = R / "step4" / "results" / "experiment_results.csv"


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{100 * x:.{d}f}"


def f3(x):
    return "–" if x is None or pd.isna(x) else f"{x:.3f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build() -> list[tuple[str, str, pd.DataFrame, list[str]]]:
    T = []
    # T1 data
    idx = pd.read_parquet(PROJECT_ROOT / "data/sequences/dev100/index.parquet")
    per = idx[idx.split == "train"].groupby("word").size()
    t1 = pd.DataFrame([{"classes": idx.word.nunique(), "accepted clips": len(idx),
                        "train / val / test": " / ".join(str((idx.split == s).sum()) for s in ("train", "val", "test")),
                        "train clips per class (median, min–max)": f"{int(per.median())} ({per.min()}–{per.max()})",
                        "signer-independent split": "no (no signer IDs)"}])
    T.append(("T1", "Dataset (dev100 store)", t1, ["data/sequences/dev100/index.parquet"]))

    # T2 LSTM vs BiLSTM (validation and PREVIOUSLY REPORTED test)
    sm = pd.read_csv(R / "e0/bootstrap/seed_mean.csv")
    dd = pd.read_csv(R / "e0/bootstrap/bilstm_minus_lstm.csv")
    rows = []
    for _, r in sm.iterrows():
        rows.append({"split": r.split, "model": r.model.upper().replace("BILSTM", "BiLSTM"),
                     "accuracy % (mean ± SD)": f"{pct(r.accuracy_mean)} ± {pct(r.accuracy_sd)}",
                     "95 % CI": f"{pct(r.acc_clip_lo)}–{pct(r.acc_clip_hi)}",
                     "macro-F1 (mean ± SD)": f"{f3(r.macro_f1_mean)} ± {f3(r.macro_f1_sd)}",
                     "95 % CI ": f"{f3(r.f1_clip_lo)}–{f3(r.f1_clip_hi)}"})
    for _, r in dd[dd.seed == "mean"].iterrows():
        rows.append({"split": r.split, "model": "BiLSTM − LSTM (paired)", "accuracy % (mean ± SD)": f"{100 * r.d_accuracy:+.1f} pts",
                     "95 % CI": f"{100 * r.d_acc_lo:+.1f} to {100 * r.d_acc_hi:+.1f}", "macro-F1 (mean ± SD)": f"{r.d_macro_f1:+.3f}",
                     "95 % CI ": f"{r.d_f1_lo:+.3f} to {r.d_f1_hi:+.3f}"})
    T.append(("T2", "LSTM vs BiLSTM, hands + pose, 30 frames, 5 seeds (test values were reported during development)",
              pd.DataFrame(rows), ["reports/e0/bootstrap/seed_mean.csv", "reports/e0/bootstrap/bilstm_minus_lstm.csv"]))

    # T3 landmark representation (validation only), each comparison in its own training setting
    rows, src = [], []
    hc = R / "paper/hands_only/comparison"
    if (hc / "summary.json").exists():
        s = json.loads((hc / "summary.json").read_text())
        ps = pd.read_csv(hc / "per_seed.csv")
        for rep, feats in (("hands_pose", 261), ("hands", 128)):
            g = ps[ps.representation == rep]
            rows.append({"setting": "BiLSTM, no augmentation", "representation": {"hands_pose": "hands + full pose", "hands": "hands only"}[rep],
                         "features": feats, "val macro-F1 (mean ± SD)": f"{f3(g.val_macro_f1.mean())} ± {f3(g.val_macro_f1.std())}",
                         "paired change (95 % CI)": "reference" if rep == "hands_pose" else
                         f"{s['mean_difference_hands_minus_hands_pose']:+.3f} ({s['difference_ci'][0]:+.3f} to {s['difference_ci'][1]:+.3f})",
                         "train − val acc (pts)": f"{100 * g.train_minus_val_acc.mean():.1f}"})
        src += [str(hc.relative_to(PROJECT_ROOT)) + "/summary.json", str(hc.relative_to(PROJECT_ROOT)) + "/per_seed.csv"]
    else:
        rows.append({"setting": "BiLSTM, no augmentation", "representation": "hands only", "features": 128,
                     "val macro-F1 (mean ± SD)": "PENDING", "paired change (95 % CI)": "PENDING", "train − val acc (pts)": "PENDING"})
    e2 = json.loads((R / "e2/comparison/summary.json").read_text())
    e2s = pd.read_csv(R / "e2/comparison/per_seed.csv")
    for m, rep, feats in (("e1", "hands + full pose", 261), ("e2", "hands + upper-body pose (0–22)", 221)):
        g = e2s[e2s.model == m]
        rows.append({"setting": "BiLSTM + temporal augmentation", "representation": rep, "features": feats,
                     "val macro-F1 (mean ± SD)": f"{f3(g.val_macro_f1.mean())} ± {f3(g.val_macro_f1.std())}",
                     "paired change (95 % CI)": "reference" if m == "e1" else
                     f"{e2['rule']['mean_change']:+.3f} ({e2['rule']['mean_change_ci'][0]:+.3f} to {e2['rule']['mean_change_ci'][1]:+.3f})",
                     "train − val acc (pts)": f"{100 * g.train_minus_val_acc.mean():.1f}"})
    rows.append({"setting": "—", "representation": "hands + pose + face", "features": "–", "val macro-F1 (mean ± SD)": "not evaluated",
                 "paired change (95 % CI)": "face landmarks not extracted for dev100", "train − val acc (pts)": "–"})
    T.append(("T3", "Landmark representation, VALIDATION, 5 paired seeds", pd.DataFrame(rows),
              src + ["reports/e2/comparison/summary.json", "reports/e2/comparison/per_seed.csv", "configs/step2_landmarks.yaml"]))

    # T4 resampling length (single seed, exploratory; test reported during development)
    ex = pd.read_csv(EXP_TABLE)
    w = ex[ex.experiment_id.isin(["hands_pose_w30_lstm_e100", "hands_pose_w45_lstm_e100"])]
    t4 = pd.DataFrame([{"frames per sign": int(r.window_size), "model": "LSTM", "seed": int(r.seed),
                        "val macro-F1": f3(r.best_val_macro_f1), "test macro-F1 (reported in development)": f3(r.test_macro_f1),
                        "model latency b=1 (ms, training run)": f"{r.mean_latency_ms:.2f}"} for r in w.itertuples()])
    T.append(("T4", "Temporal resampling length (EXPLORATORY: single seed, LSTM; not a sliding-window study)", t4,
              ["reports/step4/results/experiment_results.csv"]))

    # T5 development ladder (validation; test only for the baseline, reported earlier)
    e1s = pd.read_csv(R / "e1/comparison/per_seed.csv")
    e1 = json.loads((R / "e1/comparison/summary.json").read_text())
    ft = R / "paper/final_test"
    ftm = pd.read_csv(ft / "seed_mean.csv").set_index("model") if (ft / "seed_mean.csv").exists() else None

    def test_cell(key):
        if ftm is None or key not in ftm.index:
            return "not evaluated"
        r = ftm.loc[key]
        return f"{pct(r.accuracy_mean)} % / {f3(r.macro_f1_mean)}" + (" (reported in development)" if key == "baseline" else " (one-time final evaluation)")
    rows = []
    for name, g, ch in (("Baseline BiLSTM", e1s[e1s.model == "ref"], "reference"),
                        ("E1: + temporal augmentation", e1s[e1s.model == "e1"],
                         f"{e1['decision_rule']['mean_gain']:+.3f} ({e1['decision_rule']['mean_gain_ci'][0]:+.3f} to {e1['decision_rule']['mean_gain_ci'][1]:+.3f}); 5/5 seeds; §3 met"),
                        ("E2: E1 + upper-body pose", e2s[e2s.model == "e2"],
                         f"{e2['rule']['mean_change']:+.3f} vs E1 ({e2['rule']['mean_change_ci'][0]:+.3f} to {e2['rule']['mean_change_ci'][1]:+.3f}); 4/5 seeds; §3a non-inferiority met, §3 not met")):
        rows.append({"model": name, "val macro-F1 (mean ± SD)": f"{f3(g.val_macro_f1.mean())} ± {f3(g.val_macro_f1.std())}",
                     "val accuracy %": pct(g.val_accuracy.mean()), "train − val (pts)": f"{100 * g.train_minus_val_acc.mean():.1f}",
                     "paired change (95 % CI)": ch,
                     "test acc / macro-F1": test_cell("baseline" if "Baseline" in name else ("e1" if name.startswith("E1") else "e2"))})
    T.append(("T5", "Development ladder, VALIDATION (paired changes) with the test column from the one-time final evaluation", pd.DataFrame(rows),
              ["reports/e1/comparison/per_seed.csv", "reports/e1/comparison/summary.json", "reports/e2/comparison/per_seed.csv",
               "reports/e2/comparison/summary.json", "reports/e0/bootstrap/seed_mean.csv"]))

    # T5b one-time final test evaluation (E1, E2; baseline reused from saved predictions)
    if ftm is not None:
        fp = pd.read_csv(ft / "paired.csv")
        names = {"baseline": "Baseline BiLSTM (saved predictions, reported in development)", "e1": "E1: + temporal augmentation",
                 "e2": "E2: E1 + upper-body pose (221 features)"}
        rows = [{"model": names[k], "test accuracy % (mean ± SD)": f"{pct(r.accuracy_mean)} ± {pct(r.accuracy_sd)}",
                 "95 % CI": f"{pct(r.acc_lo)}–{pct(r.acc_hi)}", "test macro-F1 (mean ± SD)": f"{f3(r.macro_f1_mean)} ± {f3(r.macro_f1_sd)}",
                 "95 % CI ": f"{f3(r.f1_lo)}–{f3(r.f1_hi)}"} for k, r in ftm.iterrows()]
        rows += [{"model": f"paired {r.comparison}", "test accuracy % (mean ± SD)": f"{100 * r.d_accuracy:+.1f} pts",
                  "95 % CI": f"{100 * r.d_acc_lo:+.1f} to {100 * r.d_acc_hi:+.1f}",
                  "test macro-F1 (mean ± SD)": f"{r.d_macro_f1:+.3f} ({int(r.seeds_better_macro_f1)}/5 seeds better)",
                  "95 % CI ": f"{r.d_f1_lo:+.3f} to {r.d_f1_hi:+.3f}"} for r in fp.itertuples()]
        T.append(("T5b", "TEST, one-time final evaluation after development closed (706 clips, 5 seeds; class-stratified bootstrap)",
                  pd.DataFrame(rows), ["reports/paper/final_test/seed_mean.csv", "reports/paper/final_test/paired.csv",
                                       "reports/paper/final_test/summary.json"]))

    # T6 latency and throughput (one machine)
    L = json.loads((R / "step6/latency_hands_pose_w30_bilstm_e100_s3.json").read_text())
    ses = {k: json.loads((R / f"step6/webcam_eval/{v}_camera_S01/summary.json").read_text())
           for k, v in (("S2", "20261019T075328Z"), ("S3", "20261019T084115Z"), ("S4", "20261020T045903Z"))}
    lm = ex[ex.experiment_id.str.match(r"hands_pose_w30_(lstm|bilstm)_e100(_s\d)?$")]
    rows = [{"component": "MediaPipe Holistic per frame", "median ms": f"{L['mediapipe_per_frame']['median_ms']:.1f}",
             "p95 ms": f"{L['mediapipe_per_frame']['p95_ms']:.1f}", "note": f"max {L['mediapipe_max_fps_single_process']:.1f} frames/s single process"},
            {"component": "BiLSTM forward, batch 1", "median ms": f"{L['model_only_forward']['median_ms']:.2f}",
             "p95 ms": f"{L['model_only_forward']['p95_ms']:.2f}", "note": "dedicated benchmark, 500 runs"},
            {"component": "LSTM forward, batch 1 (5 training runs)", "median ms": f"{lm[lm.model == 'lstm'].median_latency_ms.median():.2f}",
             "p95 ms": "–", "note": "measured inside training runs (median of runs)"},
            {"component": "BiLSTM forward, batch 1 (5 training runs)", "median ms": f"{lm[lm.model == 'bilstm'].median_latency_ms.median():.2f}",
             "p95 ms": "–", "note": "measured inside training runs (median of runs)"},
            {"component": "streaming push per frame", "median ms": f"{L['push_per_frame_no_segment']['median_ms']:.3f}",
             "p95 ms": f"{L['push_per_frame_no_segment']['p95_ms']:.3f}", "note": ""},
            {"component": "segment close: preprocessing + model", "median ms": f"{L['segment_close_preprocess_plus_model']['median_ms']:.1f}",
             "p95 ms": f"{L['segment_close_preprocess_plus_model']['p95_ms']:.1f}", "note": "20 dataset clips"}]
    for k, s in ses.items():
        i = s["input_diagnostics"]
        rows.append({"component": f"live capture {k}", "median ms": f"{i['frame_interval_ms']['median']:.1f} (frame interval)",
                     "p95 ms": f"{i['frame_interval_ms']['p95']:.1f}",
                     "note": f"{i['processed_fps']:.1f} processed fps; threaded={s.get('capture_options', {}).get('threaded', False)}"})
    T.append(("T6", f"Latency and throughput (CPU, {L['hardware']['platform']}, {L['hardware']['ram_gb']:.0f} GB, torch threads {L['hardware']['torch_threads']})",
              pd.DataFrame(rows), ["reports/step6/latency_hands_pose_w30_bilstm_e100_s3.json", "reports/step4/results/experiment_results.csv",
                                   "reports/step6/webcam_eval/<session>/summary.json"]))

    # T7 webcam replay benchmark (single signer)
    sb = pd.read_csv(R / "e0/webcam_replay_set/summary_by_model.csv")
    sb = sb[(sb.scope == "core_valid") & sb.model.str.match(r"(lstm|bilstm)_s")]
    e1w = pd.read_csv(R / "e1/comparison/webcam_replay_paired.csv")
    e2w = pd.read_csv(R / "e2/comparison/webcam_replay_paired.csv")
    rows = [{"model (5 seeds)": "LSTM", "top-1 % mean (min–max)": f"{pct(sb[sb.model.str.startswith('lstm')].top1.mean())} ({pct(sb[sb.model.str.startswith('lstm')].top1.min())}–{pct(sb[sb.model.str.startswith('lstm')].top1.max())})",
             "top-5 % mean": pct(sb[sb.model.str.startswith('lstm')].top5.mean())},
            {"model (5 seeds)": "BiLSTM (baseline)", "top-1 % mean (min–max)": f"{pct(sb[sb.model.str.startswith('bilstm')].top1.mean())} ({pct(sb[sb.model.str.startswith('bilstm')].top1.min())}–{pct(sb[sb.model.str.startswith('bilstm')].top1.max())})",
             "top-5 % mean": pct(sb[sb.model.str.startswith('bilstm')].top5.mean())},
            {"model (5 seeds)": "E1", "top-1 % mean (min–max)": f"{pct(e1w.e1_top1.mean())} ({pct(e1w.e1_top1.min())}–{pct(e1w.e1_top1.max())})", "top-5 % mean": pct(e1w.e1_top5.mean())},
            {"model (5 seeds)": "E2", "top-1 % mean (min–max)": f"{pct(e2w.e2_top1.mean())} ({pct(e2w.e2_top1.min())}–{pct(e2w.e2_top1.max())})", "top-5 % mean": pct(e2w.e2_top5.mean())}]
    T.append(("T7", "Webcam development replay set — ONE SIGNER, 92 core attempts, frozen segments (not generalization evidence)",
              pd.DataFrame(rows), ["reports/e0/webcam_replay_set/summary_by_model.csv", "reports/e1/comparison/webcam_replay_paired.csv",
                                   "reports/e2/comparison/webcam_replay_paired.csv"]))

    # T8 streaming sessions (single signer, live baseline candidate)
    rows = []
    for k, sub in (("S3", "replay_firsthand/"), ("S4", "")):
        ts = {"S3": "20261019T084115Z", "S4": "20261020T045903Z"}[k]
        s = json.loads((R / f"step6/webcam_eval/{ts}_camera_S01/{sub}summary.json").read_text())
        live = json.loads((R / f"step6/webcam_eval/{ts}_camera_S01/summary.json").read_text())
        a = pd.read_csv(R / f"step6/webcam_eval/{ts}_camera_S01/attempts.csv")
        oc = s["outcomes"]
        rows.append({"session": k, "protocol": live["notes"][:40], "attempts": s["attempts"], "correct": oc.get("correct", 0),
                     "missed": oc.get("missed", 0), "attempts with duplicates": s["attempts_with_duplicates"],
                     "answers forced-cut (live)": int((a.forced_cut == True).sum()),  # noqa: E712
                     "median delay rest-closed s (live)": f"{a[a.forced_cut == False].delay_s.median():.2f}",  # noqa: E712
                     "processed fps": f"{live['input_diagnostics']['processed_fps']:.1f}",
                     "dropped frames": f"{live['dropped_frames']}/{live['camera_frames_captured']}"})
    T.append(("T8", "Live streaming sessions — ONE SIGNER, baseline BiLSTM candidate (S3 re-scored by replay with the final scoring rules)",
              pd.DataFrame(rows), ["reports/step6/webcam_eval/20261019T084115Z_camera_S01/replay_firsthand/summary.json",
                                   "reports/step6/webcam_eval/20261020T045903Z_camera_S01/summary.json", "…/attempts.csv"]))

    # T9 post-processing grid (single signer)
    P = pd.read_csv(R / "paper/postprocessing/summary_by_setting.csv")
    t9 = pd.DataFrame([{"cooldown s": r.cooldown_s, "confidence threshold": r.confidence_threshold, "attempts": r.attempts,
                        "correct": r.correct, "top-1 %": pct(r.top1), "coverage %": pct(r.coverage),
                        "precision of answers %": pct(r.precision_of_answers), "duplicate emissions": r.duplicate_emissions,
                        "missed (no segment)": r.no_segment, "spurious outside attempts": r.spurious_emissions_outside_attempts}
                       for r in P.itertuples()])
    T.append(("T9", "Streaming post-processing replay — ONE SIGNER, 92 core attempts, baseline BiLSTM candidate", t9,
              ["reports/paper/postprocessing/summary_by_setting.csv", "reports/paper/postprocessing/summary.json"]))

    # T10 calibration (validation; test check reported during development)
    C = json.loads((R / "step6/calibration_hands_pose_w30_bilstm_e100_s3.json").read_text())
    t10 = pd.DataFrame([{"split": sp, "accuracy %": pct(C["splits"][sp]["accuracy"]), "ECE (15 bins)": f3(C["splits"][sp]["ece_15_bins"]),
                         "mean confidence": f3(C["splits"][sp]["mean_confidence"])} for sp in ("val", "test")] +
                       [{"split": f"threshold {C['suggested_threshold_from_validation']} (chosen on val, 80 % precision target) on test",
                         "accuracy %": pct(C["test_check_at_that_threshold"]["test_accuracy_of_accepted"]) + " of accepted",
                         "ECE (15 bins)": "–", "mean confidence": f"coverage {pct(C['test_check_at_that_threshold']['test_coverage'])} %"}])
    T.append(("T10", "Confidence calibration, baseline candidate (BiLSTM seed 3); test check was run during development",
              t10, ["reports/step6/calibration_hands_pose_w30_bilstm_e100_s3.json"]))

    # T11 robustness (validation)
    rb0 = pd.read_csv(R / "e0/robustness/by_model.csv")
    rb1 = pd.read_csv(R / "e1/comparison/robustness_e1_vs_ref.csv")
    rb2 = pd.read_csv(R / "e2/comparison/robustness_e2.csv")
    t11 = rb0[rb0.model == "bilstm"][["perturbation", "d_accuracy_mean"]].rename(columns={"d_accuracy_mean": "baseline"}) \
        .merge(rb1[["perturbation", "mean"]].rename(columns={"mean": "E1"}), on="perturbation") \
        .merge(rb2[["perturbation", "mean"]].rename(columns={"mean": "E2"}), on="perturbation")
    t11 = t11[t11.perturbation != "clean"]
    for c in ("baseline", "E1", "E2"):
        t11[c] = t11[c].map(lambda v: f"{100 * v:+.1f}")
    T.append(("T11", "Robustness diagnostics, VALIDATION, accuracy change vs clean (pts, mean of 5 BiLSTM seeds)", t11,
              ["reports/e0/robustness/by_model.csv", "reports/e1/comparison/robustness_e1_vs_ref.csv", "reports/e2/comparison/robustness_e2.csv"]))

    # T12 test-set use disclosure
    t12 = pd.DataFrame([
        {"use": "test metrics computed for development runs", "count": int(ex[ex.store.eq("dev100_hp")].test_accuracy.notna().sum()),
         "source": "reports/step4/results/experiment_results.csv (rows with test values)"},
        {"use": "confidence-threshold check on test (candidate s3)", "count": 1, "source": "reports/step6/calibration_hands_pose_w30_bilstm_e100_s3.json"},
        {"use": "latency/parity script: predictions on 20 test clips (equality check only)", "count": 1, "source": "reports/step6/latency_hands_pose_w30_bilstm_e100_s3.json"},
        {"use": "simulated live replay of dataset test clips", "count": 3, "source": "reports/step6/webcam_eval/*_video_replay_dataset-test-clip"},
        {"use": "one-time final test evaluation of E1 and E2 (5 seeds each), after development closed",
         "count": 1 if (R / "paper/final_test/summary.json").exists() else 0, "source": "reports/paper/final_test/"},
        {"use": "hands-only test inference", "count": 0, "source": "metadata.json test_evaluated = false"}])
    T.append(("T12", "Test-set use disclosure (no test result was used to choose a model)", t12, ["as listed"]))
    return T


def figures(out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    # F2 per-seed validation macro-F1 ladder
    e1s = pd.read_csv(R / "e1/comparison/per_seed.csv"); e2s = pd.read_csv(R / "e2/comparison/per_seed.csv")
    seeds = [42, 1, 2, 3, 4]
    series = {"Baseline": e1s[e1s.model == "ref"].set_index("seed").val_macro_f1, "E1": e1s[e1s.model == "e1"].set_index("seed").val_macro_f1,
              "E2": e2s[e2s.model == "e2"].set_index("seed").val_macro_f1}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for s in seeds:
        ax.plot(range(3), [series[k][s] for k in series], marker="o", color="0.6", lw=1)
    ax.plot(range(3), [series[k].mean() for k in series], marker="s", color="k", lw=2, label="mean of 5 seeds")
    ax.set_xticks(range(3), list(series)); ax.set_ylabel("validation macro-F1"); ax.legend(frameon=False)
    ax.set_title("Paired seeds (validation only)", fontsize=10); fig.tight_layout(); fig.savefig(out / "F2_val_macro_f1_paired_seeds.png", dpi=150)
    plt.close(fig); made.append("F2_val_macro_f1_paired_seeds.png")
    # F3 latency breakdown
    L = json.loads((R / "step6/latency_hands_pose_w30_bilstm_e100_s3.json").read_text())
    comp = {"MediaPipe / frame": L["mediapipe_per_frame"]["median_ms"], "BiLSTM / sign": L["model_only_forward"]["median_ms"],
            "segment close / sign": L["segment_close_preprocess_plus_model"]["median_ms"], "push / frame": L["push_per_frame_no_segment"]["median_ms"]}
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(list(comp), list(comp.values()), color="0.4"); ax.set_xlabel("median ms (CPU, one laptop)")
    for i, v in enumerate(comp.values()):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "F3_latency_breakdown.png", dpi=150); plt.close(fig); made.append("F3_latency_breakdown.png")
    # F4 post-processing trade-off
    P = pd.read_csv(R / "paper/postprocessing/summary_by_setting.csv")
    th = P[P.cooldown_s == 0].sort_values("confidence_threshold")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(th.coverage, th.precision_of_answers, marker="o", color="k")
    for r in th.itertuples():
        ax.annotate(f"t={r.confidence_threshold}", (r.coverage, r.precision_of_answers), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("coverage (answered attempts)"); ax.set_ylabel("precision of answers")
    ax.set_title("Confidence threshold, one signer (92 attempts)", fontsize=10); fig.tight_layout()
    fig.savefig(out / "F4_threshold_coverage_precision.png", dpi=150); plt.close(fig); made.append("F4_threshold_coverage_precision.png")
    # F5 calibration (coverage curve from validation and test)
    C = json.loads((R / "step6/calibration_hands_pose_w30_bilstm_e100_s3.json").read_text())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for sp, ls in (("val", "-"), ("test", "--")):
        cv = pd.DataFrame(C["splits"][sp]["coverage"])
        ax.plot(cv.coverage, cv.accuracy_of_accepted, ls, marker=".", label=f"{sp} (dataset)", color="k")
    ax.plot(th.coverage, th.precision_of_answers, ":", marker="o", label="webcam replay (one signer)", color="0.5")
    ax.set_xlabel("coverage"); ax.set_ylabel("accuracy of accepted"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "F5_coverage_accuracy_dataset_vs_webcam.png", dpi=150); plt.close(fig)
    made.append("F5_coverage_accuracy_dataset_vs_webcam.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    parts = ["# Paper tables (generated from saved reports by scripts/paper_01_tables.py)\n"]
    for tid, title, df, src in build():
        df.to_csv(out / f"{tid}.csv", index=False)
        parts += [f"## {tid}. {title}\n", md(df), "", "Source: " + "; ".join(f"`{s}`" for s in src), ""]
    figs = figures(out)
    parts += ["## Figures\n"] + [f"- `{f}`" for f in figs]
    (out / "TABLES.md").write_text("\n".join(parts) + "\n")
    print(f"saved {len(figs)} figures and tables to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
