"""Step 5.2: seed study summary for the dev100 hands_pose w30 LSTM vs BiLSTM runs (no training).

Reads experiments/dev100_hp/hands_pose_w30_{lstm,bilstm}_e100[_s<seed>]/ (seed 42 = untagged
"_e100" run) and writes reports/step5/seed_study.{md,json}.

Protocol (recorded in the report): every run selects its checkpoint on validation macro-F1 only;
the test split is evaluated once per run, after reloading the selected checkpoint, and is never
used to choose a run, a seed or an architecture. Statistics are descriptive first; tests are
reported with their assumptions and limits (5 seeds per model).
"""
from __future__ import annotations

import json
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from asl.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "experiments" / "dev100_hp"
METRICS = [("val", "macro_f1"), ("val", "accuracy"), ("test", "accuracy"), ("test", "macro_f1"),
           ("train", "accuracy")]


def run_dir(model: str, seed: int):
    return EXP / (f"hands_pose_w30_{model}_e100" if seed == 42 else f"hands_pose_w30_{model}_e100_s{seed}")


def load(model: str, seed: int) -> dict | None:
    d = run_dir(model, seed)
    if not (d / "metadata.json").exists():
        return None
    m = json.loads((d / "metadata.json").read_text())
    ev = {s: json.loads((d / f"metrics_{s}.json").read_text()) for s in ("train", "val", "test")}
    row = {"model": model, "seed": m["seed"], "experiment_id": m["experiment_id"], "best_epoch": m["best_epoch"],
           "epochs": m["epochs"], "train_seconds": m["train_seconds"],
           "reload_ok": m.get("checkpoint_reload_reproduces_val_f1")}
    for s, k in METRICS:
        row[f"{s}_{k}"] = ev[s][k]
    row["train_minus_val_acc"] = row["train_accuracy"] - row["val_accuracy"]
    row["train_minus_test_acc"] = row["train_accuracy"] - row["test_accuracy"]
    row["config"] = json.loads((d / "config.json").read_text())
    row["preds_test"] = pd.read_csv(d / "predictions_test.csv").set_index("sample_id")["correct"]
    return row


def mcnemar_exact(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    b = b.reindex(a.index)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(n01, n10) + 1)) / 2**n)
    return n01, n10, p


def main() -> None:
    seeds = [42, 1, 2, 3, 4]
    runs = [r for m in ("lstm", "bilstm") for s in seeds if (r := load(m, s)) is not None]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("config", "preds_test")} for r in runs])
    # comparability: every run's config must equal the seed-42 run of its model except the seed
    issues = []
    for m in ("lstm", "bilstm"):
        base = next((r for r in runs if r["model"] == m and r["seed"] == 42), None)
        for r in runs:
            if r["model"] != m or base is None:
                continue
            a, b = dict(base["config"]["step4"]), dict(r["config"]["step4"])
            a.pop("seed"), b.pop("seed")
            if a != b or base["config"]["spec"] != r["config"]["spec"]:
                issues.append(f"{r['experiment_id']}: config differs from seed 42 beyond the seed")
    agg = {}
    for m in ("lstm", "bilstm"):
        sub = df[df.model == m]
        agg[m] = {"runs": int(len(sub)), "seeds": sub["seed"].tolist()}
        for c in [f"{s}_{k}" for s, k in METRICS] + ["train_minus_val_acc", "train_minus_test_acc", "train_seconds",
                                                    "best_epoch"]:
            v = sub[c].astype(float)
            agg[m][c] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                         "min": float(v.min()), "max": float(v.max())}
    # paired by seed
    paired, mcn = [], []
    for s in seeds:
        a = next((r for r in runs if r["model"] == "lstm" and r["seed"] == s), None)
        b = next((r for r in runs if r["model"] == "bilstm" and r["seed"] == s), None)
        if a and b:
            paired.append({"seed": s, **{f"diff_{c}": b[c] - a[c] for c in ("val_macro_f1", "test_accuracy", "test_macro_f1")}})
            n01, n10, p = mcnemar_exact(a["preds_test"], b["preds_test"])
            mcn.append({"seed": s, "lstm_only_correct": n01, "bilstm_only_correct": n10, "exact_mcnemar_p": p})
    pdf = pd.DataFrame(paired)
    tests = {}
    if len(pdf) >= 2:
        for c in ("diff_val_macro_f1", "diff_test_accuracy", "diff_test_macro_f1"):
            d = pdf[c].to_numpy()
            pos = int((d > 0).sum())
            n = len(d)
            sign_p = min(1.0, 2 * sum(comb(n, k) for k in range(min(pos, n - pos) + 1)) / 2**n)
            t = stats.ttest_rel(pdf[c].to_numpy() + 0, np.zeros(n)) if n > 1 else None
            w = stats.ttest_ind(df[df.model == "bilstm"][c.replace("diff_", "")],
                                df[df.model == "lstm"][c.replace("diff_", "")], equal_var=False)
            tests[c] = {"n_pairs": n, "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
                        "bilstm_better_in": pos, "exact_sign_test_p_two_sided": sign_p,
                        "min_achievable_sign_p": 2 / 2**n,
                        "paired_t_p": float(t.pvalue) if t is not None else None,
                        "welch_t_p": float(w.pvalue)}
    out = {"protocol": "checkpoint per run selected on validation macro-F1; test evaluated once per run after "
                       "reloading; test never used for any choice. Seeds change weight init, dropout masks and "
                       "training shuffle order; data, split, features, window, optimizer, schedule, early "
                       "stopping and epoch budget are fixed.",
           "comparability_issues": issues, "runs": df.to_dict("records"), "aggregate": agg,
           "paired_by_seed": paired, "mcnemar_test_per_seed": mcn, "tests": tests}
    rep = PROJECT_ROOT / "reports" / "step5"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "seed_study.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    def ms(m, c, pct=True):
        a = agg[m][c]
        f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
        return f"{f(a['mean'])} ± {f(a['std'])} ({f(a['min'])}–{f(a['max'])})"
    L = ["# Seed study - dev100, hands_pose, 30 frames: LSTM vs BiLSTM\n",
         "Generated by `scripts/s5_02_seed_study.py` from the experiment directories. " + out["protocol"] + "\n",
         f"Comparability issues: {issues or 'none'}\n",
         "| model | seed | best / run epochs | val macro-F1 | val acc | test acc | test macro-F1 | train acc | train − val acc | train s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.sort_values(["model", "seed"]).itertuples():
        L.append(f"| {r.model} | {r.seed} | {r.best_epoch} / {r.epochs} | {r.val_macro_f1:.3f} | {100*r.val_accuracy:.1f} % | "
                 f"{100*r.test_accuracy:.1f} % | {r.test_macro_f1:.3f} | {100*r.train_accuracy:.1f} % | "
                 f"{100*r.train_minus_val_acc:.1f} pts | {r.train_seconds:.0f} |")
    L += ["", "Mean ± sample std (min–max) over seeds:\n", "| metric | LSTM | BiLSTM |", "|---|---|---|"]
    for c, pct in (("val_macro_f1", False), ("val_accuracy", True), ("test_accuracy", True), ("test_macro_f1", False),
                   ("train_accuracy", True), ("train_minus_val_acc", True)):
        L.append(f"| {c} | {ms('lstm', c, pct)} | {ms('bilstm', c, pct)} |")
    L += ["", "Paired by seed (BiLSTM − LSTM):\n", "| seed | Δ val macro-F1 | Δ test acc | Δ test macro-F1 | test clips right only LSTM / only BiLSTM | exact McNemar p |",
          "|---|---|---|---|---|---|"]
    for p, m_ in zip(paired, mcn):
        L.append(f"| {p['seed']} | {p['diff_val_macro_f1']:+.3f} | {100*p['diff_test_accuracy']:+.1f} pts | "
                 f"{p['diff_test_macro_f1']:+.3f} | {m_['lstm_only_correct']} / {m_['bilstm_only_correct']} | {m_['exact_mcnemar_p']:.3g} |")
    L += ["", "Tests on the per-seed differences (n = number of seed pairs):\n", "```json", json.dumps(tests, indent=1), "```",
          "", "Limits: with 5 seed pairs the exact two-sided sign test cannot go below p = 0.0625, so it cannot show "
          "significance at 0.05 even if BiLSTM wins every pair; t-tests assume roughly normal per-seed differences, "
          "which 5 values cannot verify. Per-seed McNemar tests compare two fixed models on the same 706 test clips "
          "(clip-level variation only, not seed variation). All runs share one frozen split that is not "
          "signer-independent."]
    (rep / "seed_study.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
