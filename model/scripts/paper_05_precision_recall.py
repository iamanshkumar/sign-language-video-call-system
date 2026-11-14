"""Paper: precision / recall / F1 from SAVED test predictions (no inference, no model loading).

  PYTHONPATH=src .venv/bin/python scripts/paper_05_precision_recall.py

Inputs (read-only):
- reports/paper/final_test/predictions_test_all.csv: baseline, E1, E2 x 5 seeds x 706 test clips, written once
  by the one-time final evaluation (scripts/e3_01_final_test.py);
- experiments/dev100_hp/hands_pose_w30_lstm_e100[_s<k>]/predictions_test.csv: the LSTM test predictions saved
  during development (reported in the paper as development-time test results).
Unit of analysis: one training run (model x seed) on the 706 test clips; metrics are computed per run and then
summarised over the 5 seeds (mean, sample SD, min-max). Rows are never pooled across seeds or models.
Averaging (asl.models.metrics.classification_metrics, zero_division=0, labels = union of true and predicted,
= all 100 classes here): macro = unweighted mean over classes; weighted = support-weighted mean; micro P = R =
F1 = accuracy for single-label classification. Seed-mean 95 % intervals: class-stratified bootstrap over clips,
2,000 resamples, seed 0, the same resample applied to every seed (docs/evaluation_protocol.md §2).
Reconciliation: accuracy and macro-F1 must equal reports/paper/final_test/per_run.csv; for runs that have a
development metrics_test.json (baseline, LSTM) every metric must equal it. Any mismatch aborts.
Writes reports/paper/final_test_prf/{per_run.csv, seed_mean.csv, reconciliation.csv, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import SEEDS, boot_indices, ci, run_dirs  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402
from asl.models.metrics import classification_metrics  # noqa: E402

FINAL = PROJECT_ROOT / "reports" / "paper" / "final_test"
N_CLASSES, N_TEST = 100, 706
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
           "weighted_precision", "weighted_recall", "weighted_f1"]
TOL = 1e-12


def load_predictions() -> pd.DataFrame:
    d = pd.read_csv(FINAL / "predictions_test_all.csv")
    rows = [d.assign(source="predictions_test_all.csv (one-time final evaluation; baseline rows reused)")]
    for s, run in run_dirs("lstm").items():
        p = pd.read_csv(run / "predictions_test.csv")
        rows.append(pd.DataFrame({"model": "lstm", "seed": s, "sample_id": p.sample_id, "y_true": p.y_true,
                                  "y_pred": p.y_pred, "source": f"{run.name}/predictions_test.csv (development)"}))
    return pd.concat(rows, ignore_index=True)


def check_structure(d: pd.DataFrame) -> None:
    """One row per (model, seed, clip); every run covers the same 706 clips with identical true labels."""
    if d.duplicated(["model", "seed", "sample_id"]).any():
        raise SystemExit("duplicate (model, seed, sample_id) rows")
    ref = None
    for (m, s), g in d.groupby(["model", "seed"]):
        g = g.sort_values("sample_id")
        if len(g) != N_TEST:
            raise SystemExit(f"{m} seed {s}: {len(g)} rows, expected {N_TEST}")
        key = g[["sample_id", "y_true"]].to_numpy()
        if ref is None:
            ref = key
        elif not (key == ref).all():
            raise SystemExit(f"{m} seed {s}: clips or true labels differ from the other runs")
    if sorted(d.seed.unique()) != sorted(SEEDS) or d.y_true.nunique() != N_CLASSES:
        raise SystemExit("unexpected seeds or classes")


def boot_macro_pr(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Macro precision / recall for each bootstrap index set (supports per class are fixed by stratification,
    so the union of labels is always all classes)."""
    P, R = np.empty(len(idx)), np.empty(len(idx))
    for b, i in enumerate(idx):
        yb, pb = y[i], p[i]
        tp = np.bincount(yb[yb == pb], minlength=N_CLASSES)
        npred, ntrue = np.bincount(pb, minlength=N_CLASSES), np.bincount(yb, minlength=N_CLASSES)
        P[b] = np.mean(np.where(npred > 0, tp / np.maximum(npred, 1), 0.0))
        R[b] = np.mean(np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0))
    return P, R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="reports/paper/final_test_prf")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    d = load_predictions()
    check_structure(d)

    per_run, recon, boots = [], [], {}
    final_runs = pd.read_csv(FINAL / "per_run.csv").set_index(["model", "seed"])
    y_order = None
    for (m, s), g in d.groupby(["model", "seed"], sort=False):
        g = g.sort_values("sample_id")
        y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
        if y_order is None:
            y_order, idx = y, boot_indices(len(y), strata=y)
        r = classification_metrics(y, p, N_CLASSES)
        if r["classes_evaluated"] != N_CLASSES:
            raise SystemExit(f"{m} seed {s}: {r['classes_evaluated']} classes evaluated")
        per_run.append({"model": m, "seed": int(s), **{k: r[k] for k in METRICS}, "source": g.source.iloc[0]})
        boots[(m, s)] = boot_macro_pr(y, p, idx)
        refs = []
        if (m, s) in final_runs.index:
            refs.append(("reports/paper/final_test/per_run.csv", final_runs.loc[(m, s)].to_dict()))
        run = run_dirs("bilstm" if m == "baseline" else m).get(s) if m in ("baseline", "lstm") else None
        if run is not None and (run / "metrics_test.json").exists():
            refs.append((f"experiments/dev100_hp/{run.name}/metrics_test.json",
                         json.loads((run / "metrics_test.json").read_text())))
        for src, ref in refs:
            for k in METRICS:
                if k in ref:
                    diff = abs(r[k] - float(ref[k]))
                    recon.append({"model": m, "seed": int(s), "metric": k, "computed": r[k], "reference": float(ref[k]),
                                  "abs_diff": diff, "reference_source": src, "match": diff <= TOL})
    per_run, recon = pd.DataFrame(per_run), pd.DataFrame(recon)
    if not recon.match.all():
        print(recon[~recon.match].to_string(index=False))
        raise SystemExit("reconciliation failed: computed metrics differ from saved reports (see above)")

    means = []
    for m, g in per_run.groupby("model", sort=False):
        row = {"model": m, "seeds": len(g)}
        for k in METRICS:
            row.update({f"{k}_mean": g[k].mean(), f"{k}_sd": g[k].std(ddof=1), f"{k}_min": g[k].min(), f"{k}_max": g[k].max()})
        bp = np.mean([boots[(m, s)][0] for s in g.seed], 0)
        br = np.mean([boots[(m, s)][1] for s in g.seed], 0)
        row.update({"macro_precision_lo": ci(bp)[0], "macro_precision_hi": ci(bp)[1],
                    "macro_recall_lo": ci(br)[0], "macro_recall_hi": ci(br)[1]})
        means.append(row)
    means = pd.DataFrame(means)

    out.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out / "per_run.csv", index=False)
    means.to_csv(out / "seed_mean.csv", index=False)
    recon.to_csv(out / "reconciliation.csv", index=False)
    summary = {
        "inputs": sorted(set(per_run.source)),
        "unit_of_analysis": "one run (model x seed) on 706 test clips; summarised over 5 seeds (mean, sample SD)",
        "averaging": {"macro": "unweighted mean over the 100 classes (union of true and predicted labels)",
                      "weighted": "mean weighted by true-class support",
                      "micro": "equals accuracy for single-label classification (not reported separately)",
                      "zero_division": 0},
        "bootstrap": "class-stratified over clips, 2000 resamples, seed 0, same resample for every seed",
        "reconciliation": {"checks": int(len(recon)), "all_match": bool(recon.match.all()),
                           "max_abs_diff": float(recon.abs_diff.max()), "tolerance": TOL},
        "new_inference": False,
        "note": "lstm test predictions were produced during development (not part of the one-time final evaluation)",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cols = ["model"] + [f"{k}_mean" for k in METRICS[:4]]
    print(means[cols].round(4).to_string(index=False))
    print(f"reconciliation: {len(recon)} checks, max |diff| {recon.abs_diff.max():.2e}; saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
