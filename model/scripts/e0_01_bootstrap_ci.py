"""E0.1 (OFFLINE): confidence intervals for the 10 reference runs from their SAVED predictions.

  PYTHONPATH=src .venv/bin/python scripts/e0_01_bootstrap_ci.py --out reports/e0/bootstrap

Reads experiments/dev100_hp/hands_pose_w30_{lstm,bilstm}_e100[_s1.._s4]/predictions_{val,test}.csv and
metrics_{val,test}.json (no model is run, no new test inference). Point estimates must reproduce the
saved metrics exactly. Per run: accuracy and macro-F1 with 95 % clip bootstrap intervals (resampled within each true class); per model:
seed mean with the same clip resample applied to all seeds; BiLSTM - LSTM paired by seed and pooled.
Protocol: docs/evaluation_protocol.md §2. Output folder must be new or empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import SEEDS, boot_indices, ci, macro_f1, run_dirs  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402


def load_preds(run: Path, split: str) -> pd.DataFrame:
    d = pd.read_csv(run / f"predictions_{split}.csv").sort_values("sample_id", ignore_index=True)
    return d[["sample_id", "y_true", "y_pred"]]


def run_all(out: Path) -> dict:
    rows, means, diffs, checks = [], [], [], []
    for split in ("val", "test"):
        preds = {(m, s): load_preds(r, split) for m in ("lstm", "bilstm") for s, r in run_dirs(m).items()}
        ids = next(iter(preds.values())).sample_id
        if not all(p.sample_id.equals(ids) for p in preds.values()):
            raise ValueError(f"{split}: runs do not share the same clips")
        idx = boot_indices(len(ids), strata=next(iter(preds.values())).y_true.to_numpy())
        boot = {}
        for (m, s), p in preds.items():
            y, q = p.y_true.to_numpy(), p.y_pred.to_numpy()
            saved = json.loads((run_dirs(m)[s] / f"metrics_{split}.json").read_text())
            acc, f1 = float((y == q).mean()), macro_f1(y, q)
            checks.append(dict(split=split, model=m, seed=s, accuracy_matches=abs(acc - saved["accuracy"]) < 1e-12,
                               macro_f1_matches=abs(f1 - saved["macro_f1"]) < 1e-9))
            b_acc = (y[idx] == q[idx]).mean(axis=1)
            b_f1 = np.array([macro_f1(y[i], q[i]) for i in idx])
            boot[(m, s)] = (b_acc, b_f1)
            rows.append(dict(split=split, model=m, seed=s, n=len(y), accuracy=acc, acc_lo=ci(b_acc)[0], acc_hi=ci(b_acc)[1],
                             macro_f1=f1, f1_lo=ci(b_f1)[0], f1_hi=ci(b_f1)[1]))
        for m in ("lstm", "bilstm"):
            pt = [r for r in rows if r["split"] == split and r["model"] == m]
            mb_acc = np.mean([boot[(m, s)][0] for s in SEEDS], axis=0)
            mb_f1 = np.mean([boot[(m, s)][1] for s in SEEDS], axis=0)
            means.append(dict(split=split, model=m, seeds=len(pt),
                              accuracy_mean=np.mean([r["accuracy"] for r in pt]), accuracy_sd=np.std([r["accuracy"] for r in pt], ddof=1),
                              acc_clip_lo=ci(mb_acc)[0], acc_clip_hi=ci(mb_acc)[1],
                              macro_f1_mean=np.mean([r["macro_f1"] for r in pt]), macro_f1_sd=np.std([r["macro_f1"] for r in pt], ddof=1),
                              f1_clip_lo=ci(mb_f1)[0], f1_clip_hi=ci(mb_f1)[1]))
        for s in SEEDS + ["mean"]:
            if s == "mean":
                da = np.mean([boot[("bilstm", t)][0] - boot[("lstm", t)][0] for t in SEEDS], axis=0)
                df = np.mean([boot[("bilstm", t)][1] - boot[("lstm", t)][1] for t in SEEDS], axis=0)
                pa = np.mean([r["accuracy"] for r in rows if r["split"] == split and r["model"] == "bilstm"]) - \
                    np.mean([r["accuracy"] for r in rows if r["split"] == split and r["model"] == "lstm"])
                pf = np.mean([r["macro_f1"] for r in rows if r["split"] == split and r["model"] == "bilstm"]) - \
                    np.mean([r["macro_f1"] for r in rows if r["split"] == split and r["model"] == "lstm"])
            else:
                da, df = boot[("bilstm", s)][0] - boot[("lstm", s)][0], boot[("bilstm", s)][1] - boot[("lstm", s)][1]
                g = {r["model"]: r for r in rows if r["split"] == split and r["seed"] == s}
                pa, pf = g["bilstm"]["accuracy"] - g["lstm"]["accuracy"], g["bilstm"]["macro_f1"] - g["lstm"]["macro_f1"]
            diffs.append(dict(split=split, seed=s, d_accuracy=pa, d_acc_lo=ci(da)[0], d_acc_hi=ci(da)[1],
                              d_macro_f1=pf, d_f1_lo=ci(df)[0], d_f1_hi=ci(df)[1]))
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_run", rows), ("seed_mean", means), ("bilstm_minus_lstm", diffs), ("reproduction_checks", checks)):
        pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
    summary = {"all_point_estimates_reproduce_saved_metrics": bool(all(c["accuracy_matches"] and c["macro_f1_matches"] for c in checks)),
               "bootstrap": {"resamples": len(boot_indices(2)), "seed": 0, "unit": "clip, stratified by true class", "interval": "95% percentile"},
               "note": "computed from saved predictions; no new test inference"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    print(json.dumps(run_all(out), indent=1))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
