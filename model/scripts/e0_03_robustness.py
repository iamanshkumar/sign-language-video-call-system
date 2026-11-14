"""E0.3 (OFFLINE): robustness diagnostics of the 10 reference runs on the VALIDATION split.

  PYTHONPATH=src .venv/bin/python scripts/e0_03_robustness.py --out reports/e0/robustness

Applies each predeclared perturbation (docs/evaluation_protocol.md §6; e0_common.PERTURBATIONS) to the
stored arrays of every validation clip inside its active interval, then the unchanged resampling and
feature assembly, and evaluates the 5 LSTM and 5 BiLSTM runs. Clean inputs must reproduce each run's
saved validation predictions exactly. Reports accuracy / macro-F1 and the drop versus clean with a paired,
class-stratified bootstrap 95 % interval. Validation only: the test split is not touched. Diagnostic only:
used to decide which training experiment comes first, not a reported result. --out must be new or empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import PERTURBATIONS, boot_indices, ci, features, load_run, logits, macro_f1, perturb, run_dirs  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402

INDEX = PROJECT_ROOT / "data" / "sequences" / "dev100" / "index.parquet"
ARRAYS = PROJECT_ROOT / "data" / "landmarks" / "dev100_hp" / "arrays"


def val_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    idx = pd.read_parquet(INDEX)
    rows = idx[idx.split == "val"].sort_values("sample_id", ignore_index=True)   # SequenceDataset order
    X = {p: [] for p in PERTURBATIONS}
    for k, r in enumerate(rows.itertuples()):
        with np.load(ARRAYS / r.array_file) as z:
            arrays = {key: z[key] for key in ("left_hand", "right_hand", "pose", "flags", "pose_vis")}
        for p in PERTURBATIONS:
            X[p].append(features(perturb(arrays, int(r.active_start_frame), int(r.active_end_frame), p, seed=k)))
    return rows, {p: np.stack(v) for p, v in X.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    rows, X = val_inputs()
    y = rows.label.to_numpy()
    bidx = boot_indices(len(y), strata=y)
    res, checks, boots = [], [], {}
    for m in ("lstm", "bilstm"):
        for s, d in run_dirs(m).items():
            model, meta = load_run(d)
            saved = pd.read_csv(d / "predictions_val.csv").sort_values("sample_id", ignore_index=True)
            pred = {p: logits(model, X[p]).argmax(axis=1) for p in PERTURBATIONS}
            checks.append(dict(model=m, seed=s, clean_reproduces_saved=bool(
                saved.sample_id.equals(rows.sample_id) and (saved.y_pred.to_numpy() == pred["clean"]).all())))
            c_ok = pred["clean"] == y
            c_bf1 = np.array([macro_f1(y[i], pred["clean"][i]) for i in bidx])
            for p in PERTURBATIONS:
                ok = pred[p] == y
                bf1 = np.array([macro_f1(y[i], pred[p][i]) for i in bidx])
                d_acc = ok[bidx].mean(axis=1) - c_ok[bidx].mean(axis=1)
                d_f1 = bf1 - c_bf1
                boots[(m, s, p)] = (d_acc, d_f1)
                res.append(dict(model=m, seed=s, perturbation=p, accuracy=ok.mean(), macro_f1=macro_f1(y, pred[p]),
                                d_accuracy=ok.mean() - c_ok.mean(), d_acc_lo=ci(d_acc)[0], d_acc_hi=ci(d_acc)[1],
                                d_macro_f1=macro_f1(y, pred[p]) - macro_f1(y, pred["clean"]), d_f1_lo=ci(d_f1)[0], d_f1_hi=ci(d_f1)[1],
                                changed_predictions=float((pred[p] != pred["clean"]).mean())))
    R = pd.DataFrame(res)
    agg = []
    for (m, p), g in R.groupby(["model", "perturbation"], sort=False):
        seeds = sorted(g.seed)
        mb = np.mean([boots[(m, s, p)][0] for s in seeds], axis=0)
        agg.append(dict(model=m, perturbation=p, accuracy_mean=g.accuracy.mean(), d_accuracy_mean=g.d_accuracy.mean(),
                        d_acc_lo=ci(mb)[0], d_acc_hi=ci(mb)[1], d_accuracy_min=g.d_accuracy.min(), d_accuracy_max=g.d_accuracy.max(),
                        seeds_drop_ci_excludes_0=int((g.d_acc_hi < 0).sum()), d_macro_f1_mean=g.d_macro_f1.mean(),
                        changed_predictions_mean=g.changed_predictions.mean()))
    A = pd.DataFrame(agg)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "per_run.csv", index=False)
    A.to_csv(out / "by_model.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "reproduction_checks.csv", index=False)
    summary = {"clean_reproduces_saved_validation_predictions": bool(all(c["clean_reproduces_saved"] for c in checks)),
               "validation_clips": int(len(y)), "perturbations": PERTURBATIONS,
               "bootstrap": "2000 resamples, class-stratified, seed 0, paired with clean",
               "note": "validation only; diagnostic for choosing the first training experiment"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    pd.set_option("display.width", 220)
    print(A.round(4).to_string(index=False))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
