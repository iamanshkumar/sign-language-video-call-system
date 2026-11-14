"""E3 (OFFLINE, ONE-TIME): final test-set evaluation of the development ladder baseline -> E1 -> E2.

  PYTHONPATH=src .venv/bin/python scripts/e3_01_final_test.py --out reports/paper/final_test \
      --confirm-development-closed

docs/evaluation_protocol.md §5: test is evaluated once, for the final candidates, after every development
choice is fixed. Models (5 seeds each, 42, 1, 2, 3, 4):
- baseline BiLSTM (hands_pose, 261 features): its SAVED test predictions are reused (it was test-evaluated
  during development; no new inference);
- E1 temporal augmentation (261 features) and E2 E1 + upper-body pose (221 features): test inference here,
  once, with the unchanged training preprocessing (columns of pose landmarks 23-32 removed for E2).
Safeguards: refuses unless --confirm-development-closed is given; refuses if --out exists (one run only);
refuses if any E1/E2 run directory already holds test outputs; writes nothing into run directories.
Before touching test, each E1/E2 model must reproduce its saved validation predictions.
Reports per run and seed-mean accuracy / macro-F1 (class-stratified bootstrap 95 % intervals, 2,000
resamples, seed 0), paired differences E1 - baseline, E2 - E1, E2 - baseline, and per-class test F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import EXP, GROUPS, SEEDS, boot_indices, ci, features, load_run, logits, macro_f1, perturb, run_dirs  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402
from asl.landmarks.layout import pose_landmark_columns  # noqa: E402

INDEX = PROJECT_ROOT / "data" / "sequences" / "dev100" / "index.parquet"
ARRAYS = PROJECT_ROOT / "data" / "landmarks" / "dev100_hp" / "arrays"
TEST_FILES = ("predictions_test.csv", "metrics_test.json", "confusion_test.npz", "per_class_test.csv")


def ladder() -> dict[str, dict[int, Path]]:
    def d(base):
        return {s: EXP / (base if s == 42 else f"{base}_s{s}") for s in SEEDS}
    return {"baseline": run_dirs("bilstm"), "e1": d("hands_pose_w30_bilstm_e100_augtemp"),
            "e2": d("hands_pose_w30_bilstm_e100_augtemp_upperpose")}


def split_inputs(index: pd.DataFrame, arrays_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows = index[index.split == split].sort_values("sample_id", ignore_index=True)   # SequenceDataset order
    X = []
    for r in rows.itertuples():
        with np.load(arrays_dir / r.array_file) as z:
            arrays = {k: z[k] for k in ("left_hand", "right_hand", "pose", "flags", "pose_vis")}
        X.append(features(perturb(arrays, int(r.active_start_frame), int(r.active_end_frame), "clean")))
    return rows, np.stack(X)


def predict_runs(runs: dict[int, Path], X: np.ndarray, drop: list[int]) -> dict[int, np.ndarray]:
    Xi = np.ascontiguousarray(np.delete(X, drop, axis=-1)) if drop else X
    out = {}
    for s, d in runs.items():
        model, meta = load_run(d)
        if meta["feature_dim"] != Xi.shape[-1] or list(meta.get("dropped_feature_columns") or []) != list(drop):
            raise ValueError(f"{d}: feature set does not match ({meta['feature_dim']} vs {Xi.shape[-1]})")
        out[s] = logits(model, Xi).argmax(1)
    return out


def check_no_test_outputs(runs: dict[str, dict[int, Path]], names=("e1", "e2")) -> None:
    for n in names:
        for d in runs[n].values():
            found = [f for f in TEST_FILES if (d / f).exists()]
            if found or json.loads((d / "metadata.json").read_text()).get("test_evaluated"):
                raise SystemExit(f"{d} already has test outputs {found}: refusing (test is evaluated once)")


def per_class_f1(y: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    f = np.zeros(n)
    for c in range(n):
        tp, fp, fn = np.sum((y == c) & (p == c)), np.sum((y != c) & (p == c)), np.sum((y == c) & (p != c))
        f[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f


def analyse(y: np.ndarray, preds: dict[str, dict[int, np.ndarray]], words: list[str]) -> dict:
    bidx = boot_indices(len(y), strata=y)
    per_run, means, boots = [], [], {}
    for name, ps in preds.items():
        for s, p in ps.items():
            ba = (y[bidx] == p[bidx]).mean(axis=1)
            bf = np.array([macro_f1(y[i], p[i]) for i in bidx])
            boots[(name, s)] = (ba, bf)
            per_run.append(dict(model=name, seed=s, accuracy=float((y == p).mean()), acc_lo=ci(ba)[0], acc_hi=ci(ba)[1],
                                macro_f1=macro_f1(y, p), f1_lo=ci(bf)[0], f1_hi=ci(bf)[1]))
        R = [r for r in per_run if r["model"] == name]
        ma, mf = np.mean([boots[(name, s)][0] for s in ps], 0), np.mean([boots[(name, s)][1] for s in ps], 0)
        means.append(dict(model=name, seeds=len(ps), accuracy_mean=np.mean([r["accuracy"] for r in R]),
                          accuracy_sd=np.std([r["accuracy"] for r in R], ddof=1), acc_lo=ci(ma)[0], acc_hi=ci(ma)[1],
                          macro_f1_mean=np.mean([r["macro_f1"] for r in R]), macro_f1_sd=np.std([r["macro_f1"] for r in R], ddof=1),
                          f1_lo=ci(mf)[0], f1_hi=ci(mf)[1]))
    diffs = []
    for a, b in (("e1", "baseline"), ("e2", "e1"), ("e2", "baseline")):
        if a not in preds or b not in preds:
            continue
        seeds = sorted(set(preds[a]) & set(preds[b]))
        da = np.mean([boots[(a, s)][0] - boots[(b, s)][0] for s in seeds], 0)
        df = np.mean([boots[(a, s)][1] - boots[(b, s)][1] for s in seeds], 0)
        pa = np.mean([(y == preds[a][s]).mean() - (y == preds[b][s]).mean() for s in seeds])
        pf = np.mean([macro_f1(y, preds[a][s]) - macro_f1(y, preds[b][s]) for s in seeds])
        better = sum(macro_f1(y, preds[a][s]) > macro_f1(y, preds[b][s]) for s in seeds)
        diffs.append(dict(comparison=f"{a} - {b}", seeds=len(seeds), d_accuracy=pa, d_acc_lo=ci(da)[0], d_acc_hi=ci(da)[1],
                          d_macro_f1=pf, d_f1_lo=ci(df)[0], d_f1_hi=ci(df)[1], seeds_better_macro_f1=int(better)))
    pc = pd.DataFrame({"word": words, **{f"{n}_f1": np.mean([per_class_f1(y, p, len(words)) for p in ps.values()], 0)
                                         for n, ps in preds.items()}})
    return {"per_run": pd.DataFrame(per_run), "seed_mean": pd.DataFrame(means), "paired": pd.DataFrame(diffs), "per_class": pc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm-development-closed", action="store_true")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    if not args.confirm_development_closed:
        raise SystemExit("refusing: pass --confirm-development-closed (test is evaluated once, after development)")
    if out.exists():
        raise SystemExit(f"{out} exists: the final test evaluation has already been run or started")
    runs = ladder()
    check_no_test_outputs(runs)
    index = pd.read_parquet(INDEX)
    drop = {"baseline": [], "e1": [], "e2": pose_landmark_columns(GROUPS, range(23, 33))}
    # 1) validation sanity: loaded models reproduce their saved validation predictions (no test yet)
    vrows, Xv = split_inputs(index, ARRAYS, "val")
    for name in ("e1", "e2"):
        pv = predict_runs(runs[name], Xv, drop[name])
        for s, p in pv.items():
            saved = pd.read_csv(runs[name][s] / "predictions_val.csv").sort_values("sample_id", ignore_index=True)
            if not (saved.sample_id.equals(vrows.sample_id) and (saved.y_pred.to_numpy() == p).all()):
                raise SystemExit(f"{name} seed {s}: validation predictions not reproduced; test not touched")
    # 2) test, once
    out.mkdir(parents=True)
    trows, Xt = split_inputs(index, ARRAYS, "test")
    y = trows.label.to_numpy()
    preds = {"baseline": {}}
    for s, d in runs["baseline"].items():
        saved = pd.read_csv(d / "predictions_test.csv").sort_values("sample_id", ignore_index=True)
        if not saved.sample_id.equals(trows.sample_id):
            raise SystemExit(f"{d}: saved test predictions do not cover the test split")
        preds["baseline"][s] = saved.y_pred.to_numpy()
    for name in ("e1", "e2"):
        preds[name] = predict_runs(runs[name], Xt, drop[name])
    words = trows.drop_duplicates("label").set_index("label").word.sort_index().reindex(range(100)).tolist()
    res = analyse(y, preds, words)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    pd.DataFrame([{"model": n, "seed": s, "sample_id": sid, "y_true": t, "y_pred": int(p)}
                  for n, ps in preds.items() for s, pp in ps.items()
                  for sid, t, p in zip(trows.sample_id, y, pp)]).to_csv(out / "predictions_test_all.csv", index=False)
    summary = {"test_clips": int(len(y)), "models": {n: [str(d) for d in runs[n].values()] for n in runs},
               "baseline_predictions": "reused from saved predictions_test.csv (test evaluated during development)",
               "e1_e2_test_inference": "this run only", "validation_reproduced": True,
               "bootstrap": "2000 resamples, class-stratified, seed 0"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(res["seed_mean"].round(4).to_string(index=False)); print(res["paired"].round(4).to_string(index=False))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
