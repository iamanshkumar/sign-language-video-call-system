"""E1 (OFFLINE): temporal-augmentation runs vs the BiLSTM reference, by the predefined rule.

  PYTHONPATH=src .venv/bin/python scripts/e1_01_compare.py --out reports/e1/comparison

Reference: hands_pose_w30_bilstm_e100[_s1.._s4]; E1: hands_pose_w30_bilstm_e100_augtemp[_s1.._s4].
Validation only (no test inference; E1 runs were trained with --no-test-eval). Checks that each E1
config.json equals its reference apart from the augmentation / test-evaluation entries. Reports:
- decision rule (docs/evaluation_protocol.md §3): mean validation macro-F1 gain >= 0.02, better in >= 4/5
  paired seeds, no class with validation F1 = 0 (per-class F1 averaged over the 5 seeds);
- paired class-stratified bootstrap interval of the mean gain; train accuracy and train - val gap;
- per-class validation F1 (mean over seeds) reference vs E1;
- robustness drops on validation (e0_03 perturbations) for the E1 runs;
- paired results on the frozen webcam development replay set (reports/e0/webcam_replay_set).
--out must be new or empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import EXP, PERTURBATIONS, SEEDS, boot_indices, ci, load_run, logits, macro_f1, run_dirs  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402

REPLAY = PROJECT_ROOT / "reports" / "e0" / "webcam_replay_set"
ALLOWED_CONFIG_DIFF = {"train_augmentation", "evaluate_test"}


def e1_dirs() -> dict[int, Path]:
    return {s: EXP / ("hands_pose_w30_bilstm_e100_augtemp" if s == 42 else f"hands_pose_w30_bilstm_e100_augtemp_s{s}")
            for s in SEEDS}


def sign_test(k: int, n: int) -> float:
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def per_class_f1(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    f = np.zeros(n_classes)
    for c in range(n_classes):
        tp, fp, fn = np.sum((y == c) & (p == c)), np.sum((y != c) & (p == c)), np.sum((y == c) & (p != c))
        f[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    ref, e1 = run_dirs("bilstm"), e1_dirs()
    checks, seeds_rows = [], []
    preds = {}
    for s in SEEDS:
        cr, ce = (json.loads((d[s] / "config.json").read_text()) for d in (ref, e1))
        diff = {k for k in set(cr["step4"]) | set(ce["step4"]) if cr["step4"].get(k) != ce["step4"].get(k)}
        me = json.loads((e1[s] / "metadata.json").read_text())
        checks.append(dict(seed=s, spec_equal=cr["spec"] == ce["spec"], config_diff_only_allowed=diff <= ALLOWED_CONFIG_DIFF,
                           config_diff=sorted(diff), test_evaluated=me.get("test_evaluated"),
                           no_test_files=not any(p.name.endswith(("_test.csv", "_test.json", "_test.npz")) for p in e1[s].iterdir()),
                           augmentation=me.get("train_augmentation", {}).get("type"),
                           augmentation_counts=me.get("train_augmentation_counts")))
        for name, d in (("ref", ref[s]), ("e1", e1[s])):
            pv = pd.read_csv(d / "predictions_val.csv").sort_values("sample_id", ignore_index=True)
            preds[(name, s)] = pv
            mv, mt = (json.loads((d / f"metrics_{x}.json").read_text()) for x in ("val", "train"))
            seeds_rows.append(dict(model=name, seed=s, val_macro_f1=mv["macro_f1"], val_accuracy=mv["accuracy"],
                                   train_accuracy=mt["accuracy"], train_minus_val_acc=mt["accuracy"] - mv["accuracy"],
                                   best_epoch=json.loads((d / "metadata.json").read_text()).get("best_epoch"),
                                   epochs=json.loads((d / "metadata.json").read_text()).get("epochs")))
    S = pd.DataFrame(seeds_rows)
    y = preds[("ref", 42)].y_true.to_numpy()
    assert all(p.sample_id.equals(preds[("ref", 42)].sample_id) for p in preds.values())
    bidx = boot_indices(len(y), strata=y)
    gains, bgain = [], []
    for s in SEEDS:
        pr, pe = preds[("ref", s)].y_pred.to_numpy(), preds[("e1", s)].y_pred.to_numpy()
        g = macro_f1(y, pe) - macro_f1(y, pr)
        bg = np.array([macro_f1(y[i], pe[i]) - macro_f1(y[i], pr[i]) for i in bidx])
        bgain.append(bg)
        gains.append(dict(seed=s, val_macro_f1_gain=g, ci_lo=ci(bg)[0], ci_hi=ci(bg)[1],
                          e1_only_correct=int(((pe == y) & (pr != y)).sum()), ref_only_correct=int(((pr == y) & (pe != y)).sum())))
    G = pd.DataFrame(gains)
    mean_gain, mb = G.val_macro_f1_gain.mean(), np.mean(bgain, axis=0)
    pc_ref = np.mean([per_class_f1(y, preds[("ref", s)].y_pred.to_numpy(), 100) for s in SEEDS], axis=0)
    pc_e1 = np.mean([per_class_f1(y, preds[("e1", s)].y_pred.to_numpy(), 100) for s in SEEDS], axis=0)
    words = (preds[("ref", 42)].drop_duplicates("y_true").set_index("y_true").word.sort_index())
    PC = pd.DataFrame({"word": words.reindex(range(100)).to_numpy(), "ref_f1": pc_ref, "e1_f1": pc_e1})
    PC["diff"] = PC.e1_f1 - PC.ref_f1
    better = int((G.val_macro_f1_gain > 0).sum())
    rule = {"mean_gain": float(mean_gain), "mean_gain_ci": ci(mb), "criterion_gain_ge_0.02": bool(mean_gain >= 0.02),
            "seeds_better": better, "criterion_ge_4_of_5": bool(better >= 4),
            "classes_with_zero_f1_e1": PC.loc[PC.e1_f1 == 0, "word"].tolist(), "criterion_no_zero_class": bool((PC.e1_f1 > 0).all()),
            "sign_test_p": sign_test(better, 5)}
    rule["all_criteria_met"] = bool(rule["criterion_gain_ge_0.02"] and rule["criterion_ge_4_of_5"] and rule["criterion_no_zero_class"])

    # robustness (validation) for E1 runs, same perturbations as E0
    import e0_03_robustness as rob  # noqa: E402
    rows, X = rob.val_inputs()
    assert rows.sample_id.equals(preds[("ref", 42)].sample_id)
    RB = []
    for s, d in e1.items():
        model, _ = load_run(d)
        pc = logits(model, X["clean"]).argmax(1)
        if not (pc == preds[("e1", s)].y_pred.to_numpy()).all():
            raise RuntimeError(f"E1 seed {s}: clean validation inputs do not reproduce the saved predictions")
        for p in PERTURBATIONS:
            RB.append(dict(seed=s, perturbation=p, d_accuracy=(logits(model, X[p]).argmax(1) == y).mean() - (pc == y).mean()))
    RB = pd.DataFrame(RB).groupby("perturbation", sort=False).d_accuracy.agg(["mean", "min", "max"]).reset_index()
    e0r = pd.read_csv(PROJECT_ROOT / "reports" / "e0" / "robustness" / "by_model.csv")
    RB = RB.merge(e0r[e0r.model == "bilstm"][["perturbation", "d_accuracy_mean"]].rename(columns={"d_accuracy_mean": "ref_mean"}),
                  on="perturbation")

    # webcam development replay set: paired with the reference seed
    man = pd.read_csv(REPLAY / "manifest.csv")
    Xw = np.load(REPLAY / "features.npz")["X"]
    classes = json.loads((PROJECT_ROOT / "models/candidates/hands_pose_w30_bilstm_e100_s3/class_mapping.json").read_text())["index_to_class"]
    core = (man.core & man.excluded.isna()).to_numpy()
    WR = []
    for s, d in e1.items():
        model, _ = load_run(d)
        lg = logits(model, Xw)
        top = np.argsort(-lg, axis=1)[:, :5]
        pe = np.array([classes[str(top[r, 0])] if r >= 0 else None for r in man.feature_row], dtype=object)
        t5 = np.array([(r >= 0) and (e in [classes[str(k)] for k in top[r]]) for r, e in zip(man.feature_row, man.expected)])
        ce = (pe == man.expected.to_numpy())
        rp = pd.read_csv(REPLAY / f"predictions_bilstm_s{s}.csv")
        cr = rp.correct.to_numpy().astype(bool)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"session": man.session, "attempt": man.attempt, "expected": man.expected, "core": core,
                      "predicted": pe, "correct": ce, "top5": t5}).to_csv(out / f"webcam_predictions_e1_s{s}.csv", index=False)
        b, c = int((ce & ~cr)[core].sum()), int((cr & ~ce)[core].sum())
        WR.append(dict(seed=s, core_attempts=int(core.sum()), ref_top1=cr[core].mean(), e1_top1=ce[core].mean(),
                       ref_top5=rp.top5.to_numpy().astype(bool)[core].mean(), e1_top5=t5[core].mean(),
                       e1_only=b, ref_only=c, exact_mcnemar_p=sign_test(b, b + c)))
    WR = pd.DataFrame(WR)

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(out / "config_checks.csv", index=False)
    S.to_csv(out / "per_seed.csv", index=False)
    G.to_csv(out / "val_gain_paired.csv", index=False)
    PC.to_csv(out / "per_class_val_f1.csv", index=False)
    RB.to_csv(out / "robustness_e1_vs_ref.csv", index=False)
    WR.to_csv(out / "webcam_replay_paired.csv", index=False)
    summary = {"decision_rule": rule,
               "configs_ok": bool(all(c["spec_equal"] and c["config_diff_only_allowed"] and c["test_evaluated"] is False
                                      and c["no_test_files"] for c in checks)),
               "overfitting": S.groupby("model")[["train_accuracy", "val_accuracy", "train_minus_val_acc"]].mean().to_dict("index"),
               "per_class": {"improved": int((PC["diff"] > 0).sum()), "worse": int((PC["diff"] < 0).sum()),
                             "unchanged": int((PC["diff"] == 0).sum())},
               "webcam_core_top1_mean": {"ref": float(WR.ref_top1.mean()), "e1": float(WR.e1_top1.mean())},
               "note": "validation and webcam development set only; test not evaluated"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=1, default=float))
    print(S.round(4).to_string(index=False)); print(G.round(4).to_string(index=False))
    print(RB.round(4).to_string(index=False)); print(WR.round(3).to_string(index=False))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
