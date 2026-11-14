"""E2 (OFFLINE): upper-body pose ablation vs its reference E1, by the non-inferiority rule.

  PYTHONPATH=src .venv/bin/python scripts/e2_01_compare.py --out reports/e2/comparison

Reference E1: hands_pose_w30_bilstm_e100_augtemp[_s1.._s4] (temporal augmentation). E2: the same with pose
landmarks 23-32 (x, y, z, visibility; 40 columns) removed: hands_pose_w30_bilstm_e100_augtemp_upperpose[_s..].
Validation only; no test inference. Checks each E2 config.json equals its E1 config apart from
`feature_ablation`. Reports (docs/evaluation_protocol.md §3a, written before E2 was run):
- non-inferiority: mean validation macro-F1 change >= -0.01, lower bound of the paired class-stratified 95 %
  bootstrap interval >= -0.02, no class with validation F1 = 0; §3 improvement rule also evaluated;
- train accuracy and train - val gap; per-class validation F1 (mean over seeds);
- robustness drops on validation (E0 perturbations, columns removed after the perturbation);
- paired results on the frozen webcam development replay set (same 92 core attempts).
--out must be new or empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import EXP, GROUPS, PERTURBATIONS, SEEDS, boot_indices, ci, load_run, logits, macro_f1  # noqa: E402
from e1_01_compare import per_class_f1, sign_test  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402
from asl.landmarks.layout import pose_landmark_columns  # noqa: E402

REPLAY = PROJECT_ROOT / "reports" / "e0" / "webcam_replay_set"
E1_WEBCAM = PROJECT_ROOT / "reports" / "e1" / "comparison"
DROP = pose_landmark_columns(GROUPS, range(23, 33))


def dirs(suffix: str) -> dict[int, Path]:
    base = f"hands_pose_w30_bilstm_e100_augtemp{suffix}"
    return {s: EXP / (base if s == 42 else f"{base}_s{s}") for s in SEEDS}


def keep(X: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.delete(X, DROP, axis=-1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    ref, e2 = dirs(""), dirs("_upperpose")
    checks, seed_rows, preds = [], [], {}
    for s in SEEDS:
        cr, ce = (json.loads((d[s] / "config.json").read_text()) for d in (ref, e2))
        diff = {k for k in set(cr["step4"]) | set(ce["step4"]) if cr["step4"].get(k) != ce["step4"].get(k)}
        me = json.loads((e2[s] / "metadata.json").read_text())
        checks.append(dict(seed=s, spec_equal=cr["spec"] == ce["spec"], config_diff=sorted(diff),
                           config_diff_only_ablation=diff == {"feature_ablation"}, feature_dim=me["feature_dim"],
                           dropped_columns_match=me.get("dropped_feature_columns") == DROP,
                           test_evaluated=me.get("test_evaluated"),
                           no_test_files=not any(p.name.endswith(("_test.csv", "_test.json", "_test.npz")) for p in e2[s].iterdir())))
        for name, d in (("e1", ref[s]), ("e2", e2[s])):
            preds[(name, s)] = pd.read_csv(d / "predictions_val.csv").sort_values("sample_id", ignore_index=True)
            mv, mt = (json.loads((d / f"metrics_{x}.json").read_text()) for x in ("val", "train"))
            md = json.loads((d / "metadata.json").read_text())
            seed_rows.append(dict(model=name, seed=s, val_macro_f1=mv["macro_f1"], val_accuracy=mv["accuracy"],
                                  train_accuracy=mt["accuracy"], train_minus_val_acc=mt["accuracy"] - mv["accuracy"],
                                  best_epoch=md.get("best_epoch"), epochs=md.get("epochs")))
    S = pd.DataFrame(seed_rows)
    y = preds[("e1", 42)].y_true.to_numpy()
    assert all(p.sample_id.equals(preds[("e1", 42)].sample_id) for p in preds.values())
    bidx = boot_indices(len(y), strata=y)
    G, boots = [], []
    for s in SEEDS:
        pr, pe = preds[("e1", s)].y_pred.to_numpy(), preds[("e2", s)].y_pred.to_numpy()
        bg = np.array([macro_f1(y[i], pe[i]) - macro_f1(y[i], pr[i]) for i in bidx])
        boots.append(bg)
        G.append(dict(seed=s, val_macro_f1_change=macro_f1(y, pe) - macro_f1(y, pr), ci_lo=ci(bg)[0], ci_hi=ci(bg)[1],
                      e2_only_correct=int(((pe == y) & (pr != y)).sum()), e1_only_correct=int(((pr == y) & (pe != y)).sum())))
    G = pd.DataFrame(G)
    mean_change, mci = float(G.val_macro_f1_change.mean()), ci(np.mean(boots, axis=0))
    pc = {m: np.mean([per_class_f1(y, preds[(m, s)].y_pred.to_numpy(), 100) for s in SEEDS], axis=0) for m in ("e1", "e2")}
    words = preds[("e1", 42)].drop_duplicates("y_true").set_index("y_true").word.sort_index().reindex(range(100))
    PC = pd.DataFrame({"word": words.to_numpy(), "e1_f1": pc["e1"], "e2_f1": pc["e2"]})
    PC["diff"] = PC.e2_f1 - PC.e1_f1
    better = int((G.val_macro_f1_change > 0).sum())
    rule = {"mean_change": mean_change, "mean_change_ci": mci,
            "ni_mean_ge_-0.01": bool(mean_change >= -0.01), "ni_ci_lower_ge_-0.02": bool(mci[0] >= -0.02),
            "classes_with_zero_f1_e2": PC.loc[PC.e2_f1 == 0, "word"].tolist(), "no_zero_class": bool((PC.e2_f1 > 0).all()),
            "seeds_better": better, "sign_test_p": sign_test(better, 5),
            "improvement_rule_met": bool(mean_change >= 0.02 and better >= 4 and (PC.e2_f1 > 0).all())}
    rule["non_inferiority_met"] = bool(rule["ni_mean_ge_-0.01"] and rule["ni_ci_lower_ge_-0.02"] and rule["no_zero_class"])

    import e0_03_robustness as rob  # noqa: E402
    rows, X = rob.val_inputs()
    assert rows.sample_id.equals(preds[("e1", 42)].sample_id)
    RB = []
    for s, d in e2.items():
        model, _ = load_run(d)
        pc_ = logits(model, keep(X["clean"])).argmax(1)
        if not (pc_ == preds[("e2", s)].y_pred.to_numpy()).all():
            raise RuntimeError(f"E2 seed {s}: clean validation inputs do not reproduce the saved predictions")
        for p in PERTURBATIONS:
            RB.append(dict(seed=s, perturbation=p, d_accuracy=(logits(model, keep(X[p])).argmax(1) == y).mean() - (pc_ == y).mean()))
    RB = pd.DataFrame(RB).groupby("perturbation", sort=False).d_accuracy.agg(["mean", "min", "max"]).reset_index()

    man = pd.read_csv(REPLAY / "manifest.csv")
    Xw = keep(np.load(REPLAY / "features.npz")["X"])
    classes = json.loads((PROJECT_ROOT / "models/candidates/hands_pose_w30_bilstm_e100_s3/class_mapping.json").read_text())["index_to_class"]
    core = (man.core & man.excluded.isna()).to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    WR = []
    for s, d in e2.items():
        model, _ = load_run(d)
        top = np.argsort(-logits(model, Xw), axis=1)[:, :5]
        pe = np.array([classes[str(top[r, 0])] if r >= 0 else None for r in man.feature_row], dtype=object)
        t5 = np.array([(r >= 0) and (e in [classes[str(k)] for k in top[r]]) for r, e in zip(man.feature_row, man.expected)])
        ce = pe == man.expected.to_numpy()
        pd.DataFrame({"session": man.session, "attempt": man.attempt, "expected": man.expected, "core": core,
                      "predicted": pe, "correct": ce, "top5": t5}).to_csv(out / f"webcam_predictions_e2_s{s}.csv", index=False)
        r1 = pd.read_csv(E1_WEBCAM / f"webcam_predictions_e1_s{s}.csv")
        cr, t5r = r1.correct.to_numpy().astype(bool), r1.top5.to_numpy().astype(bool)
        b, c = int((ce & ~cr)[core].sum()), int((cr & ~ce)[core].sum())
        WR.append(dict(seed=s, core_attempts=int(core.sum()), e1_top1=cr[core].mean(), e2_top1=ce[core].mean(),
                       e1_top5=t5r[core].mean(), e2_top5=t5[core].mean(), e2_only=b, e1_only=c, exact_mcnemar_p=sign_test(b, b + c)))
    WR = pd.DataFrame(WR)

    pd.DataFrame(checks).to_csv(out / "config_checks.csv", index=False)
    S.to_csv(out / "per_seed.csv", index=False)
    G.to_csv(out / "val_change_paired.csv", index=False)
    PC.to_csv(out / "per_class_val_f1.csv", index=False)
    RB.to_csv(out / "robustness_e2.csv", index=False)
    WR.to_csv(out / "webcam_replay_paired.csv", index=False)
    summary = {"rule": rule,
               "configs_ok": bool(all(c["spec_equal"] and c["config_diff_only_ablation"] and c["dropped_columns_match"]
                                      and c["feature_dim"] == 221 and c["test_evaluated"] is False and c["no_test_files"]
                                      for c in checks)),
               "overfitting": S.groupby("model")[["train_accuracy", "val_accuracy", "train_minus_val_acc"]].mean().to_dict("index"),
               "per_class": {"better": int((PC["diff"] > 0).sum()), "worse": int((PC["diff"] < 0).sum()),
                             "unchanged": int((PC["diff"] == 0).sum())},
               "webcam_core_top1_mean": {"e1": float(WR.e1_top1.mean()), "e2": float(WR.e2_top1.mean())},
               "note": "validation and webcam development set only; test not evaluated"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=1, default=float))
    for t in (S, G, RB, WR):
        print(t.round(4).to_string(index=False))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
