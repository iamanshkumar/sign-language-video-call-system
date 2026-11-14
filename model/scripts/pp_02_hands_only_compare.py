"""Paper Objective 1 (OFFLINE): hands-only vs hands + pose BiLSTM on VALIDATION (plan:
reports/paper/hands_only/PLAN.md, written before the runs).

  PYTHONPATH=src .venv/bin/python scripts/pp_02_hands_only_compare.py --out reports/paper/hands_only/comparison

Runs paired by seed (42, 1, 2, 3, 4): hands_w30_bilstm_e100[_sN] (128 features) vs hands_pose_w30_bilstm_e100[_sN]
(261 features); configs must be identical apart from the feature configuration and the test-evaluation
flag. Descriptive: per-seed and mean validation macro-F1 / accuracy, paired mean difference with a
class-stratified 95 % bootstrap interval, seeds in each direction, train - validation gap, per-class F1.
No test inference. --out must be new or empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import EXP, SEEDS, boot_indices, ci, macro_f1  # noqa: E402
from e1_01_compare import per_class_f1, sign_test  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402


def dirs(base: str) -> dict[int, Path]:
    return {s: EXP / (base if s == 42 else f"{base}_s{s}") for s in SEEDS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    hp, h = dirs("hands_pose_w30_bilstm_e100"), dirs("hands_w30_bilstm_e100")
    checks, rows, preds = [], [], {}
    for s in SEEDS:
        ca, cb = (json.loads((d[s] / "config.json").read_text()) for d in (hp, h))
        diff = {k for k in set(ca["step4"]) | set(cb["step4"]) if ca["step4"].get(k) != cb["step4"].get(k)}
        spec_diff = {k for k in ca["spec"] if ca["spec"][k] != cb["spec"][k]}
        mb = json.loads((h[s] / "metadata.json").read_text())
        checks.append(dict(seed=s, spec_diff=sorted(spec_diff), config_diff=sorted(diff),
                           ok=spec_diff == {"feature_config"} and diff <= {"evaluate_test"} and mb.get("test_evaluated") is False,
                           feature_dim=mb["feature_dim"]))
        for name, d in (("hands_pose", hp[s]), ("hands", h[s])):
            preds[(name, s)] = pd.read_csv(d / "predictions_val.csv").sort_values("sample_id", ignore_index=True)
            mv, mt = (json.loads((d / f"metrics_{x}.json").read_text()) for x in ("val", "train"))
            rows.append(dict(representation=name, seed=s, features=json.loads((d / "metadata.json").read_text())["feature_dim"],
                             val_macro_f1=mv["macro_f1"], val_accuracy=mv["accuracy"], train_accuracy=mt["accuracy"],
                             train_minus_val_acc=mt["accuracy"] - mv["accuracy"]))
    S = pd.DataFrame(rows)
    y = preds[("hands_pose", 42)].y_true.to_numpy()
    assert all(p.sample_id.equals(preds[("hands_pose", 42)].sample_id) for p in preds.values())
    bidx = boot_indices(len(y), strata=y)
    D, boots = [], []
    for s in SEEDS:
        a, b = preds[("hands", s)].y_pred.to_numpy(), preds[("hands_pose", s)].y_pred.to_numpy()
        bd = np.array([macro_f1(y[i], a[i]) - macro_f1(y[i], b[i]) for i in bidx])
        boots.append(bd)
        D.append(dict(seed=s, d_val_macro_f1=macro_f1(y, a) - macro_f1(y, b), ci_lo=ci(bd)[0], ci_hi=ci(bd)[1]))
    D = pd.DataFrame(D)
    words = preds[("hands_pose", 42)].drop_duplicates("y_true").set_index("y_true").word.sort_index().reindex(range(100))
    PC = pd.DataFrame({"word": words.to_numpy(),
                       **{f"{n}_f1": np.mean([per_class_f1(y, preds[(n, s)].y_pred.to_numpy(), 100) for s in SEEDS], 0)
                          for n in ("hands_pose", "hands")}})
    PC["diff"] = PC.hands_f1 - PC.hands_pose_f1
    lower = int((D.d_val_macro_f1 < 0).sum())
    summary = {"configs_ok": bool(all(c["ok"] for c in checks)),
               "mean_val_macro_f1": S.groupby("representation").val_macro_f1.mean().to_dict(),
               "mean_val_accuracy": S.groupby("representation").val_accuracy.mean().to_dict(),
               "mean_difference_hands_minus_hands_pose": float(D.d_val_macro_f1.mean()),
               "difference_ci": ci(np.mean(boots, 0)), "seeds_hands_lower": lower, "sign_test_p": sign_test(lower, 5),
               "train_minus_val_acc": S.groupby("representation").train_minus_val_acc.mean().to_dict(),
               "per_class": {"hands_better": int((PC["diff"] > 0).sum()), "hands_worse": int((PC["diff"] < 0).sum())},
               "note": "validation only; descriptive; test not evaluated"}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(out / "config_checks.csv", index=False)
    S.to_csv(out / "per_seed.csv", index=False)
    D.to_csv(out / "paired_difference.csv", index=False)
    PC.to_csv(out / "per_class_val_f1.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=1, default=float)); print(S.round(4).to_string(index=False)); print(D.round(4).to_string(index=False))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
