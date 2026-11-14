"""Paper precision / recall from saved predictions (paper_05)."""
import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_fscore_support

from asl.config import PROJECT_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prf = _load("paper_05_precision_recall")


def test_bootstrap_macro_pr_matches_sklearn_on_each_resample():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(prf.N_CLASSES), 7)
    p = np.where(rng.random(len(y)) < 0.6, y, rng.integers(0, prf.N_CLASSES, len(y)))
    idx = prf.boot_indices(len(y), n_boot=5, strata=y)
    P, R = prf.boot_macro_pr(y, p, idx)
    for b, i in enumerate(idx):
        sp, sr, _, _ = precision_recall_fscore_support(y[i], p[i], labels=np.arange(prf.N_CLASSES),
                                                       average="macro", zero_division=0)
        assert P[b] == pytest.approx(sp) and R[b] == pytest.approx(sr)


def _frame(n=prf.N_TEST):
    y = np.arange(n) % prf.N_CLASSES
    rows = [pd.DataFrame({"model": "m", "seed": s, "sample_id": [f"c{i:04d}" for i in range(n)], "y_true": y, "y_pred": y})
            for s in prf.SEEDS]
    return pd.concat(rows, ignore_index=True)


def test_structure_check_accepts_complete_runs():
    prf.check_structure(_frame())


def test_structure_check_rejects_duplicates_missing_rows_and_label_mismatch():
    d = _frame()
    with pytest.raises(SystemExit, match="duplicate"):
        prf.check_structure(pd.concat([d, d.iloc[:1]]))
    with pytest.raises(SystemExit, match="rows"):
        prf.check_structure(d.iloc[1:])
    bad = d.copy()
    bad.loc[bad.seed == 1, "y_true"] = (bad.loc[bad.seed == 1, "y_true"] + 1) % prf.N_CLASSES
    with pytest.raises(SystemExit, match="true labels differ"):
        prf.check_structure(bad)


@pytest.mark.skipif(not (PROJECT_ROOT / "reports/paper/final_test_prf/seed_mean.csv").exists(), reason="not generated")
def test_saved_outputs_reconcile_with_final_test_reports():
    rec = pd.read_csv(PROJECT_ROOT / "reports/paper/final_test_prf/reconciliation.csv")
    assert rec.match.all() and len(rec) == 100
    sm = pd.read_csv(PROJECT_ROOT / "reports/paper/final_test_prf/seed_mean.csv").set_index("model")
    ft = pd.read_csv(PROJECT_ROOT / "reports/paper/final_test/seed_mean.csv").set_index("model")
    for m in ("baseline", "e1", "e2"):
        assert sm.loc[m, "accuracy_mean"] == pytest.approx(ft.loc[m, "accuracy_mean"], abs=1e-12)
        assert sm.loc[m, "macro_f1_mean"] == pytest.approx(ft.loc[m, "macro_f1_mean"], abs=1e-12)
