"""E0 offline evaluation helpers (scripts/e0_*.py): metrics, bootstrap, perturbations, no-overwrite."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from asl.config import PROJECT_ROOT
from asl.landmarks.layout import GROUP_INDEX, OBSERVED, PRESENT
from asl.models.metrics import classification_metrics
from asl.sequences.dataset import load_sequence

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import e0_common as e0  # noqa: E402


def _arrays(n=20, seed=0):
    rng = np.random.default_rng(seed)
    flags = np.zeros((n, 4), np.uint8)
    flags[:, GROUP_INDEX["pose"]] = OBSERVED | PRESENT
    flags[:, GROUP_INDEX["right_hand"]] = OBSERVED | PRESENT
    flags[5:15, GROUP_INDEX["left_hand"]] = OBSERVED | PRESENT
    a = {"pose": rng.normal(size=(n, 33, 3)).astype(np.float32),
         "left_hand": rng.normal(size=(n, 21, 3)).astype(np.float32),
         "right_hand": rng.normal(size=(n, 21, 3)).astype(np.float32),
         "pose_vis": rng.random((n, 33)).astype(np.float32), "flags": flags}
    a["left_hand"][(flags[:, GROUP_INDEX["left_hand"]] & PRESENT) == 0] = 0.0
    return a


def test_macro_f1_matches_project_metric():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y, p = rng.integers(0, 12, 60), rng.integers(0, 15, 60)
        assert e0.macro_f1(y, p) == pytest.approx(classification_metrics(y, p, 15)["macro_f1"], abs=1e-12)


def test_stratified_bootstrap_keeps_class_supports():
    y = np.repeat(np.arange(5), [3, 7, 2, 9, 4])
    idx = e0.boot_indices(len(y), n_boot=50, strata=y)
    assert idx.shape == (50, len(y))
    for row in idx:
        assert (np.bincount(y[row], minlength=5) == np.bincount(y, minlength=5)).all()
    assert e0.boot_indices(10, n_boot=3).shape == (3, 10)


def test_clean_perturbation_equals_training_loader(tmp_path):
    a = _arrays()
    np.savez(tmp_path / "x.npz", **a)
    ref = load_sequence(tmp_path / "x.npz", e0.GROUPS, 3, 17, 30)
    assert np.array_equal(e0.features(e0.perturb(a, 3, 17, "clean")), ref)


def test_perturbations_do_what_they_say():
    a = _arrays()
    s, e = 2, 17
    n = e - s + 1
    assert (e0.perturb(a, s, e, "hips_hidden")["pose_vis"][:, 23:25] == 0).all()
    assert len(e0.perturb(a, s, e, "hold_end")["flags"]) == n + round(0.5 * n)
    assert len(e0.perturb(a, s, e, "repeat_x2")["flags"]) == 2 * n
    d = e0.perturb(a, s, e, "drop_minor_hand")                      # left hand is the minor hand here
    assert (d["flags"][:, GROUP_INDEX["left_hand"]] == 0).all() and (d["left_hand"] == 0).all()
    m = e0.perturb(e0.perturb(a, s, e, "mirror") | {}, 0, n - 1, "mirror")
    c = e0.perturb(a, s, e, "clean")
    for k in c:
        assert np.allclose(m[k], c[k])                                 # mirroring twice is the identity
    r = e0.perturb(a, s, e, "rotate_10")
    assert np.allclose(np.linalg.norm(r["pose"][..., :2], axis=-1), np.linalg.norm(c["pose"][..., :2], axis=-1), atol=1e-5)
    for name in e0.PERTURBATIONS:
        assert e0.features(e0.perturb(a, s, e, name)).shape == (30, 261)
    with pytest.raises(ValueError):
        e0.perturb(a, s, e, "unknown")


@pytest.mark.parametrize("script", ["e0_01_bootstrap_ci", "e0_02_webcam_replay_set", "e0_03_robustness",
                                    "e1_01_compare", "e2_01_compare", "pp_02_hands_only_compare"])
def test_e0_scripts_refuse_non_empty_output(tmp_path, monkeypatch, script):
    spec = importlib.util.spec_from_file_location(script, PROJECT_ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep").write_text("x")
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--out", "out"])
    with pytest.raises(SystemExit, match="not empty"):
        mod.main()
    assert (tmp_path / "out" / "keep").read_text() == "x"


def test_bootstrap_run_reproduces_saved_metrics_if_present():
    run = e0.run_dirs("bilstm")[3]
    if not (run / "predictions_val.csv").exists():
        pytest.skip("experiment runs not present")
    import json
    p = pd.read_csv(run / "predictions_val.csv")
    saved = json.loads((run / "metrics_val.json").read_text())
    assert e0.macro_f1(p.y_true.to_numpy(), p.y_pred.to_numpy()) == pytest.approx(saved["macro_f1"], abs=1e-9)
