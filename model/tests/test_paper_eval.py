"""Paper evaluation scripts: post-processing replay scoring (pp_01) and the one-time final test (e3_01)."""
import importlib.util
import sys

import pandas as pd
import pytest

from asl.config import PROJECT_ROOT
from asl.inference.evaluation import Attempt
from asl.inference.postprocess import Prediction
from asl.inference.recognizer import SignEvent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load("pp_01_postprocessing_replay")


def _ev(label, first, status="ok", conf=0.9, suppressed=False):
    p = Prediction(status, label, 0, conf, ((label, conf),), "")
    emitted = None if (status != "ok" or suppressed) else label
    return SignEvent(p, emitted, first - 0.5, first + 1.4, 40, False, first + 1.0, first, suppressed, first)


def test_grid_is_predeclared_one_change_at_a_time():
    assert pp.GRID[0] == {"cooldown_s": 0.0, "confidence_threshold": 0.0}
    assert all(g["cooldown_s"] == 0.0 or g["confidence_threshold"] == 0.0 for g in pp.GRID)
    assert sorted({g["cooldown_s"] for g in pp.GRID}) == [0.0, 0.5, 1.0]
    assert sorted({g["confidence_threshold"] for g in pp.GRID}) == [0.0, 0.4, 0.57, 0.7]


def test_first_emitted_label_is_the_answer_and_later_ones_are_duplicates():
    a = Attempt(0, "water", 3.0, 8.0, ready_start_s=0.0)
    a.events = [(_ev("wet", 2.0), 3.5),                                  # started before GO: ignored
                (_ev("water", 4.0, status="low_confidence", conf=0.3), 5.0),  # not emitted
                (_ev("water", 5.0), 6.0),                                # the answer
                (_ev("red", 6.0), 7.0),                                  # duplicate emission
                (_ev("red", 6.5, suppressed=True), 7.2)]                 # cooldown: not emitted
    r = pp.score_attempt(a)
    assert r["answer"] == "water" and r["correct"] and r["answered"]
    assert r["emitted"] == 2 and r["duplicate_emissions"] == 1
    assert r["low_confidence"] == 1 and r["suppressed"] == 1 and not r["no_segment"]
    empty = Attempt(1, "red", 11.0, 16.0, ready_start_s=8.0)
    r = pp.score_attempt(empty)
    assert r["no_segment"] and not r["answered"] and not r["correct"] and r["answer"] is None


def test_summary_coverage_and_precision():
    rows = pd.DataFrame([{"correct": True, "answered": True, "duplicate_emissions": 1, "no_segment": False,
                          "suppressed": 0, "low_confidence": 0},
                         {"correct": False, "answered": True, "duplicate_emissions": 0, "no_segment": False,
                          "suppressed": 1, "low_confidence": 0},
                         {"correct": False, "answered": False, "duplicate_emissions": 0, "no_segment": True,
                          "suppressed": 0, "low_confidence": 2}])
    s = pp.summarize(rows)
    assert s["top1"] == pytest.approx(1 / 3) and s["coverage"] == pytest.approx(2 / 3)
    assert s["precision_of_answers"] == pytest.approx(0.5) and s["attempts_with_duplicates"] == 1
    assert s["no_segment"] == 1 and s["suppressed_segments"] == 1 and s["low_confidence_segments"] == 2


def test_pp_refuses_non_empty_output(tmp_path, monkeypatch):
    mod = _load("pp_01_postprocessing_replay")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep").write_text("x")
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["pp", "--out", "out"])
    with pytest.raises(SystemExit, match="not empty"):
        mod.main()


# ---------------------------------------------------------------- one-time final test (e3_01)
import json  # noqa: E402

import numpy as np  # noqa: E402

from asl.config import STEP2_CONFIG, load_yaml  # noqa: E402
from asl.landmarks.layout import pose_landmark_columns  # noqa: E402
from asl.models.experiment import ExperimentSpec, run_experiment  # noqa: E402
from asl.sequences.dataset import SequenceDataset  # noqa: E402
from asl.sequences.labels import build_label_mapping  # noqa: E402
from asl.sequences.synthetic import make_synthetic_store  # noqa: E402

e3 = _load("e3_01_final_test")
C2 = load_yaml(STEP2_CONFIG)


def test_final_test_analysis_intervals_and_pairs():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 8)
    good = {s: np.where(rng.random(len(y)) < 0.8, y, (y + 1) % 10) for s in (1, 2, 3)}
    bad = {s: np.where(rng.random(len(y)) < 0.5, y, (y + 1) % 10) for s in (1, 2, 3)}
    res = e3.analyse(y, {"baseline": bad, "e1": good}, [f"w{i}" for i in range(10)])
    m = res["seed_mean"].set_index("model")
    assert m.loc["e1", "accuracy_mean"] > m.loc["baseline", "accuracy_mean"]
    assert m.loc["e1", "acc_lo"] <= m.loc["e1", "accuracy_mean"] <= m.loc["e1", "acc_hi"]
    d = res["paired"].iloc[0]
    assert d["comparison"] == "e1 - baseline" and d["d_acc_lo"] > 0 and d["seeds_better_macro_f1"] == 3
    assert list(res["per_class"].columns) == ["word", "baseline_f1", "e1_f1"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    words = [f"w{i:02d}" for i in range(3)]
    mapping = build_label_mapping(pd.DataFrame({"word": words}), "test")
    root = tmp_path_factory.mktemp("final")
    idx, arrays = make_synthetic_store(root, words, mapping, C2["output"], [30],
                                       per_class={"train": 4, "val": 2, "test": 2}, seed=4)
    c4 = json.loads(json.dumps(load_yaml(PROJECT_ROOT / "configs" / "step4_training.yaml")))
    c4["latency"].update(warmup_runs=1, timed_runs=2, end_to_end_samples=2)
    drop = pose_landmark_columns(C2["configurations"]["hands_pose"], range(23, 33))
    out = {}
    for name, cols in (("full", []), ("upper", drop)):
        r = run_experiment(ExperimentSpec("bilstm", "hands_pose", 30), store="f", index=idx, arrays_dir=arrays,
                           mapping=mapping, configurations=C2["configurations"], cfg=c4, experiments_root=root / "exp",
                           results_csv=root / "results.csv", max_epochs=2, tag=name, evaluate_test=False, drop_columns=cols)
        out[name] = r["exp_dir"]
    return idx, arrays, mapping, out, drop


def test_final_test_inputs_and_predictions_match_the_training_pipeline(runs):
    idx, arrays, mapping, out, drop = runs
    rows, X = e3.split_inputs(idx, arrays, "test")
    for name, cols in (("full", []), ("upper", drop)):
        p = e3.predict_runs({0: out[name]}, X, cols)[0]
        ds = SequenceDataset(idx, arrays, "test", 30, "hands_pose", C2["configurations"], mapping, drop_columns=cols)
        assert ds.rows.sample_id.equals(rows.sample_id)
        import torch
        from e0_common import load_run
        model, _ = load_run(out[name])
        with torch.no_grad():
            ref = np.array([int(model(ds[i][0][None]).argmax(1)) for i in range(len(ds))])
        assert (p == ref).all()
    with pytest.raises(ValueError, match="feature set"):
        e3.predict_runs({0: out["upper"]}, X, [])                       # wrong feature set is refused


def test_final_test_refuses_repeat_and_unconfirmed_runs(runs, tmp_path, monkeypatch):
    idx, arrays, mapping, out, drop = runs
    (out["full"] / "predictions_test.csv").write_text("x")
    with pytest.raises(SystemExit, match="already has test outputs"):
        e3.check_no_test_outputs({"e1": {0: out["full"]}}, names=("e1",))
    (out["full"] / "predictions_test.csv").unlink()
    e3.check_no_test_outputs({"e1": {0: out["full"]}}, names=("e1",))
    mod = _load("e3_01_final_test")
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["e3", "--out", "o"])
    with pytest.raises(SystemExit, match="confirm-development-closed"):
        mod.main()
    (tmp_path / "o").mkdir()
    monkeypatch.setattr(sys, "argv", ["e3", "--out", "o", "--confirm-development-closed"])
    with pytest.raises(SystemExit, match="already been run"):
        mod.main()
