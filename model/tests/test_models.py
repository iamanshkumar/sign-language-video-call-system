"""Step 4 tests: models, RF statistics, training loop, selection, metrics, latency, experiments.

Tiny synthetic tensors / stores only; CPU only."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.layout import feature_dim
from asl.models import latency as lat
from asl.models.experiment import ExperimentSpec, experiment_matrix, run_experiment
from asl.models.metrics import classification_metrics, confusion, per_class_table
from asl.models.recurrent import BiLSTMClassifier, LSTMClassifier, build_model
from asl.models.rf import build_rf, temporal_statistics
from asl.models.train import BestTracker, fit, load_checkpoint, make_optimizer, make_scheduler, set_seed
from asl.sequences.dataset import SequenceDataset, make_dataloaders
from asl.sequences.labels import build_label_mapping
from asl.sequences.synthetic import make_synthetic_store

CFG2 = load_yaml(STEP2_CONFIG)
CFG4 = load_yaml(PROJECT_ROOT / "configs" / "step4_training.yaml")
CONFIGS = CFG2["configurations"]
NC = 2207
FEATS = {"hands": 128, "hands_pose": 261, "hands_pose_face": 1666}


@pytest.fixture(scope="module")
def mapping():
    words = [f"w{i:04d}" for i in range(NC)]
    return build_label_mapping(pd.DataFrame({"word": words}), "test")


@pytest.fixture(scope="module")
def store(tmp_path_factory, mapping):
    root = tmp_path_factory.mktemp("store")
    words = sorted(mapping["word_to_index"])[:3]
    idx, arrays = make_synthetic_store(root, words, mapping, CFG2["output"], [30, 45, 60],
                                       per_class={"train": 4, "val": 2, "test": 2}, seed=1)
    return idx, arrays


def _cfg(**neural):
    c = json.loads(json.dumps(CFG4))
    c["neural"].update(neural)
    c["latency"].update(warmup_runs=2, timed_runs=5, end_to_end_samples=3)
    c["random_forest"]["n_estimators"] = 10
    return c


# ---------------------------------------------------------------- 1-4 models
@pytest.mark.parametrize("cls,directions", [(LSTMClassifier, 1), (BiLSTMClassifier, 2)])
def test_forward_shapes_and_classes(cls, directions):
    m = cls(input_dim=128, num_classes=NC)
    x = torch.randn(5, 30, 128)
    assert m(x).shape == (5, NC)
    assert m.lstm.bidirectional == (directions == 2) and m.lstm.num_layers == 2 and m.lstm.hidden_size == 128
    assert m.fc.in_features == 128 * directions and m.fc.out_features == NC
    assert m.lstm.dropout == 0.30 and m.dropout.p == 0.30
    with pytest.raises(ValueError):
        m(torch.randn(5, 30, 127))


def test_bilstm_uses_both_directions():
    torch.manual_seed(0)
    m = BiLSTMClassifier(4, 3, hidden_size=8).eval()  # eval: no inter-layer dropout
    x = torch.randn(2, 10, 4)
    out, _ = m.lstm(x)
    enc = m.encode(x)
    assert torch.allclose(enc[:, :8], out[:, -1, :8], atol=1e-6)   # forward direction at the last step
    assert torch.allclose(enc[:, 8:], out[:, 0, 8:], atol=1e-6)    # backward direction at the first step


# ---------------------------------------------------------------- 5-7 RF statistics
def test_temporal_statistics_values_and_dimension():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)  # W=4, F=3
    s = temporal_statistics(x)
    assert s.shape == (12,)
    assert np.allclose(s[:3], x.mean(0)) and np.allclose(s[3:6], x.std(0)) and np.allclose(s[6:9], x.min(0))
    assert np.allclose(s[9:], x.max(0))
    for w in (30, 45, 60):  # same dimension for every window
        assert temporal_statistics(np.random.rand(w, 261)).shape == (4 * 261,)
    b = temporal_statistics(np.stack([x, x * 2]))
    assert b.shape == (2, 12) and np.allclose(b[0], s)


def test_temporal_statistics_deterministic_and_rf_deterministic():
    rng = np.random.default_rng(0)
    X = np.stack([temporal_statistics(rng.normal(size=(30, 8))) for _ in range(20)])
    y = np.arange(20) % 4
    a = build_rf({**CFG4["random_forest"], "n_estimators": 15}).fit(X, y).predict_proba(X)
    b = build_rf({**CFG4["random_forest"], "n_estimators": 15}).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)
    seq = rng.normal(size=(45, 10))
    assert np.array_equal(temporal_statistics(seq), temporal_statistics(seq.copy()))


# ---------------------------------------------------------------- 8, 19, 20 data + labels
@pytest.mark.parametrize("fc", ["hands", "hands_pose", "hands_pose_face"])
@pytest.mark.parametrize("w", [30, 45, 60])
def test_every_configuration_and_window_through_models(store, mapping, fc, w):
    idx, arrays = store
    ds = SequenceDataset(idx, arrays, "train", w, fc, CONFIGS, mapping)
    x, y = next(iter(make_dataloaders({"train": ds}, 4, 0, 0)["train"]))
    assert x.shape == (4, w, FEATS[fc]) and y.dtype == torch.long
    for name in ("lstm", "bilstm"):
        assert build_model(name, FEATS[fc], NC, CFG4["neural"])(x).shape == (4, NC)
    assert temporal_statistics(x[0].numpy()).shape == (4 * FEATS[fc],)


def test_labels_are_shared_mapping_integers(store, mapping):
    idx, arrays = store
    for split in ("train", "val", "test"):
        ds = SequenceDataset(idx, arrays, split, 30, "hands", CONFIGS, mapping)
        for i in range(len(ds)):
            assert int(ds[i][1]) == mapping["word_to_index"][ds.metadata(i)["word"]]


# ---------------------------------------------------------------- 11-13 selection, early stopping, scheduler
def test_best_tracker_selects_strictly_best_and_stops():
    t = BestTracker(patience=3)
    seq = [0.1, 0.3, 0.3, 0.2, 0.25, 0.29]  # best at epoch 2; tie at 3 does not count
    flags = [t.update(e, m) for e, m in enumerate(seq, start=1)]
    assert t.best_epoch == 2 and t.best == 0.3
    assert [f[0] for f in flags] == [True, True, False, False, False, False]
    assert [f[1] for f in flags] == [False, False, False, False, True, True]


def test_scheduler_halves_lr_after_patience():
    m = torch.nn.Linear(2, 2)
    opt = make_optimizer(m, CFG4["neural"])
    sch = make_scheduler(opt, CFG4["neural"])
    lrs = []
    for metric in [0.5, 0.5, 0.5, 0.5, 0.5]:  # no improvement after the first step
        sch.step(metric)
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[:4] == [1e-3] * 4 and lrs[4] == pytest.approx(5e-4)
    assert isinstance(opt, torch.optim.AdamW) and opt.param_groups[0]["weight_decay"] == 1e-4


def test_fit_saves_best_checkpoint_and_early_stops(tmp_path, store, mapping):
    idx, arrays = store
    set_seed(0)
    dsets = {s: SequenceDataset(idx, arrays, s, 30, "hands", CONFIGS, mapping) for s in ("train", "val")}
    loaders = make_dataloaders(dsets, 4, 0, 0)
    cfg = {**CFG4["neural"], "early_stopping_patience": 2}
    model = LSTMClassifier(128, NC)
    out = fit(model, loaders["train"], loaders["val"], cfg, tmp_path, torch.device("cpu"), max_epochs=20)
    h = pd.DataFrame(out["history"])
    assert (tmp_path / "best.pt").exists() and out["checkpoint"] == "best.pt"
    assert out["best_epoch"] == int(h.loc[h["val_macro_f1"].idxmax(), "epoch"])
    assert out["best_val_macro_f1"] == h["val_macro_f1"].max()
    if out["stopped_early"]:
        assert out["epochs_run"] == out["best_epoch"] + 2
    meta = load_checkpoint(tmp_path / "best.pt", LSTMClassifier(128, NC))
    assert meta["epoch"] == out["best_epoch"]
    assert {"train_loss", "val_loss", "val_macro_f1", "train_macro_f1", "lr"} <= set(h.columns)


def test_training_loss_decreases_on_learnable_data():
    set_seed(0)
    x = torch.randn(32, 30, 6)
    y = (x[:, :, 0].mean(1) > 0).long()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)
    out = fit(LSTMClassifier(6, 2, hidden_size=16), loader, loader, CFG4["neural"], __import__("pathlib").Path(
        __import__("tempfile").mkdtemp()), torch.device("cpu"), max_epochs=15)
    losses = [r["train_loss"] for r in out["history"]]
    assert losses[-1] < losses[0]


def test_fit_refuses_without_validation_unless_explicit(tmp_path, store, mapping):
    idx, arrays = store
    ds = SequenceDataset(idx, arrays, "train", 30, "hands", CONFIGS, mapping)
    loader = make_dataloaders({"train": ds}, 4, 0, 0)["train"]
    with pytest.raises(ValueError):
        fit(LSTMClassifier(128, NC), loader, None, CFG4["neural"], tmp_path, torch.device("cpu"), max_epochs=1)
    out = fit(LSTMClassifier(128, NC), loader, None, CFG4["neural"], tmp_path, torch.device("cpu"),
              allow_no_validation=True, max_epochs=2)
    assert (tmp_path / "final.pt").exists() and out["best_epoch"] is None and "none" in out["selection"]


# ---------------------------------------------------------------- 14-15 metrics
def test_metrics_hand_computed():
    y = np.array([0, 0, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 0])
    m = classification_metrics(y, p, NC)
    assert m["accuracy"] == pytest.approx(0.6) and m["correct"] == 3 and m["incorrect"] == 2
    # per class (0,1,2): precision 1/2, 2/3, 0; recall 1/2, 1, 0
    assert m["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert m["macro_recall"] == pytest.approx((0.5 + 1 + 0) / 3)
    f1 = [0.5, 0.8, 0.0]
    assert m["macro_f1"] == pytest.approx(np.mean(f1))
    assert m["weighted_f1"] == pytest.approx((2 * 0.5 + 2 * 0.8 + 0) / 5)
    assert m["vocabulary_size"] == NC and m["classes_evaluated"] == 3
    t = per_class_table(y, p, [f"w{i}" for i in range(NC)])
    assert t["support"].tolist() == [2, 2, 1] and t["word"].tolist() == ["w0", "w1", "w2"]


def test_confusion_matrix_dimensions():
    y, p = np.array([0, 5, 2206, 5]), np.array([0, 4, 2206, 5])
    cm = confusion(y, p, NC)
    assert cm.shape == (NC, NC) and cm.sum() == 4 and cm.tocsr()[5, 4] == 1 and cm.tocsr()[5, 5] == 1


# ---------------------------------------------------------------- 16 latency
def test_inference_timing():
    m = LSTMClassifier(128, NC)
    r = lat.model_latency(m, 30, 128, 4, torch.device("cpu"), warmup=2, runs=5)
    assert r["runs"] == 5 and r["batch_size"] == 4 and r["warmup"] == 2
    assert 0 < r["median_ms"] <= r["p95_ms"] and r["throughput_samples_per_s"] > 0
    c = lat.callable_latency(lambda: sum(range(100)), 1, 4, 1, "x")
    assert c["runs"] == 4 and c["mean_ms"] >= 0


# ---------------------------------------------------------------- 17 configuration
def test_experiment_configuration_matches_project_defaults():
    n = CFG4["neural"]
    assert (n["hidden_size"], n["num_layers"], n["dropout"]) == (128, 2, 0.30)
    assert (n["optimizer"], n["learning_rate"], n["weight_decay"]) == ("adamw", 1e-3, 1e-4)
    assert (n["batch_size"], n["fallback_batch_size"], n["max_epochs"], n["early_stopping_patience"]) == (64, 32, 50, 7)
    assert (n["scheduler"]["name"], n["scheduler"]["factor"], n["scheduler"]["patience"]) == ("reduce_lr_on_plateau", 0.5, 3)
    assert n["selection_metric"] == "val_macro_f1"
    specs = experiment_matrix(CFG4)
    neural = [s for s in specs if s.model != "rf"]
    assert len(neural) == 18 and len({s.experiment_id for s in specs}) == len(specs)
    assert {s.experiment_id for s in neural} >= {"hands_w30_lstm", "hands_pose_face_w60_bilstm"}
    assert ExperimentSpec("lstm", "hands", 45).experiment_id == "hands_w45_lstm"


# ---------------------------------------------------------------- 9, 10, 18 experiments
def _run(tmp_path, store, mapping, spec, name="s", **kw):
    idx, arrays = store
    return run_experiment(spec, store=name, index=idx, arrays_dir=arrays, mapping=mapping, configurations=CONFIGS,
                          cfg=_cfg(), experiments_root=tmp_path / "exp", results_csv=tmp_path / "results.csv",
                          max_epochs=3, **kw)


def test_experiment_artifacts_separation_and_no_overwrite(tmp_path, store, mapping):
    r = _run(tmp_path, store, mapping, ExperimentSpec("lstm", "hands", 30))
    d = r["exp_dir"]
    for f in ("best.pt", "history.csv", "training_curves.png", "config.json", "metadata.json", "metrics_val.json",
              "metrics_test.json", "confusion_test.npz", "per_class_test.csv", "predictions_test.csv"):
        assert (d / f).exists(), f
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["samples"] == {"train": 12, "val": 6, "test": 6} and meta["checkpoint_reload_reproduces_val_f1"]
    preds = pd.read_csv(d / "predictions_test.csv")
    idx = store[0]
    assert set(preds["sample_id"]) == set(idx.loc[idx.split == "test", "sample_id"])  # only test videos
    assert {"active_frames", "stretch", "word", "pred_word"} <= set(preds.columns)
    with pytest.raises(FileExistsError):
        _run(tmp_path, store, mapping, ExperimentSpec("lstm", "hands", 30))
    _run(tmp_path, store, mapping, ExperimentSpec("bilstm", "hands", 30))
    res = pd.read_csv(tmp_path / "results.csv")
    assert res["experiment_id"].tolist() == ["hands_w30_lstm", "hands_w30_bilstm"]  # appended, not overwritten
    assert res.loc[0, "test_samples"] == 6 and res.loc[0, "best_epoch"] >= 1


def test_rf_experiment(tmp_path, store, mapping):
    r = _run(tmp_path, store, mapping, ExperimentSpec("rf", "hands_pose", 45))
    meta = r["meta"]
    assert meta["rf_feature_dim"] == 4 * 261 and (r["exp_dir"] / "model.joblib").exists()
    assert r["row"]["test_macro_f1"] is not None and meta["selection"].startswith("none")


def test_experiment_reproducible(tmp_path, store, mapping):
    a = _run(tmp_path, store, mapping, ExperimentSpec("lstm", "hands_pose", 30), name="a")
    b = _run(tmp_path, store, mapping, ExperimentSpec("lstm", "hands_pose", 30), name="b")
    ha = pd.read_csv(a["exp_dir"] / "history.csv").drop(columns="seconds")
    hb = pd.read_csv(b["exp_dir"] / "history.csv").drop(columns="seconds")
    pd.testing.assert_frame_equal(ha, hb)
    assert a["evals"]["val"] == b["evals"]["val"] and a["evals"]["test"] == b["evals"]["test"]


def test_train_only_run_needs_explicit_flag(tmp_path, store, mapping):
    idx, arrays = store
    train_only = idx[idx.split == "train"]
    kw = dict(store="t", index=train_only, arrays_dir=arrays, mapping=mapping, configurations=CONFIGS, cfg=_cfg(),
              experiments_root=tmp_path / "exp", results_csv=tmp_path / "r.csv", max_epochs=2)
    with pytest.raises(ValueError):
        run_experiment(ExperimentSpec("lstm", "hands", 30), **kw)
    r = run_experiment(ExperimentSpec("lstm", "hands", 30), **{**kw, "store": "t2"}, allow_no_validation=True)
    assert r["row"]["test_macro_f1"] is None and "train_fit" in r["evals"] and (r["exp_dir"] / "final.pt").exists()


def test_refused_run_leaves_no_directory(tmp_path, store, mapping):
    idx, arrays = store
    kw = dict(store="r", index=idx[idx.split == "train"], arrays_dir=arrays, mapping=mapping, configurations=CONFIGS,
              cfg=_cfg(), experiments_root=tmp_path / "exp", results_csv=tmp_path / "r.csv", max_epochs=1)
    with pytest.raises(ValueError):
        run_experiment(ExperimentSpec("lstm", "hands", 30), **kw)
    assert not (tmp_path / "exp" / "r" / "hands_w30_lstm").exists()
