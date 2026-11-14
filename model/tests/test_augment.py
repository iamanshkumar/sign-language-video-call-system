"""E1 temporal augmentation: realism of each operation, training-only use, no test inference."""
import json

import numpy as np
import pandas as pd
import pytest

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.layout import GROUP_INDEX, OBSERVED, PRESENT
from asl.models.experiment import ExperimentSpec, run_experiment
from asl.sequences.augment import OPS, TemporalAugment, TemporalAugmentConfig, apply_op, find_loop
from asl.sequences.dataset import SequenceDataset, load_sequence
from asl.sequences.labels import build_label_mapping
from asl.sequences.synthetic import make_synthetic_store

CFG = TemporalAugmentConfig()


def _clip(n=40, kind="cycle", cycles=2):
    """Interval arrays with a right hand moving either in two full circles (cyclic) or in a straight line."""
    t = np.linspace(0, 1, n)
    flags = np.zeros((n, 4), np.uint8)
    flags[:, GROUP_INDEX["pose"]] = OBSERVED | PRESENT
    flags[:, GROUP_INDEX["right_hand"]] = OBSERVED | PRESENT
    if kind == "cycle":
        cx, cy = 0.5 * np.cos(2 * cycles * np.pi * t), 0.5 * np.sin(2 * cycles * np.pi * t)
    else:
        cx, cy = 2.0 * t, 0.0 * t
    rh = np.zeros((n, 21, 3), np.float32)
    rh[:, :, 0] = cx[:, None] + np.linspace(0, 0.1, 21)[None]
    rh[:, :, 1] = cy[:, None]
    return {"left_hand": np.zeros((n, 21, 3), np.float32), "right_hand": rh,
            "pose": np.ones((n, 33, 3), np.float32), "pose_vis": np.ones((n, 33), np.float32), "flags": flags,
            "frame": np.arange(n)}            # tracer: which original frame each output frame came from


def _max_step(a):
    c = a["right_hand"][:, :, :2].mean(axis=1)
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).max())


@pytest.mark.parametrize("op", OPS)
def test_every_operation_keeps_order_and_frames(op):
    a = _clip()
    out, changed = apply_op(a, op, CFG, np.random.default_rng(3))
    assert changed
    fr = out["frame"]
    assert fr[0] == 0 and fr[-1] == len(a["frame"]) - 1          # starts and ends with the sign's own ends
    assert set(fr) == set(range(len(a["frame"]))) or op == "tempo"
    if op != "repeat_span":
        assert (np.diff(fr) >= 0).all()                            # never reordered
    for k, v in out.items():
        assert len(v) == len(fr) and np.isfinite(v).all()
        if k != "frame" and op != "tempo":
            assert np.array_equal(v, a[k][fr])                     # only original frames, no synthesis
    if op == "tempo":                                              # interpolated, speed-up at most 1.5x
        assert _max_step(out) <= CFG.tempo_max * _max_step(a) + 1e-5
    elif op != "repeat_span":
        assert _max_step(out) <= _max_step(a) + 1e-6


def test_hold_and_tempo_bounds():
    a = _clip(n=40)
    for seed in range(30):
        h, _ = apply_op(a, "hold_start", CFG, np.random.default_rng(seed))
        extra = len(h["frame"]) - 40
        assert 4 <= extra <= 20 and (h["frame"][:extra + 1] == 0).all()
        t, _ = apply_op(a, "tempo", CFG, np.random.default_rng(seed))
        assert 40 / 1.5 - 2 <= len(t["frame"]) <= 40 * 1.5 + 2
        assert np.diff(t["frame"]).max() <= CFG.tempo_max + 1e-9    # local speed never above 1.5x
        assert np.diff(t["frame"]).min() >= 0


def test_repetition_only_where_the_movement_loops():
    cyc = _clip(kind="cycle")
    rng = np.random.default_rng(0)
    i, j = find_loop(cyc, CFG, rng)
    c = cyc["right_hand"][:, :, :2].mean(axis=1)
    assert j - i >= 10 and np.linalg.norm(c[j] - c[i + 1]) <= min(CFG.loop_tol, _max_step(cyc))
    for seed in range(20):
        out, changed = apply_op(cyc, "repeat_span", CFG, np.random.default_rng(seed))
        assert changed and _max_step(out) <= _max_step(cyc) + 1e-6      # splice no larger than a natural step
    line = _clip(kind="line")
    assert find_loop(line, CFG, rng) is None
    out, changed = apply_op(line, "repeat_span", CFG, rng)
    assert not changed and out is line


def test_loop_requires_the_same_hands_present():
    a = _clip(n=120, kind="cycle", cycles=3)                          # 3 cycles of 40 frames
    a["flags"][60:, GROUP_INDEX["left_hand"]] = OBSERVED | PRESENT    # a second hand appears halfway
    i, j = find_loop(a, CFG, np.random.default_rng(0))
    li = GROUP_INDEX["left_hand"]
    assert (a["flags"][i + 1, li] != 0) == (a["flags"][j, li] != 0)


def test_augmenter_probability_determinism_and_counts():
    a = _clip()
    off = TemporalAugment(TemporalAugmentConfig(p=0.0), seed=1)
    assert all(off(a) is a for _ in range(20)) and off.counts["none"] == 20
    x, y = TemporalAugment(CFG, seed=5), TemporalAugment(CFG, seed=5)
    for _ in range(50):
        assert np.array_equal(x(a)["frame"], y(a)["frame"])
    assert sum(x.counts.values()) == 50 and 10 <= x.counts["none"] <= 40
    assert x.describe()["type"] == "temporal" and x.describe()["seed"] == 5


# ---------------------------------------------------------------- training-only use
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    words = [f"w{i:02d}" for i in range(3)]
    mapping = build_label_mapping(pd.DataFrame({"word": words}), "test")
    root = tmp_path_factory.mktemp("aug_store")
    idx, arrays = make_synthetic_store(root, words, mapping, load_yaml(STEP2_CONFIG)["output"], [30],
                                       per_class={"train": 4, "val": 2, "test": 2}, seed=2)
    return idx, arrays, mapping


def test_dataset_augments_training_split_only(store):
    idx, arrays, mapping = store
    cfgs = load_yaml(STEP2_CONFIG)["configurations"]
    aug = TemporalAugment(CFG, seed=0)
    for split in ("val", "test"):
        with pytest.raises(ValueError):
            SequenceDataset(idx, arrays, split, 30, "hands", cfgs, mapping, augment=aug)
    with pytest.raises(ValueError):
        SequenceDataset(idx, arrays, "train", 30, "hands", cfgs, mapping, preload=True, augment=aug)
    plain = SequenceDataset(idx, arrays, "train", 30, "hands", cfgs, mapping)
    ident = SequenceDataset(idx, arrays, "train", 30, "hands", cfgs, mapping, augment=lambda a: a)
    for i in range(len(plain)):
        assert np.array_equal(plain[i][0].numpy(), ident[i][0].numpy())   # the hook itself changes nothing
        assert plain[i][1] == ident[i][1]
    r = plain.rows.iloc[0]
    ref = load_sequence(arrays / r.array_file, cfgs["hands"], int(r.active_start_frame), int(r.active_end_frame), 30)
    assert np.array_equal(plain[0][0].numpy(), ref)


def test_run_with_augmentation_and_without_test_inference(tmp_path, store):
    idx, arrays, mapping = store
    c4 = json.loads(json.dumps(load_yaml(PROJECT_ROOT / "configs" / "step4_training.yaml")))
    c4["latency"].update(warmup_runs=1, timed_runs=2, end_to_end_samples=2)
    aug = TemporalAugment(CFG, seed=c4["seed"])
    r = run_experiment(ExperimentSpec("lstm", "hands", 30), store="a", index=idx, arrays_dir=arrays, mapping=mapping,
                       configurations=load_yaml(STEP2_CONFIG)["configurations"], cfg=c4,
                       experiments_root=tmp_path / "exp", results_csv=tmp_path / "results.csv", max_epochs=2,
                       tag="aug", train_augment=aug, evaluate_test=False)
    d = r["exp_dir"]
    assert not any(p.name.endswith("_test.csv") or p.name.endswith("_test.json") or p.name.endswith("_test.npz")
                   for p in d.iterdir())
    assert (d / "predictions_val.csv").exists() and (d / "predictions_train.csv").exists()
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["test_evaluated"] is False and meta["train_augmentation"]["type"] == "temporal"
    assert sum(meta["train_augmentation_counts"].values()) > 0 and "test" not in meta["evaluations"]
    row = pd.read_csv(tmp_path / "results.csv").iloc[0]
    assert pd.isna(row["test_accuracy"]) and row["experiment_id"] == "hands_w30_lstm_aug"
