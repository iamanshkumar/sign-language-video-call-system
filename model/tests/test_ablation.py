"""E2 feature ablation (drop pose landmarks): columns, every split alike, preflight, no export."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.layout import feature_dim, feature_names, pose_landmark_columns
from asl.models.experiment import ExperimentSpec, export_model, preflight, run_experiment
from asl.sequences.augment import TemporalAugment, TemporalAugmentConfig
from asl.sequences.dataset import SequenceDataset, load_sequence
from asl.sequences.labels import build_label_mapping
from asl.sequences.synthetic import make_synthetic_store

C2 = load_yaml(STEP2_CONFIG)
HP = C2["configurations"]["hands_pose"]
LOWER = pose_landmark_columns(HP, range(23, 33))


def test_lower_body_columns_are_exactly_landmarks_23_to_32():
    names = feature_names(HP)
    assert len(LOWER) == 40 and feature_dim(HP) - len(LOWER) == 221
    assert LOWER == list(range(197, 227)) + list(range(251, 261))
    assert {names[c] for c in LOWER} == {f"pose_{l}_{k}" for l in range(23, 33) for k in ("x", "y", "z", "vis")}
    assert "pose_present" not in {names[c] for c in LOWER}
    with pytest.raises(ValueError):
        pose_landmark_columns(C2["configurations"]["hands"], [23])
    with pytest.raises(ValueError):
        pose_landmark_columns(HP, [33])


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    words = [f"w{i:02d}" for i in range(3)]
    mapping = build_label_mapping(pd.DataFrame({"word": words}), "test")
    root = tmp_path_factory.mktemp("abl_store")
    idx, arrays = make_synthetic_store(root, words, mapping, C2["output"], [30],
                                       per_class={"train": 4, "val": 2, "test": 2}, seed=3)
    return idx, arrays, mapping


def test_columns_removed_identically_in_every_split(store):
    idx, arrays, mapping = store
    cfgs = C2["configurations"]
    for split in ("train", "val", "test"):
        full = SequenceDataset(idx, arrays, split, 30, "hands_pose", cfgs, mapping)
        abl = SequenceDataset(idx, arrays, split, 30, "hands_pose", cfgs, mapping, drop_columns=LOWER)
        assert abl.feature_dim == 221 and full.feature_dim == 261
        for i in range(len(full)):
            assert np.array_equal(abl[i][0].numpy(), np.delete(full[i][0].numpy(), LOWER, axis=1))
            assert abl[i][1] == full[i][1]
    r = full.rows.iloc[0]
    seq = load_sequence(arrays / r.array_file, HP, int(r.active_start_frame), int(r.active_end_frame), 30,
                        drop_columns=LOWER)
    assert seq.shape == (30, 221) and seq.flags["C_CONTIGUOUS"]
    aug = SequenceDataset(idx, arrays, "train", 30, "hands_pose", cfgs, mapping,
                          augment=TemporalAugment(TemporalAugmentConfig(), 0), drop_columns=LOWER)
    assert all(aug[i][0].shape == (30, 221) for i in range(len(aug)))
    with pytest.raises(ValueError):
        SequenceDataset(idx, arrays, "val", 30, "hands_pose", cfgs, mapping, drop_columns=[261])


def test_ablation_run_preflight_metadata_and_no_export(tmp_path, store):
    idx, arrays, mapping = store
    cfgs = C2["configurations"]
    spec = ExperimentSpec("bilstm", "hands_pose", 30)
    pre = preflight(idx, arrays, spec, cfgs, mapping, drop_columns=LOWER)
    assert pre["feature_dim"] == 221
    assert preflight(idx, arrays, spec, cfgs, mapping)["feature_dim"] == 261   # default unchanged
    c4 = json.loads(json.dumps(load_yaml(PROJECT_ROOT / "configs" / "step4_training.yaml")))
    c4["latency"].update(warmup_runs=1, timed_runs=2, end_to_end_samples=2)
    r = run_experiment(spec, store="a", index=idx, arrays_dir=arrays, mapping=mapping, configurations=cfgs, cfg=c4,
                       experiments_root=tmp_path / "exp", results_csv=tmp_path / "results.csv", max_epochs=2,
                       tag="upperpose", train_augment=TemporalAugment(TemporalAugmentConfig(), c4["seed"]),
                       evaluate_test=False, drop_columns=LOWER)
    d = r["exp_dir"]
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["feature_dim"] == 221 and meta["dropped_feature_columns"] == LOWER
    assert meta["dropped_feature_names"][0] == "pose_23_x" and meta["test_evaluated"] is False
    sd = torch.load(d / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
    assert next(v for k, v in sd.items() if "weight_ih_l0" in k).shape[1] == 221
    with pytest.raises(ValueError, match="feature-ablation"):
        export_model(d, tmp_path / "export", mapping, cfgs)
    assert not (tmp_path / "export" / "best_model.pt").exists()


def test_default_is_the_full_feature_vector(store):
    idx, arrays, mapping = store
    ds = SequenceDataset(idx, arrays, "val", 30, "hands_pose", C2["configurations"], mapping)
    assert ds.drop_columns == () and ds[0][0].shape == (30, 261)
