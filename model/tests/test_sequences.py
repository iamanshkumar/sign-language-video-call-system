"""Step 3 tests on small synthetic landmark stores (no dataset, no MediaPipe)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.layout import GROUP_INDEX, N_LANDMARKS, PRESENT, SLICES, feature_dim
from asl.landmarks.preprocess import ActiveInterval
from asl.landmarks.process import array_name, pack_arrays, save_arrays
from asl.sequences.dataset import SequenceDataset, load_sequence, make_dataloaders
from asl.sequences.index import build_index, check_index_leakage, interval_bin
from asl.sequences.labels import build_label_mapping, encode, load_label_mapping, save_label_mapping
from asl.sequences.resample import positions, resample_interval

CFG2 = load_yaml(STEP2_CONFIG)
CFG3 = load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml")
CONFIGS = CFG2["configurations"]
WINDOWS = CFG3["sequences"]["windows"]
F = {"hands": 128, "hands_pose": 261, "hands_pose_face": 1666}


# ---------------------------------------------------------------- synthetic store
def _arrays(t, start, end, out_cfg=None, absent=None):
    """Synthetic v2 arrays: every coordinate x = frame index, y = landmark index / 1000.

    absent: {group: [frames]} where that group is not present (zero-filled)."""
    norm = np.zeros((t, N_LANDMARKS, 3))
    norm[:, :, 0] = np.arange(t)[:, None]
    norm[:, :, 1] = np.arange(N_LANDMARKS)[None, :] / 1000
    present = np.ones((t, 4), bool)
    for g, frames in (absent or {}).items():
        present[frames, GROUP_INDEX[g]] = False
        norm[np.ix_(frames, range(SLICES[g].start, SLICES[g].stop))] = 0
    vis = np.tile(np.linspace(0.5, 1.0, t)[:, None], (1, 33))
    return pack_arrays(norm, present, present.copy(), np.zeros_like(present), vis, np.zeros((t, 3)), np.ones(t),
                       np.ones(t, bool), ActiveInterval(start, end), out_cfg or CFG2["output"])


SPECS = [  # sample_id, word, split, T, start, end, sha, split_group, aug_source, rejected
    ("p/a1.mp4", "apple", "train", 90, 10, 79, "h1", "g1", "", None),     # L = 70
    ("p/a2.mp4", "apple", "train", 80, 5, 54, "h2", "g2", "", None),      # L = 50
    ("p/b1.mp4", "ball", "train", 70, 20, 39, "h3", "g3", "", None),      # L = 20 (short)
    ("p/b1_ball_aug_0.mp4", "ball", "train", 70, 20, 39, "h4", "g3", "b1.mp4", None),
    ("p/c1.mp4", "cat", "train", 60, 0, 59, "h5", "g5", "", None),        # L = 60
    ("p/a3.mp4", "apple", "val", 75, 10, 49, "h6", "g6", "", None),       # L = 40
    ("p/b2.mp4", "ball", "val", 60, 10, 14, "h7", "g7", "", None),        # L = 5 (short)
    ("p/c2.mp4", "cat", "test", 70, 5, 49, "h8", "g8", "", None),         # L = 45
    ("p/a4.mp4", "apple", "test", 50, 10, 39, "h9", "g9", "", None),      # L = 30
    ("p/x1.mp4", "cat", "test", 40, 0, 0, "h10", "g10", "", "active_both_hands_missing_run"),  # rejected
]


def _frozen():
    rows = [{"sample_id": s, "repo_path": s, "word": w, "split": sp, "sha256": h, "dup_group_size": "1",
             "split_group": g, "aug_source": aug, "source_proxy": "x"} for s, w, sp, _, _, _, h, g, aug, _ in SPECS]
    rows.append({"sample_id": "p/z1.mp4", "repo_path": "p/z1.mp4", "word": "zebra", "split": "train", "sha256": "h99",
                 "dup_group_size": "1", "split_group": "g99", "aug_source": "", "source_proxy": "x"})  # never extracted
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path):
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    meta = []
    for s, w, sp, t, a, b, _, _, _, rej in SPECS:
        rec = {"sample_id": s, "repo_path": s, "word": w, "split": sp, "rejected": rej is not None,
               "rejection_reason": rej, "array_file": None, "active_start_frame": a, "active_end_frame": b,
               "std_frame_count": t}
        if rej is None:
            name = array_name(s)
            save_arrays(_arrays(t, a, b), arrays_dir / name)
            rec["array_file"] = name
        meta.append(rec)
    frozen = _frozen()
    mapping = build_label_mapping(frozen, "fake-split-sha")
    idx, excluded = build_index(pd.DataFrame(meta), frozen, mapping, arrays_dir, WINDOWS, 30)
    return {"dir": arrays_dir, "meta": pd.DataFrame(meta), "frozen": frozen, "mapping": mapping, "idx": idx,
            "excluded": excluded}


def _ds(store, split, window=30, cfg="hands"):
    return SequenceDataset(store["idx"], store["dir"], split, window, cfg, CONFIGS, store["mapping"])


# ---------------------------------------------------------------- 1-3, 17: shapes for every window / configuration
@pytest.mark.parametrize("window", [30, 45, 60])
@pytest.mark.parametrize("cfg", ["hands", "hands_pose", "hands_pose_face"])
def test_output_shape_every_window_and_configuration(store, window, cfg):
    for split in ("train", "val", "test"):
        ds = _ds(store, split, window, cfg)
        for i in range(len(ds)):
            x, y = ds[i]
            assert tuple(x.shape) == (window, F[cfg]) == (window, feature_dim(CONFIGS[cfg]))


# ---------------------------------------------------------------- 4, 5, 7: resampling
def test_resampling_is_deterministic(store):
    r = store["idx"].iloc[0]
    a = load_sequence(store["dir"] / r.array_file, CONFIGS["hands_pose_face"], r.active_start_frame, r.active_end_frame, 45)
    b = load_sequence(store["dir"] / r.array_file, CONFIGS["hands_pose_face"], r.active_start_frame, r.active_end_frame, 45)
    assert np.array_equal(a, b)


def test_temporal_order_preserved_and_endpoints_included():
    arr = _arrays(90, 10, 79)
    for w in (30, 45, 60):
        out = resample_interval(arr, ["left_hand"], 10, 79, w)
        x = out["left_hand"][:, 0, 0]
        assert np.all(np.diff(x) > 0)                     # strictly increasing time
        assert x[0] == 10 and x[-1] == 79                 # first and last active frame
        assert np.allclose(x, positions(10, 79, w), atol=0.05)  # float16 storage of frame index


def test_boundary_cases_identity_and_bins():
    arr = _arrays(70, 5, 34)  # L = 30
    out = resample_interval(arr, ["pose"], 5, 34, 30)
    assert np.array_equal(out["pose"], arr["pose"][5:35])  # L == W reproduces stored frames exactly
    assert [interval_bin(n) for n in (29, 30, 44, 45, 59, 60)] == ["<30", "30-44", "30-44", "45-59", "45-59", ">=60"]
    assert positions(0, 59, 60).tolist() == list(range(60))
    with pytest.raises(ValueError):
        positions(5, 4, 30)


# ---------------------------------------------------------------- 6: short active intervals
def test_short_interval_is_upsampled_by_interpolation_not_duplication():
    arr = _arrays(60, 10, 14)  # L = 5
    out = resample_interval(arr, ["left_hand", "pose"], 10, 14, 60)
    x = out["left_hand"][:, 0, 0]
    assert len(x) == 60 and x[0] == 10 and x[-1] == 14
    assert len(np.unique(x)) == 60          # no output frame is a copy of another
    assert np.allclose(x, np.linspace(10, 14, 60), atol=1e-6)


def test_short_intervals_are_kept_and_reported(store):
    idx = store["idx"]
    short = idx[idx["short_active_interval"]]
    assert set(short["sample_id"]) == {"p/b1.mp4", "p/b1_ball_aug_0.mp4", "p/b2.mp4"}
    assert (short["stretch_60"] > 1).all() and idx.loc[idx.sample_id == "p/b2.mp4", "stretch_60"].item() == 12
    assert len(_ds(store, "val", 60)) == 2  # the 5-frame clip is still a sample


# ---------------------------------------------------------------- 8-10: labels
def test_label_mapping_reproducible_and_shared(tmp_path, store):
    frozen = store["frozen"]
    m1 = build_label_mapping(frozen, "s")
    m2 = build_label_mapping(frozen.sample(frac=1, random_state=3), "s")
    assert m1 == m2 and m1["word_to_index"] == {"apple": 0, "ball": 1, "cat": 2, "zebra": 3}
    assert m1["num_classes"] == 4  # 'zebra' has no extracted sample but keeps its index
    path = tmp_path / "m.json"
    save_label_mapping(m1, path)
    assert load_label_mapping(path) == m1
    save_label_mapping(m1, path)  # identical: allowed
    with pytest.raises(FileExistsError):
        save_label_mapping(build_label_mapping(frozen[frozen.word != "zebra"], "s"), path)
    bad = json.loads(path.read_text())
    bad["word_to_index"]["apple"], bad["word_to_index"]["ball"] = 1, 0
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_label_mapping(path)


def test_same_word_same_integer_in_every_split(store):
    seen = {}
    for split in ("train", "val", "test"):
        ds = _ds(store, split)
        for i in range(len(ds)):
            m = ds.metadata(i)
            seen.setdefault(m["word"], set()).add(int(ds[i][1]))
            assert int(ds[i][1]) == store["mapping"]["word_to_index"][m["word"]]
    assert all(len(v) == 1 for v in seen.values())


def test_unknown_label_is_rejected(store):
    with pytest.raises(KeyError):
        encode(pd.Series(["apple", "dragon"]), store["mapping"])


# ---------------------------------------------------------------- 11-12: Dataset / DataLoader
def test_dataset_item_types_and_metadata(store):
    ds = _ds(store, "train", 45, "hands_pose")
    x, y = ds[0]
    assert x.dtype == torch.float32 and y.dtype == torch.long and y.ndim == 0
    m = ds.metadata(0)
    r = store["idx"].set_index("sample_id").loc[m["sample_id"]]
    assert (m["split"], m["word"], m["window"], m["configuration"]) == ("train", r.word, 45, "hands_pose")
    assert m["array_file"] == r.array_file and m["active_interval"] == (r.active_start_frame, r.active_end_frame)


def test_dataloader_batching_and_determinism(store):
    dsets = {s: _ds(store, s, 30, "hands") for s in ("train", "val", "test")}
    loaders = make_dataloaders(dsets, batch_size=2, num_workers=0, seed=42)
    batches = list(loaders["train"])
    assert [tuple(b[0].shape) for b in batches] == [(2, 30, 128), (2, 30, 128), (1, 30, 128)]
    again = [b[1].tolist() for b in make_dataloaders(dsets, 2, 0, 42)["train"]]
    assert [b[1].tolist() for b in batches] == again        # seeded shuffle reproducible
    val = torch.cat([b[1] for b in loaders["val"]]).tolist()
    assert val == [int(dsets["val"][i][1]) for i in range(len(dsets["val"]))]  # val in fixed order
    assert loaders["val"].sampler.__class__.__name__ == "SequentialSampler"
    assert loaders["test"].sampler.__class__.__name__ == "SequentialSampler"


# ---------------------------------------------------------------- 13-14, leakage
def test_split_inherited_from_frozen_split(store):
    idx = store["idx"].set_index("sample_id")
    frozen = store["frozen"].set_index("sample_id")
    assert (idx["split"] == frozen.loc[idx.index, "split"]).all()
    bad = store["meta"].copy()
    bad.loc[bad.sample_id == "p/a3.mp4", "split"] = "train"  # metadata disagrees with the frozen split
    with pytest.raises(ValueError):
        build_index(bad, store["frozen"], store["mapping"], store["dir"], WINDOWS, 30)


def test_no_source_video_in_more_than_one_split_across_windows(store):
    owner = {}
    for w in WINDOWS:
        for split in ("train", "val", "test"):
            ds = _ds(store, split, w)
            for i in range(len(ds)):
                owner.setdefault(ds.metadata(i)["sample_id"], set()).add(split)
    assert all(len(s) == 1 for s in owner.values())
    assert len(owner) == len(store["idx"])


def test_leakage_checks_catch_violations(store):
    idx = store["idx"]
    assert check_index_leakage(idx) == []
    dup = idx.copy()
    dup.loc[dup.sample_id == "p/a3.mp4", "sha256"] = "h1"  # same content as a train video
    assert any("SHA-256" in p for p in check_index_leakage(dup))
    aug = idx.copy()
    aug.loc[aug.sample_id == "p/b1_ball_aug_0.mp4", ["split", "split_group"]] = ["val", "g_other"]
    assert any("augmented" in p for p in check_index_leakage(aug))
    grp = idx.copy()
    grp.loc[grp.sample_id == "p/c2.mp4", "split_group"] = "g1"
    assert any("split group" in p for p in check_index_leakage(grp))
    twice = pd.concat([idx, idx.iloc[:1]])
    assert any("more than once" in p for p in check_index_leakage(twice))


def test_no_identical_sequence_across_splits(store):
    split_of = {}
    for split in ("train", "val", "test"):
        ds = _ds(store, split, 30, "hands_pose_face")
        for i in range(len(ds)):
            split_of.setdefault(ds[i][0].numpy().tobytes(), set()).add(split)
    assert all(len(s) == 1 for s in split_of.values())


# ---------------------------------------------------------------- 15-16: precision and presence
def test_mixed_precision_input_gives_float32_close_to_float32_store():
    f32 = {**CFG2["output"], "coord_dtype": "float32"}
    a16, a32 = _arrays(80, 5, 54), _arrays(80, 5, 54, out_cfg=f32)
    assert a16["left_hand"].dtype == np.float16 and a16["pose"].dtype == np.float32
    groups = CONFIGS["hands_pose_face"]
    o16, o32 = resample_interval(a16, groups, 5, 54, 45), resample_interval(a32, groups, 5, 54, 45)
    for g in groups:
        assert o16[g].dtype == np.float32
        assert np.abs(o16[g] - o32[g]).max() <= np.abs(o32[g]).max() * 2**-10
    assert np.array_equal(o16["pose"], o32["pose"])  # pose is float32 in both


def test_presence_mask_preserved():
    arr = _arrays(60, 0, 59, absent={"right_hand": list(range(20, 30))})
    out = resample_interval(arr, ["left_hand", "right_hand"], 0, 59, 60)  # identity grid
    rh = (out["flags"][:, GROUP_INDEX["right_hand"]] & PRESENT) != 0
    assert not rh[20:30].any() and rh[:20].all() and rh[30:].all()
    assert (out["right_hand"][20:30] == 0).all()
    out30 = resample_interval(arr, ["left_hand", "right_hand"], 0, 59, 30)  # positions between frames
    rh30 = (out30["flags"][:, GROUP_INDEX["right_hand"]] & PRESENT) != 0
    p = positions(0, 59, 30)
    for k, pk in enumerate(p):
        i, j = int(np.floor(pk)), min(int(np.floor(pk)) + 1, 59)
        if 20 <= i <= 29 and 20 <= j <= 29:
            assert not rh30[k] and (out30["right_hand"][k] == 0).all()   # inside the gap: absent, zero
        if not (20 <= i <= 29 or 20 <= j <= 29):
            assert rh30[k]                                                # away from the gap: present
    # the presence column of the assembled features mirrors the resampled flags
    from asl.landmarks.layout import assemble_features
    f = assemble_features(out30, ["left_hand", "right_hand"])
    assert np.array_equal(f[:, 127] == 1, rh30) and f[:, 63].all()  # right-hand / left-hand presence columns


# ---------------------------------------------------------------- 18-20
def test_rejected_samples_are_not_included(store):
    assert "p/x1.mp4" not in set(store["idx"]["sample_id"])
    assert store["excluded"].set_index("sample_id").loc["p/x1.mp4", "rejection_reason"] == "active_both_hands_missing_run"
    assert len(_ds(store, "test")) == 2


def test_metadata_traceable_to_original_video(store):
    for split in ("train", "val", "test"):
        ds = _ds(store, split, 60, "hands_pose_face")
        for i in range(len(ds)):
            m = ds.metadata(i)
            spec = next(s for s in SPECS if s[0] == m["sample_id"])
            assert (m["word"], m["split"], m["active_interval"]) == (spec[1], spec[2], (spec[4], spec[5]))
            assert m["array_file"] == array_name(m["sample_id"])


def test_regeneration_produces_identical_results(store):
    idx2, exc2 = build_index(store["meta"], store["frozen"], store["mapping"], store["dir"], WINDOWS, 30)
    pd.testing.assert_frame_equal(idx2, store["idx"])
    pd.testing.assert_frame_equal(exc2, store["excluded"])
    a = [x.numpy().tobytes() for x, _ in _ds(store, "train", 60, "hands_pose_face")]
    b = [x.numpy().tobytes() for x, _ in _ds(store, "train", 60, "hands_pose_face")]
    assert a == b


def test_preload_matches_lazy(store):
    lazy = _ds(store, "train", 45, "hands_pose")
    pre = SequenceDataset(store["idx"], store["dir"], "train", 45, "hands_pose", CONFIGS, store["mapping"], preload=True)
    assert all(torch.equal(lazy[i][0], pre[i][0]) for i in range(len(lazy)))
