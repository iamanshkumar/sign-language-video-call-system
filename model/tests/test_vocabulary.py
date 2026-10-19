"""Step 5: subset vocabulary rules on a tiny hand-built split (no dataset needed)."""
import json

import pandas as pd
import pytest

from asl.config import PROJECT_ROOT, load_yaml
from asl.data.vocabulary import (add_file_columns, class_features, components, exclusion_reasons, label_pairs,
                                 rank_candidates)

RULES = load_yaml(PROJECT_ROOT / "configs" / "step5_vocabulary.yaml")["rules"]


def _row(word, i, split, sha=None, proxy="numericid-WORD", aug=None, name=None):
    name = name or f"{i}-{word.upper()}.mp4"
    return {"sample_id": f"p/{name}", "repo_path": f"p/{name}", "word": word, "sha256": sha or f"{word}{i}",
            "split_group": sha or f"{word}{i}", "aug_source": aug, "source_proxy": proxy, "split": split}


def _clean(word, n_train=20, n_val=6, n_test=6, start=0):
    rows, i = [], start
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(n):
            rows.append(_row(word, i, split, proxy="numericid-WORD" if i % 2 else "word_n"))
            i += 1
    return rows


@pytest.fixture()
def feats():
    rows = _clean("good") + _clean("thin", n_val=4) + _clean("aa") + _clean("bb") + _clean("two words")
    rows += [_row("aa", 100, "train", sha="X"), _row("bb", 100, "train", sha="X")]       # shared clip
    rows += [_row("aug", i, "train", aug="src.mp4") for i in range(20)] + _clean("aug", 2, 6, 6, start=50)
    rows += [_row("batchy", i, s, proxy="word_timestamp_n", name=f"batchy_20241119_172637_{i}.mp4")
             for i, s in enumerate(["train"] * 14 + ["val"] * 6 + ["test"] * 6)] + _clean("batchy", 2, 0, 0, 90)
    s = add_file_columns(pd.DataFrame(rows))
    f = class_features(s, pd.DataFrame({"label": ["x"], "anomaly": ["y"]}))
    f["reasons"] = exclusion_reasons(f, RULES, 0.5)
    return s, f


def test_each_rule_fires_with_its_reason(feats):
    _, f = feats
    r = f["reasons"]
    assert r["good"] == ""
    assert "too_few_val_units" in r["thin"]
    assert "shares_clips_with_other_labels" in r["aa"] and "shares_clips_with_other_labels" in r["bb"]
    assert "multiword_label" in r["two words"]
    assert "augmentation_dominated" in r["aug"]
    assert "single_batch_dominated" in r["batchy"]


def test_counts_and_fractions(feats):
    _, f = feats
    assert f.loc["aa", "cross_label_clips"] == 1 and f.loc["aa", "clips"] == 33
    assert f.loc["good", ["train_units", "val_units", "test_units"]].tolist() == [20, 6, 6]
    assert f.loc["aug", "augmented_fraction"] == pytest.approx(20 / 34)


def test_pairs_and_components(feats):
    s, f = feats
    p = label_pairs(s, f["clips"])
    assert p[["label_a", "label_b", "shared_clips"]].values.tolist() == [["aa", "bb", 1]]
    assert components(p) == [{"aa", "bb"}]


def test_ranking_is_deterministic_and_ordered(feats):
    _, f = feats
    a = rank_candidates(f, ["min_eval_units", "source_patterns", "units"], 42)
    b = rank_candidates(f.sample(frac=1, random_state=3), ["min_eval_units", "source_patterns", "units"], 42)
    assert list(a.index) == list(b.index)
    assert (a["min_eval_units"].diff().dropna() <= 0).all()


def test_committed_vocabularies_are_consistent():
    dev = json.loads((PROJECT_ROOT / "configs" / "vocabulary_100.json").read_text())
    exp = json.loads((PROJECT_ROOT / "configs" / "vocabulary_expanded.json").read_text())
    d, e = [c["class"] for c in dev["classes"]], [c["class"] for c in exp["classes"]]
    assert len(d) == 100 and len(e) == 300 and set(d) <= set(e)
    assert d == sorted(d) and [c["index"] for c in dev["classes"]] == list(range(100))
    for c in exp["classes"]:
        assert c["cross_label_shared_fraction"] == 0 and c["augmented_fraction"] == 0
        assert c["val_units"] >= 6 and c["test_units"] >= 6 and c["dataset_label"] == c["class"]


# ---------------------------------------------------------------- Step 5 extraction helpers
def test_select_samples_subset_is_deterministic_and_keeps_frozen_split():
    from asl.data.vocabulary import select_samples

    rows = []
    for w in ("aa", "bb", "cc"):
        rows += _clean(w)
    frozen = pd.DataFrame(rows)
    a = select_samples(frozen, ["aa", "bb", "cc"], {"train": 4, "val": 1, "test": 1}, n_classes=2, seed=42)
    b = select_samples(frozen.sample(frac=1, random_state=0), ["aa", "bb", "cc"], {"train": 4, "val": 1, "test": 1},
                       n_classes=2, seed=42)
    assert a["sample_id"].tolist() == b["sample_id"].tolist()
    assert a["word"].nunique() == 2 and a.groupby("word")["split"].value_counts().unstack()[["train", "val", "test"]].values.tolist() == [[4, 1, 1]] * 2
    merged = a.merge(frozen, on="sample_id", suffixes=("", "_f"))
    assert (merged["split"] == merged["split_f"]).all()
    with pytest.raises(ValueError):
        select_samples(frozen, ["aa", "zz"])


def test_subset_label_mapping_matches_vocabulary_index():
    from asl.data.vocabulary import load_vocabulary, subset_label_mapping
    from asl.data.split import load_frozen_split
    from asl.config import load_config

    c1 = load_config()
    lock = json.loads((c1.path("splits") / "split_v1.lock.json").read_text())
    v = load_vocabulary(PROJECT_ROOT / "configs" / "vocabulary_100.json", lock["sha256"])
    m = subset_label_mapping(load_frozen_split(c1.path("splits"), "v1"), v, lock["sha256"])
    assert m["num_classes"] == 100
    assert all(m["word_to_index"][c["class"]] == c["index"] for c in v["classes"])
    with pytest.raises(ValueError):
        load_vocabulary(PROJECT_ROOT / "configs" / "vocabulary_100.json", "0" * 64)


def test_storage_profile_drops_face_only():
    from asl.config import STEP2_CONFIG
    from asl.landmarks.layout import feature_dim
    from asl.landmarks.storage import apply_profile

    c2 = load_yaml(STEP2_CONFIG)
    p = apply_profile(c2, "hands_pose_f32")
    assert p["output"]["groups_to_store"] == ["pose", "left_hand", "right_hand"]
    assert set(p["output"]["coord_dtype"].values()) == {"float32"}
    assert {k: feature_dim(v) for k, v in p["configurations"].items()} == {"hands": 128, "hands_pose": 261}
    for k in ("mediapipe", "preprocessing", "rejection"):  # extraction and preprocessing are unchanged
        assert p[k] == c2[k]
    assert "face" in c2["output"]["groups_to_store"]  # the original config is not modified
    with pytest.raises(ValueError):
        apply_profile(c2, "nope")


def test_hands_pose_profile_roundtrip(tmp_path):
    """A synthetic clip processed with the profile stores no face, float32, and F = 261."""
    import numpy as np
    from asl.config import STEP2_CONFIG
    from asl.landmarks.layout import assemble_features
    from asl.landmarks.process import pack_arrays, save_arrays
    from asl.landmarks.storage import apply_profile

    p = apply_profile(load_yaml(STEP2_CONFIG), "hands_pose_f32")
    t = 12
    norm = np.random.default_rng(0).normal(size=(t, 543, 3))
    on = np.ones((t, 4), bool)
    a = pack_arrays(norm, on, on, np.zeros((t, 4), bool), np.ones((t, 33)), np.zeros((t, 3)), np.ones(t),
                    np.ones(t, bool), None, p["output"])
    save_arrays(a, tmp_path / "x.npz")
    with np.load(tmp_path / "x.npz") as z:
        assert "face" not in z.files and all(z[g].dtype == np.float32 for g in ("pose", "left_hand", "right_hand"))
        assert assemble_features(z, p["configurations"]["hands_pose"]).shape == (t, 261)
