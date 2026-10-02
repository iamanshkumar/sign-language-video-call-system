import os
import numpy as np
import pandas as pd
import pytest

from asl.data.leakage import run_checks
from asl.data.split import assign_splits, load_frozen_split, write_frozen_split


def _samples(n_classes=20, per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        for i in range(per_class):
            rows.append({"sample_id": f"c{c}_{i}", "repo_path": f"p/c{c}_{i}.mp4", "word": f"w{c}",
                         "sha256": f"h{c}_{i}", "source_proxy": "x"})
    df = pd.DataFrame(rows)
    # byte-identical duplicates, some across labels
    df.loc[df.index[:30:3], "sha256"] = "dupA"
    df.loc[df.index[100:110], "sha256"] = "dupB"
    return df.sample(frac=1, random_state=int(rng.integers(1000)))


RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_split_is_deterministic_and_order_independent():
    a = assign_splits(_samples(seed=1), seed=42, val=0.15, test=0.15).set_index("sample_id")["split"]
    b = assign_splits(_samples(seed=2), seed=42, val=0.15, test=0.15).set_index("sample_id")["split"]
    assert a.sort_index().equals(b.sort_index())
    c = assign_splits(_samples(seed=1), seed=7, val=0.15, test=0.15).set_index("sample_id")["split"]
    assert not a.sort_index().equals(c.sort_index())


def test_split_has_no_leakage():
    s = _samples()
    out = assign_splits(s, seed=42, val=0.15, test=0.15)
    checks = run_checks(out, set(s["sample_id"]), RATIOS, tolerance=0.02)
    failed = [c for c in checks if not c[1]]
    assert not failed, failed
    assert out.groupby("sha256")["split"].nunique().max() == 1


def test_leakage_check_catches_violations():
    s = _samples()
    out = assign_splits(s, seed=42, val=0.15, test=0.15)
    bad = out.copy()
    i = bad.index[bad["sha256"] == "dupB"][0]
    bad.loc[i, "split"] = "test" if bad.loc[i, "split"] != "test" else "train"
    names = {c for c, ok, _ in run_checks(bad, set(s["sample_id"]), RATIOS, 0.02) if not ok}
    assert "no identical content (SHA-256) across splits" in names


def test_frozen_split_cannot_be_overwritten_or_tampered(tmp_path):
    out = assign_splits(_samples(), seed=42, val=0.15, test=0.15)
    path = write_frozen_split(out, tmp_path, "v1", {"seed": 42})
    assert len(load_frozen_split(tmp_path, "v1")) == len(out)
    with pytest.raises(FileExistsError):
        write_frozen_split(out, tmp_path, "v1", {"seed": 42})
    os.chmod(path, 0o644)
    path.write_text(path.read_text().replace(",train", ",test", 1))
    with pytest.raises(RuntimeError):
        load_frozen_split(tmp_path, "v1")


def test_cross_label_groups_stay_together_and_rare_class_is_covered():
    # "rare" has 6 files: 4 of them are shared (byte-identical) with "common" in 2 groups,
    # so only 4 distinct content units contain "rare".
    rows = [{"sample_id": f"c{i}", "repo_path": f"c{i}", "word": "common", "sha256": f"u{i}", "source_proxy": "x"}
            for i in range(60)]
    for g in range(2):
        for w in ("common", "rare"):
            rows.append({"sample_id": f"g{g}{w}", "repo_path": f"g{g}{w}", "word": w, "sha256": f"G{g}",
                         "source_proxy": "x"})
    rows += [{"sample_id": f"r{i}", "repo_path": f"r{i}", "word": "rare", "sha256": f"r{i}", "source_proxy": "x"}
             for i in range(2)]
    df = pd.DataFrame(rows)
    out = assign_splits(df, seed=42, val=0.15, test=0.15)
    assert out.groupby("sha256")["split"].nunique().max() == 1
    assert set(out.loc[out["word"] == "rare", "split"]) == {"train", "val", "test"}
    assert (out.loc[out["sha256"] == "G0", "dup_group_size"] == 2).all()


def test_expected_classes():
    from asl.data.split import expected_classes
    df = pd.DataFrame({"word": ["a", "a", "a", "b", "b", "b"], "sha256": ["1", "2", "3", "4", "4", "5"]})
    assert expected_classes(df) == {"a"}


def test_augmentation_families_are_grouped_with_source():
    from asl.data.groups import augmentation_source, build_split_groups
    assert augmentation_source("a_6_a_aug_0.mp4", "a") == "a_6.mp4"
    assert augmentation_source("A • ASL Dictionary_a_aug_5.mp4", "a") == "A • ASL Dictionary.mp4"
    assert augmentation_source("laugh_3.mp4", "laugh") is None
    df = pd.DataFrame({
        "sample_id": ["p/a_6.mp4", "p/a_6_a_aug_0.mp4", "p/a_6_a_aug_1.mp4", "p/x_1_a_aug_0.mp4",
                      "p/x_1_a_aug_1.mp4", "p/b.mp4", "p/b_copy.mp4", "p/c.mp4"],
        "filename": ["a_6.mp4", "a_6_a_aug_0.mp4", "a_6_a_aug_1.mp4", "x_1_a_aug_0.mp4", "x_1_a_aug_1.mp4",
                     "b.mp4", "b_copy.mp4", "c.mp4"],
        "word": ["a", "a", "a", "a", "a", "b", "bb", "c"],
        "sha256": ["1", "2", "3", "4", "5", "6", "6", "3"],
    })
    g = build_split_groups(df).set_index("sample_id")["split_group"]
    # source + its copies, and c.mp4 via identical bytes with an aug copy (transitive)
    assert g["p/a_6.mp4"] == g["p/a_6_a_aug_0.mp4"] == g["p/a_6_a_aug_1.mp4"] == g["p/c.mp4"]
    assert g["p/x_1_a_aug_0.mp4"] == g["p/x_1_a_aug_1.mp4"] != g["p/a_6.mp4"]  # source absent, copies linked
    assert g["p/b.mp4"] == g["p/b_copy.mp4"]


def test_split_keeps_augmentation_family_together():
    from asl.data.groups import build_split_groups
    rows = []
    for c in range(5):
        for i in range(8):
            rows.append({"sample_id": f"c{c}_{i}.mp4", "filename": f"c{c}_{i}.mp4", "word": f"w{c}",
                         "sha256": f"h{c}_{i}", "repo_path": f"c{c}_{i}", "source_proxy": "x"})
            for j in range(3):
                n = f"c{c}_{i}_w{c}_aug_{j}.mp4"
                rows.append({"sample_id": n, "filename": n, "word": f"w{c}", "sha256": f"h{n}",
                             "repo_path": n, "source_proxy": "x"})
    df = build_split_groups(pd.DataFrame(rows))
    out = assign_splits(df, seed=42, val=0.15, test=0.15)
    assert out.groupby("split_group")["split"].nunique().max() == 1
    assert out["split_group"].nunique() == 40
