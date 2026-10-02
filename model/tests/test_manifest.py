import pandas as pd

from asl.data import manifest as mf
from asl.data.labels import audit_labels


def _repo(rows):
    return pd.DataFrame(rows, columns=["repo_path", "folder", "filename", "size_bytes", "sha256"])


def test_reconciliation_flags_every_problem(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("word,videos\nnull,a.mp4\nh,h.mp4\nh,h.mp4\nb,missing.mp4\nx,c.mp4\ny,c.mp4\nz,amb.mp4\n")
    repo = _repo([
        ("p1/a.mp4", "p1", "a.mp4", 10, "s1"), ("p1/h.mp4", "p1", "h.mp4", 10, "s2"),
        ("p1/c.mp4", "p1", "c.mp4", 10, "s3"), ("p1/amb.mp4", "p1", "amb.mp4", 10, "s4"),
        ("p2/amb.mp4", "p2", "amb.mp4", 10, "s5"), ("p2/orphan.mp4", "p2", "orphan.mp4", 10, "s6"),
    ])
    df = mf.read_metadata_csv(csv)
    assert df.loc[0, "word"] == "null"  # not coerced to NaN
    m = mf.build_manifest(df, repo).set_index("filename")
    assert m.loc["a.mp4", "issue"] == "" and m.loc["a.mp4", "repo_path"] == "p1/a.mp4"
    assert sorted(m.loc["h.mp4", "issue"]) == ["", mf.DUPLICATE_CSV_ROW]
    assert m.loc["missing.mp4", "issue"] == mf.FILE_MISSING
    assert set(m.loc["c.mp4", "issue"]) == {mf.CONFLICTING_LABEL}
    assert m.loc["amb.mp4", "issue"] == mf.AMBIGUOUS_FILENAME
    assert m.loc["orphan.mp4", "issue"] == mf.NOT_IN_CSV
    assert m["eligible_pre_probe"].sum() == 2
    assert m.loc[m["eligible_pre_probe"], "sample_id"].is_unique


def test_duplicate_groups_and_cross_label():
    m = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"], "repo_path": ["a", "b", "c", "d"],
        "word": ["lot", "heap", "cat", "cat"], "sha256": ["x", "x", "y", "z"],
        "eligible_pre_probe": [True] * 4,
    })
    g = mf.duplicate_content_groups(m)
    assert len(g) == 1 and g.loc[0, "labels"] == "heap|lot" and g.loc[0, "n_labels"] == 2
    a = mf.annotate_duplicates(m).set_index("sample_id")
    assert a.loc["a", "dup_cross_label"] and not a.loc["c", "dup_cross_label"]


def test_source_proxy():
    assert mf.source_proxy("0001-HELLO.mp4") == "numericid-WORD"
    assert mf.source_proxy("hold_20241119_172645_16.mp4") == "word_timestamp_n"
    assert mf.source_proxy("squeeze_video_2.mp4") == "word_video_n"
    assert mf.source_proxy("create_5.mp4") == "word_n"
    assert mf.source_proxy("A.mp4") == "word_only"


def test_label_audit_reports_without_changing():
    words = pd.Series(["thank you", "thankyou", "memorizememorize", "walk-cl", "thank", "you"])
    a = audit_labels(words)
    kinds = dict(zip(a["label"] + "/" + a["anomaly"], a["detail"]))
    assert "thankyou/near_duplicate_label" in kinds
    assert "memorizememorize/repeated_word" in kinds
    assert "walk-cl/classifier_suffix" in kinds
    assert set(a["label"]) <= set(words)
