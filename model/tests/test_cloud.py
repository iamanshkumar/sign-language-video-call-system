import urllib.request

import pandas as pd

from asl.landmarks.cloud import _HFAuth, code_fingerprint, extraction_list, install_hf_auth, shard


def _frozen(n=50):
    return pd.DataFrame({"sample_id": [f"p/{i:03d}.mp4" for i in range(n)],
                         "split": ["train", "val", "test", "val", "train"] * (n // 5), "word": "w"})


def test_extraction_list_only_requested_splits():
    df = extraction_list(_frozen().sample(frac=1, random_state=0), ["val", "test"])
    assert set(df["split"]) == {"val", "test"} and len(df) == 30 and df["sample_id"].is_monotonic_increasing
    lim = extraction_list(_frozen(), ["val", "test"], limit_per_split=3)
    assert lim["split"].value_counts().to_dict() == {"val": 3, "test": 3}


def test_shards_are_deterministic_disjoint_and_complete():
    df = extraction_list(_frozen(), ["val", "test"])
    a, b = shard(df, 4), shard(df.sample(frac=1, random_state=1), 4)
    assert [x["sample_id"].tolist() for x in a] == [x["sample_id"].tolist() for x in b]
    ids = [s for part in a for s in part["sample_id"]]
    assert sorted(ids) == sorted(df["sample_id"]) and len(ids) == len(set(ids))
    assert max(map(len, a)) - min(map(len, a)) <= 1


def test_fingerprint_detects_changes(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "c.yaml").write_text("k: 1\n")
    f1 = code_fingerprint(tmp_path, ["pkg", "c.yaml"])
    assert set(f1) == {"pkg/a.py", "c.yaml"}
    (tmp_path / "c.yaml").write_text("k: 2\n")
    assert code_fingerprint(tmp_path, ["pkg", "c.yaml"])["c.yaml"] != f1["c.yaml"]


def test_hf_token_only_sent_to_huggingface():
    h = _HFAuth("secret")
    hf = h.https_request(urllib.request.Request("https://huggingface.co/datasets/x/resolve/r/a.mp4"))
    cdn = h.https_request(urllib.request.Request("https://cdn-lfs.hf.co/abc"))
    other = h.https_request(urllib.request.Request("https://example.com/"))
    assert hf.unredirected_hdrs.get("Authorization") == "Bearer secret"
    assert "Authorization" not in cdn.unredirected_hdrs and "Authorization" not in other.unredirected_hdrs
    assert "Authorization" not in hf.headers  # never copied onto redirected requests
    assert install_hf_auth(None) is False
