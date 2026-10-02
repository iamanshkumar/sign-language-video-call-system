import av
import numpy as np
import pandas as pd

from asl.data.probe import probe_video
from asl.data.validate import validate


def _make_video(path, n_frames, fps=30):
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(n_frames):
            img = np.full((48, 64, 3), i % 255, dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_probe_counts_decoded_frames(tmp_path):
    f = tmp_path / "v.mp4"
    _make_video(f, 45, fps=30)
    r = probe_video(str(f))
    assert r["decode_ok"] and r["decoded_frames"] == 45
    assert (r["width"], r["height"]) == (64, 48)
    assert abs(r["decoded_fps"] - 30) < 0.5


def test_probe_reports_corrupt_file(tmp_path):
    f = tmp_path / "bad.mp4"
    f.write_bytes(b"\x00" * 258)
    r = probe_video(str(f))
    assert not r["decode_ok"] and r["decode_error"]


def test_validate_exclusion_reasons():
    m = pd.DataFrame({
        "repo_path": ["a", "b", "c", "d", "e", "f", None],
        "issue": ["", "", "", "", "", "", "file_missing_in_repo"],
        "eligible_pre_probe": [True] * 6 + [False],
    })
    p = pd.DataFrame({
        "repo_path": ["a", "b", "c", "d", "e"],
        "download_ok": [True, False, True, True, True],
        "sha256_match": [True, None, False, True, True],
        "decode_ok": [True, None, None, False, True],
        "decoded_frames": [30, 0, 0, 0, 29],
    })
    v = validate(m, p, 30).set_index(m.index)
    assert v["exclusion_reason"].tolist() == [
        "", "download_failed", "sha256_mismatch", "undecodable",
        "fewer_than_min_decoded_frames", "not_probed", "file_missing_in_repo",
    ]
    assert v["eligible"].tolist() == [True] + [False] * 6

