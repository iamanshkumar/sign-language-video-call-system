import json
import shutil

import av
import numpy as np
import pandas as pd
import pytest

from asl.config import STEP2_CONFIG, load_yaml
from asl.landmarks import preprocess as pp
from asl.landmarks.benchmark import select_benchmark
from asl.landmarks.extract import RawLandmarks
from asl.landmarks.layout import (GROUP_INDEX, N_LANDMARKS, OBSERVED, PRESENT, SLICES, assemble_features,
                                  feature_dim, flag_mask)
from asl.landmarks.outputs import check_outputs, check_split_preserved
from asl.landmarks.process import array_name, preprocess_raw, save_arrays

CFG = load_yaml(STEP2_CONFIG)
RULES = CFG["rejection"]


def _series(n, fps):
    times = np.arange(n) / fps
    coords = np.zeros((n, N_LANDMARKS, 3))
    coords[:, :, 0] = times[:, None]  # x == time, so interpolation is easy to check
    present = np.ones((n, 4), bool)
    vis = np.ones((n, 33))
    return times, coords, present, vis


# ---------------------------------------------------------------- 1. temporal resampling
def test_resample_identity_at_30fps():
    t, c, p, v = _series(45, 30.0)
    tt, c2, p2, _ = pp.resample_uniform(t, c, p, v, 30.0, 30)
    assert len(tt) == 45 and np.allclose(c2, c) and p2.all()


def test_resample_upsamples_by_interpolation_not_repetition():
    t, c, p, v = _series(40, 15.0)  # 2.67 s at 15 FPS -> 80 frames at 30 FPS
    tt, c2, _, _ = pp.resample_uniform(t, c, p, v, 15.0, 30)
    assert len(tt) == 80
    x = c2[:, 0, 0]
    assert np.allclose(x[:-1], np.minimum(np.arange(79) / 30, t[-1]))  # linear in time
    assert len(np.unique(np.round(x[:-2], 9))) == 78  # no repeated frames


def test_resample_downsamples_60fps():
    t, c, p, v = _series(120, 60.0)
    tt, c2, _, _ = pp.resample_uniform(t, c, p, v, 60.0, 30)
    assert len(tt) == 60 and np.allclose(c2[:, 0, 0], np.arange(60) / 30)


def test_resample_presence_uses_nearest_when_neighbour_missing():
    t, c, p, v = _series(20, 15.0)
    p[5, GROUP_INDEX["left_hand"]] = False
    _, _, p2, _ = pp.resample_uniform(t, c, p, v, 15.0, 30)
    lh = p2[:, GROUP_INDEX["left_hand"]]
    assert not lh[10]  # exactly at the missing source frame
    assert lh[8] and lh[12]


# ---------------------------------------------------------------- 2./3. missing landmarks
def test_short_interior_gap_is_linearly_interpolated():
    c = np.arange(10, dtype=float)[:, None, None] * np.ones((10, 2, 3))
    present = np.ones(10, bool)
    present[3:8] = False  # gap of 5
    r = pp.fill_gaps(c, present, max_gap=5)
    assert r.present.all() and r.interpolated[3:8].all() and not r.interpolated[:3].any()
    assert np.allclose(r.coords[:, 0, 0], np.arange(10))


def test_long_gap_is_zero_filled_with_mask():
    c = np.ones((20, 2, 3))
    present = np.ones(20, bool)
    present[5:11] = False  # gap of 6 > 5
    r = pp.fill_gaps(c, present, max_gap=5)
    assert not r.present[5:11].any() and r.zero_filled[5:11].all()
    assert (r.coords[5:11] == 0).all() and (r.coords[:5] == 1).all()


def test_edge_gaps_are_not_interpolated():
    c = np.ones((10, 1, 3))
    present = np.ones(10, bool)
    present[:2] = present[-1] = False
    r = pp.fill_gaps(c, present, max_gap=5)
    assert r.zero_filled[[0, 1, 9]].all() and not r.interpolated.any()


def test_runs():
    assert pp.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 1)]
    assert pp.longest_run(np.zeros(4, bool)) == 0


# ---------------------------------------------------------------- 4./5. normalization
def _pose(t, ls, rs):
    pose = np.zeros((t, 33, 3))
    pose[:, 11], pose[:, 12] = ls, rs
    return pose


def test_shoulder_center_becomes_origin_and_distance_becomes_unit():
    pose = _pose(3, [300, 200, 10], [100, 200, -10])
    bf = pp.body_frame(pose, np.ones(3, bool), np.ones((3, 33)), 0.5)
    assert np.allclose(bf.center, [200, 200, 0]) and np.allclose(bf.scale, 200)
    pts = np.stack([pose[:, 11], pose[:, 12], np.tile([200, 400, 0], (3, 1))], axis=1)
    n = pp.normalize(pts, np.ones((3, 3), bool), bf)
    assert np.allclose(n[:, 0], [0.5, 0, 0.05]) and np.allclose(n[:, 1], [-0.5, 0, -0.05])
    assert np.allclose(n[:, 2], [0, 1, 0])


def test_normalization_is_invariant_to_camera_shift_and_distance():
    rng = np.random.default_rng(0)
    pose = _pose(4, [300, 200, 0], [100, 200, 0])
    pts = rng.normal(200, 50, (4, 5, 3))
    a = pp.normalize(pts, np.ones((4, 5), bool), pp.body_frame(pose, np.ones(4, bool), np.ones((4, 33)), 0.5))
    k, shift = 0.5, np.array([37.0, -12.0, 0.0])  # signer further away and moved
    b = pp.normalize(pts * k + shift, np.ones((4, 5), bool),
                     pp.body_frame(pose * k + shift, np.ones(4, bool), np.ones((4, 33)), 0.5))
    assert np.allclose(a[..., :2], b[..., :2])


def test_missing_shoulders_carry_last_valid_frame_and_fail_when_never_seen():
    pose = _pose(5, [300, 200, 0], [100, 200, 0])
    pose[3:, 11] = [900, 900, 0]  # garbage where shoulders are not visible
    vis = np.ones((5, 33))
    vis[3:, 11] = 0.1
    vis[0, 12] = 0.1
    bf = pp.body_frame(pose, np.ones(5, bool), vis, 0.5)
    assert bf.ok and list(bf.measured) == [False, True, True, False, False]
    assert np.allclose(bf.center, [200, 200, 0]) and np.allclose(bf.scale, 200)  # carried
    none = pp.body_frame(pose, np.zeros(5, bool), vis, 0.5)
    assert not none.ok


# ---------------------------------------------------------------- 6. active interval + rejection rules
def _hands(t, visible):
    """[T, 2] detections; `visible` = list of (start, stop) ranges where the left hand is seen."""
    h = np.zeros((t, 2), bool)
    for s0, s1 in visible:
        h[s0:s1, 0] = True
    return h


def test_active_interval_detection():
    iv = pp.active_interval(_hands(80, [(16, 40)]), min_run=3)
    assert (iv.start, iv.end, iv.frames) == (16, 39, 24)


def test_active_interval_ignores_isolated_blips_but_keeps_inner_gaps():
    h = _hands(90, [(2, 3), (20, 30), (34, 50), (80, 82)])  # 1-frame and 2-frame blips at the edges
    iv = pp.active_interval(h, min_run=3)
    assert (iv.start, iv.end) == (20, 49)


def test_hands_absent_at_beginning_and_end_are_not_counted_as_missing():
    h = _hands(90, [(25, 55)])  # 25 rest frames before, 35 after (61% of clip without hands)
    assert pp.fullclip_rejection_reason(90, h, RULES) == "both_hands_missing_fraction"  # old v1 rule
    assert pp.rejection_reason(90, h, RULES) is None                                   # v2 rule


def test_intermittent_disappearance_inside_interval_is_counted():
    ok = _hands(100, [(10, 20), (24, 60)])  # 4-frame gap inside: 4/50 = 8%
    assert pp.rejection_reason(100, ok, RULES) is None
    bad = _hands(100, [(10, 14), (20, 24), (30, 34), (40, 44)])  # 18 of 34 frames missing inside
    assert pp.hand_quality(bad[10:44])["both_missing_frac"] > 0.30
    assert pp.rejection_reason(100, bad, RULES) == "active_both_hands_missing_fraction"


def test_no_detectable_hands():
    assert pp.active_interval(np.zeros((60, 2), bool), 3) is None
    assert pp.rejection_reason(60, np.zeros((60, 2), bool), RULES) == "no_active_signing_interval"
    assert pp.rejection_reason(60, _hands(60, [(5, 7)]), RULES) == "no_active_signing_interval"  # blip only


def test_missing_percentage_only_inside_active_interval():
    h = _hands(120, [(40, 50), (54, 70)])  # interval 40..69 (30 frames), 4 missing inside
    iv = pp.active_interval(h, 3)
    q = pp.hand_quality(h[iv.start : iv.end + 1])
    assert iv.frames == 30 and q["both_missing_frames"] == 4 and abs(q["both_missing_frac"] - 4 / 30) < 1e-12
    assert pp.hand_quality(h)["both_missing_frac"] > 0.7  # whole clip would be 75% missing


def test_more_than_15_consecutive_missing_inside_interval():
    run16 = _hands(120, [(10, 40), (56, 90)])  # 16-frame gap inside, 16/80 = 20%
    assert pp.rejection_reason(120, run16, RULES) == "active_both_hands_missing_run"
    run15 = _hands(120, [(10, 40), (55, 90)])  # exactly 15: allowed
    assert pp.rejection_reason(120, run15, RULES) is None


def test_decoded_frame_rule_still_applies_to_whole_clip():
    assert pp.rejection_reason(29, _hands(29, [(0, 29)]), RULES) == "too_few_decoded_frames"


def _raw(n=60, fps=30.0, hands=None):
    """Synthetic extraction; `hands` = list of (start, stop) frame ranges with both hands."""
    t, c, p, v = _series(n, fps)
    c = np.random.default_rng(1).uniform(0.2, 0.8, (n, N_LANDMARKS, 3)).astype(np.float32)
    c[:, SLICES["pose"].start + 11] = [0.6, 0.4, 0.0]
    c[:, SLICES["pose"].start + 12] = [0.4, 0.4, 0.0]
    if hands is not None:
        p[:, [GROUP_INDEX["left_hand"], GROUP_INDEX["right_hand"]]] = False
        for s0, s1 in hands:
            p[s0:s1, [GROUP_INDEX["left_hand"], GROUP_INDEX["right_hand"]]] = True
    return RawLandmarks(times=t, coords=c, observed=p, pose_vis=v.astype(np.float32), repaired_nonfinite=0,
                        width=640, height=480, codec="h264", container_fps=fps, decode_s=0.0, mediapipe_s=0.0)


def test_short_active_interval_is_flagged_not_rejected():
    arrays, st = preprocess_raw(_raw(n=90, hands=[(30, 50)]), CFG)  # 20 active frames
    assert arrays is not None and st["rejection_reason"] is None
    assert st["active_frames"] == 20 and st["short_active_interval"] is True
    assert list(arrays["active_interval"]) == [30, 49] and st["original_frame_count"] == 90
    assert "short_active_interval" not in {pp.rejection_reason(90, _hands(90, [(30, 50)]), RULES)}


def test_short_active_interval_still_subject_to_hand_quality_rules():
    bad = _hands(90, [(30, 33), (38, 41), (46, 49)])  # 19-frame interval, 8 frames without hands inside (42%)
    assert pp.rejection_reason(90, bad, RULES) == "active_both_hands_missing_fraction"
    assert pp.rejection_reason(25, _hands(25, [(0, 25)]), RULES) == "too_few_decoded_frames"


def test_active_interval_30_frame_boundary():
    assert pp.rejection_reason(90, _hands(90, [(10, 39)]), RULES) is None  # 29 frames: accepted
    assert pp.rejection_reason(90, _hands(90, [(10, 40)]), RULES) is None  # 30 frames: accepted
    _, st29 = preprocess_raw(_raw(n=90, hands=[(10, 39)]), CFG)
    _, st30 = preprocess_raw(_raw(n=90, hands=[(10, 40)]), CFG)
    assert st29["rejection_reason"] is None and st29["short_active_interval"] is True
    assert st30["rejection_reason"] is None and st30["short_active_interval"] is False


def test_original_video_metadata_and_rest_frames_are_preserved():
    arrays, st = preprocess_raw(_raw(n=100, fps=25.0, hands=[(20, 60)]), CFG)
    assert st["original_frame_count"] == 100 and abs(st["original_fps"] - 25) < 1e-9
    assert abs(st["original_duration_s"] - 4.0) < 1e-9 and st["std_frame_count"] == 120
    # full-clip statistics still describe the whole clip, the interval is recorded separately
    assert st["both_hands_missing_pct"] > 50 and st["fullclip_rule_rejection"] == "both_hands_missing_fraction"
    assert st["active_start_frame"] == 24 and st["active_end_frame"] == 71 and st["active_frames"] == 48
    assert st["rest_frames_before"] == 24 and st["rest_frames_after"] == 120 - 72
    assert st["active_both_hands_missing_pct"] == 0
    # every frame is stored, including resting frames
    assert arrays["left_hand"].shape[0] == 120 and list(arrays["active_interval"]) == [24, 71]
    assert not flag_mask(arrays["flags"], "left_hand", OBSERVED)[:24].any()


def test_preprocess_rejects_no_hands_and_no_shoulders():
    arrays, st = preprocess_raw(_raw(hands=[]), CFG)
    assert arrays is None and st["rejection_reason"] == "no_active_signing_interval"
    r = _raw()
    r.pose_vis[:, 11] = 0.0
    arrays, st = preprocess_raw(r, CFG)
    assert arrays is None and st["rejection_reason"] == "normalization_no_valid_shoulders"


def test_preprocess_is_deterministic():
    a1, s1 = preprocess_raw(_raw(n=70, fps=24.0, hands=[(5, 40), (44, 60)]), CFG)
    a2, s2 = preprocess_raw(_raw(n=70, fps=24.0, hands=[(5, 40), (44, 60)]), CFG)
    assert s1 == s2 and all(np.array_equal(a1[k], a2[k]) for k in a1)


# ---------------------------------------------------------------- representation + storage dtype
def test_representation_extraction_per_configuration():
    arrays, _ = preprocess_raw(_raw(n=45), CFG)
    dims = {k: assemble_features(arrays, v).shape for k, v in CFG["configurations"].items()}
    assert dims == {"hands": (45, 128), "hands_pose": (45, 261), "hands_pose_face": (45, 1666)}
    hands_only = {k: arrays[k] for k in ("left_hand", "right_hand", "flags")}  # face/pose never read
    f = assemble_features(hands_only, ["left_hand", "right_hand"])
    assert np.array_equal(f[:, :63], arrays["left_hand"].astype(np.float32).reshape(45, -1))
    assert (f[:, 63] == 1).all()


def test_float16_storage_roundtrip_and_float32_option():
    cfg32 = {**CFG, "output": {**CFG["output"], "coord_dtype": "float32"}}
    cfg16 = {**CFG, "output": {**CFG["output"], "coord_dtype": "float16"}}
    raw = _raw(n=60, hands=[(10, 50)])
    a32, _ = preprocess_raw(raw, cfg32)
    a16, _ = preprocess_raw(raw, cfg16)
    assert a32["face"].dtype == np.float32 and a16["face"].dtype == np.float16 and a16["flags"].dtype == np.uint8
    for g in ("pose", "face", "left_hand", "right_hand"):
        err = np.abs(a16[g].astype(np.float32) - a32[g])
        assert err.max() <= np.abs(a32[g]).max() * 2**-11 + 1e-7  # float16 half-ulp bound
        assert ((a16[g] == 0) == (a32[g] == 0)).all()  # zero-fill semantics preserved
    assert a16["center"].dtype == np.float32 and a16["scale"].dtype == np.float32
    present = flag_mask(a16["flags"], "left_hand", PRESENT)
    assert not present[:10].any() and present[10:50].all()


def test_default_storage_is_approved_mixed_precision():
    assert CFG["output"]["coord_dtype"] == {"pose": "float32", "face": "float16",
                                            "left_hand": "float16", "right_hand": "float16"}
    arrays, _ = preprocess_raw(_raw(n=40), CFG)
    assert arrays["pose"].dtype == np.float32 and arrays["pose_vis"].dtype == np.float32
    assert arrays["face"].dtype == np.float16 and arrays["left_hand"].dtype == np.float16


def test_storage_groups_are_configurable():
    cfg = {**CFG, "output": {**CFG["output"], "groups_to_store": ["pose", "left_hand", "right_hand"]}}
    arrays, _ = preprocess_raw(_raw(n=40), cfg)
    assert "face" not in arrays and assemble_features(arrays, CFG["configurations"]["hands_pose"]).shape == (40, 261)


# ---------------------------------------------------------------- 7. benchmark selection
def _split_and_probe():
    rows, probe = [], []
    for c in range(40):
        for i in range(20):
            sid = f"p/w{c}_{i}.mp4"
            aug = f"w{c}_0.mp4" if i >= 18 else ""
            rows.append({"sample_id": sid, "repo_path": sid, "word": f"w{c}", "sha256": sid,
                         "dup_group_size": 2 if i == 5 else 1, "aug_source": aug,
                         "split": "train" if i < 16 or i >= 18 else "test"})
            if i % 2 == 0:
                probe.append({"sample_id": sid, "decoded_frames": 20 + 7 * i + c, "width": 640 if c % 7 else 288,
                              "height": 480 if c % 7 else 192})
    return pd.DataFrame(rows), pd.DataFrame(probe)


def test_benchmark_selection_is_deterministic_and_train_only():
    split, probe = _split_and_probe()
    q = {"short_clips": 3, "non_default_resolution": 4, "augmented": 5, "duplicate_groups": 6}
    a = select_benchmark(split, probe, seed=42, n=100, quotas=q)
    b = select_benchmark(split.sample(frac=1, random_state=1), probe.sample(frac=1, random_state=2),
                         seed=42, n=100, quotas=q)
    assert a["sample_id"].tolist() == b["sample_id"].tolist()
    assert len(a) == 100 and a["sample_id"].is_unique and (a["split"] == "train").all()
    counts = a["benchmark_reason"].value_counts()
    assert counts["augmented"] == 5 and counts["duplicate_group"] == 6 and counts["short_clip"] == 3
    c = select_benchmark(split, probe, seed=7, n=100, quotas=q)
    assert c["sample_id"].tolist() != a["sample_id"].tolist()


# ---------------------------------------------------------------- 8. resume behaviour
def _make_video(path, n, fps=30):
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(n):
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.full((48, 64, 3), i, np.uint8), format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_pipeline_resumes_without_reprocessing(tmp_path):
    from asl.landmarks.pipeline import completed_ids, load_metadata, run_pipeline

    src = tmp_path / "src"
    src.mkdir()
    for i, n in enumerate([40, 35, 20]):
        _make_video(src / f"v{i}.mp4", n)
    samples = pd.DataFrame({"sample_id": [f"v{i}.mp4" for i in range(3)], "repo_path": [f"v{i}.mp4" for i in range(3)],
                            "word": ["a", "b", "c"], "split": "train", "sha256": "x"})
    calls = []

    def fetch(repo_path, sha, dest):
        calls.append(repo_path)
        shutil.copy(src / repo_path, dest)
        return {"download_ok": True}

    out = tmp_path / "out"
    kw = dict(workers=1, download_workers=1, batch_size=2, min_free_disk_gb=0)
    run_pipeline(samples, CFG, out, tmp_path / "cache", fetch, limit=2, **kw)
    assert completed_ids(out) == {"v0.mp4", "v1.mp4"}
    run_pipeline(samples, CFG, out, tmp_path / "cache", fetch, **kw)
    assert calls == ["v0.mp4", "v1.mp4", "v2.mp4"]  # nothing fetched twice
    run_pipeline(samples, CFG, out, tmp_path / "cache", fetch, **kw)
    assert len(calls) == 3
    meta = load_metadata(out)
    assert len(meta) == 3 and meta["sample_id"].is_unique
    # synthetic clips contain no person -> rejected; the 20-frame clip by the frame rule
    reasons = dict(zip(meta["sample_id"], meta["rejection_reason"]))
    assert reasons["v2.mp4"] == "too_few_decoded_frames"
    assert reasons["v0.mp4"] == "no_active_signing_interval"
    assert check_split_preserved(meta, samples) == []
    assert not list((tmp_path / "cache").glob("*/*.mp4"))  # videos deleted after processing


# ---------------------------------------------------------------- 9. metadata consistency
def _record(arrays, st, name, size):
    return {"sample_id": "p/x.mp4", "rejected": st["rejection_reason"] is not None,
            "rejection_reason": st["rejection_reason"], "array_file": name, "bytes": size,
            "array_kept_for_rejected": False, "coord_dtype": json.dumps(CFG["output"]["coord_dtype"]),
            "stored_groups": json.dumps(CFG["output"]["groups_to_store"]),
            "configurations": json.dumps({k: feature_dim(v) for k, v in CFG["configurations"].items()}), **st}


def test_saved_arrays_match_metadata(tmp_path):
    arrays, st = preprocess_raw(_raw(n=50, fps=25.0, hands=[(5, 45)]), CFG)
    assert arrays is not None and st["std_frame_count"] == 60 and st["original_frame_count"] == 50
    name = array_name("p/x.mp4")
    size = save_arrays(arrays, tmp_path / name)
    meta = pd.DataFrame([_record(arrays, st, name, size)])
    assert check_outputs(meta, tmp_path, CFG["configurations"]) == []
    assert json.loads(meta.loc[0, "configurations"]) == {"hands": 128, "hands_pose": 261, "hands_pose_face": 1666}
    assert check_outputs(meta.assign(std_frame_count=59), tmp_path, CFG["configurations"])
    assert check_outputs(meta.assign(active_start_frame=0), tmp_path, CFG["configurations"])
    assert check_outputs(meta.assign(coord_dtype=json.dumps("float16")), tmp_path, CFG["configurations"])


def test_compressed_and_uncompressed_files_load_identically(tmp_path):
    from asl.landmarks.outputs import load_arrays
    arrays, _ = preprocess_raw(_raw(n=40), CFG)
    save_arrays(arrays, tmp_path / "a.npz", compress=False)
    save_arrays(arrays, tmp_path / "b.npz", compress=True)
    a, b = load_arrays(tmp_path / "a.npz"), load_arrays(tmp_path / "b.npz")
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_rejected_record_must_have_reason_and_no_file(tmp_path):
    meta = pd.DataFrame([{"sample_id": "a", "rejected": True, "rejection_reason": None, "array_file": None,
                          "configurations": "{}"}])
    assert check_outputs(meta, tmp_path, CFG["configurations"])


def test_checker_treats_nan_array_file_as_missing(tmp_path):
    meta = pd.DataFrame([{"sample_id": "a", "rejected": True, "rejection_reason": "too_few_decoded_frames",
                          "array_file": np.nan, "array_kept_for_rejected": False, "configurations": "{}"}])
    assert check_outputs(meta, tmp_path, CFG["configurations"]) == []


def test_split_preservation_check():
    frozen = pd.DataFrame({"sample_id": ["a", "b"], "split": ["train", "test"], "word": ["x", "y"],
                           "repo_path": ["a", "b"]})
    ok = frozen.copy()
    assert check_split_preserved(ok, frozen) == []
    assert check_split_preserved(ok.assign(split=["train", "train"]), frozen)
    assert check_split_preserved(pd.DataFrame({"sample_id": ["zz"], "split": ["train"], "word": ["x"],
                                               "repo_path": ["zz"]}), frozen)
