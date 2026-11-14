"""Step 6: inference loader, input contract, streaming segmentation and post-processing.

Synthetic checkpoints and landmark streams only (no MediaPipe, no dataset); the real-candidate
equivalence test runs only if that candidate has been exported locally."""
import json

import numpy as np
import pytest
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference import (ConfidenceFilter, InputError, ModelLoadError, Prediction, SignRecognizer, Smoother,
                           StreamingRecognizer)
from asl.inference.spec import preprocessing_spec, spec_differences
from asl.landmarks.extract import RawLandmarks
from asl.landmarks.layout import GROUP_INDEX, N_LANDMARKS, SLICES, assemble_features, feature_names
from asl.landmarks.process import preprocess_raw
from asl.landmarks.storage import apply_profile
from asl.models.recurrent import build_model
from asl.sequences.labels import vocabulary_hash
from asl.sequences.resample import resample_interval

C2 = load_yaml(STEP2_CONFIG)
C3 = load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml")
CLASSES = ["alpha", "bravo", "charlie", "delta", "echo"]
NEURAL = {"hidden_size": 16, "num_layers": 2, "dropout": 0.3}


def _export(tmp_path, *, spec=None, classes=CLASSES, model="bilstm", drop=(), mapping_classes=None, input_dim=261):
    torch.manual_seed(0)
    m = build_model(model, input_dim, len(classes), NEURAL)
    spec = spec or preprocessing_spec(C2, C3, "hands_pose_f32", "hands_pose", 30)
    ck = {"state_dict": m.state_dict(), "model": model, "input_dim": input_dim, "num_classes": len(classes), **NEURAL,
          "window": 30, "feature_config": "hands_pose", "groups": ["left_hand", "right_hand", "pose"],
          "label_mapping_sha256": vocabulary_hash(classes), "preprocessing": spec}
    for k in drop:
        ck.pop(k)
    torch.save(ck, tmp_path / "best_model.pt")
    mc = mapping_classes or classes
    (tmp_path / "class_mapping.json").write_text(json.dumps(
        {"num_classes": len(mc), "index_to_class": {str(i): w for i, w in enumerate(mc)},
         "class_to_index": {w: i for i, w in enumerate(mc)}}))
    return tmp_path / "best_model.pt", tmp_path / "class_mapping.json", m


@pytest.fixture()
def rec(tmp_path):
    p, c, _ = _export(tmp_path)
    return SignRecognizer.load(p, c, C2, C3)


def _frame(t, hands=True, nan_hand=False):
    """One synthetic frame in MediaPipe image coordinates: shoulders at x 0.4/0.6, moving hands."""
    c = np.zeros((N_LANDMARKS, 3), np.float32)
    o = np.zeros(4, bool)
    c[SLICES["pose"], :2] = 0.5
    c[SLICES["pose"].start + 11, :2] = (0.6, 0.5)
    c[SLICES["pose"].start + 12, :2] = (0.4, 0.5)
    o[GROUP_INDEX["pose"]] = True
    if hands:
        for g, x0 in (("left_hand", 0.65), ("right_hand", 0.35)):
            c[SLICES[g], 0] = x0 + 0.05 * np.sin(6 * t) + np.linspace(0, 0.05, 21)
            c[SLICES[g], 1] = 0.6 - 0.1 * np.cos(4 * t)
            o[GROUP_INDEX[g]] = True
        if nan_hand:
            c[SLICES["right_hand"].start, 0] = np.nan
    return c, o, np.ones(33, np.float32)


def _raw(n=60, fps=30.0, hands=lambda i: 10 <= i < 50):
    fr = [_frame(i / fps, hands(i)) for i in range(n)]
    return RawLandmarks(np.arange(n) / fps, np.stack([f[0] for f in fr]), np.stack([f[1] for f in fr]),
                        np.stack([f[2] for f in fr]), 0, 640, 480, "test", fps, 0.0, 0.0)


# ---------------------------------------------------------------- loading and the input contract
def test_loads_and_exposes_the_training_contract(rec):
    assert rec.window == 30 and rec.feature_dim == 261 and rec.groups == ["left_hand", "right_hand", "pose"]
    assert rec.classes == CLASSES and rec.info["model"] == "bilstm" and rec.model.fc.in_features == 2 * 16
    assert len(feature_names(rec.groups)) == 261 and feature_names(rec.groups)[63] == "left_hand_present"


@pytest.mark.parametrize("case", ["missing_file", "corrupt", "legacy_no_preprocessing", "mapping_hash",
                                  "mapping_count", "preprocessing_changed", "weights_shape"])
def test_load_failures_are_reported_not_crashing(tmp_path, case):
    if case == "missing_file":
        p, c = tmp_path / "nope.pt", tmp_path / "nope.json"
    elif case == "corrupt":
        p, c, _ = _export(tmp_path)
        p.write_bytes(b"not a checkpoint")
    elif case == "legacy_no_preprocessing":  # e.g. the pre-Step-6 models/best_model.pt
        p, c, _ = _export(tmp_path, drop=("preprocessing",))
    elif case == "mapping_hash":
        p, c, _ = _export(tmp_path, mapping_classes=["alpha", "bravo", "charlie", "delta", "zulu"])
    elif case == "mapping_count":
        p, c, _ = _export(tmp_path, mapping_classes=CLASSES[:4])
    elif case == "preprocessing_changed":
        spec = preprocessing_spec(C2, C3, "hands_pose_f32", "hands_pose", 30)
        spec = {**spec, "rejection": {**spec["rejection"], "max_interp_gap_changed": True}}
        p, c, _ = _export(tmp_path, spec=spec)
    else:  # weights for 128 features, contract says 261
        p, c, _ = _export(tmp_path, input_dim=128)
    with pytest.raises(ModelLoadError):
        SignRecognizer.load(p, c, C2, C3)


def test_spec_records_every_preprocessing_setting():
    a = preprocessing_spec(C2, C3, "hands_pose_f32", "hands_pose", 30)
    assert a["feature_dim"] == 261 and a["window"] == 30 and a["stored_groups"] == ["pose", "left_hand", "right_hand"]
    assert spec_differences(a, preprocessing_spec(C2, C3, "hands_pose_f32", "hands_pose", 45)) == ["window"]
    with pytest.raises(ValueError):  # face is not stored with this profile
        preprocessing_spec(C2, C3, "hands_pose_f32", "hands_pose_face", 30)


# ---------------------------------------------------------------- predictions
def test_predict_features_matches_model_and_validates_input(tmp_path):
    p, c, m = _export(tmp_path)
    rec = SignRecognizer.load(p, c, C2, C3, top_k=3)
    x = np.random.default_rng(0).normal(size=(30, 261)).astype(np.float32)
    pred = rec.predict_features(x)
    with torch.no_grad():
        prob = torch.softmax(m.eval()(torch.from_numpy(x)[None]), 1)[0]
    assert pred.status == "ok" and pred.index == int(prob.argmax()) and pred.label == CLASSES[pred.index]
    assert pred.confidence == pytest.approx(float(prob.max())) and len(pred.top_k) == 3
    assert [c_ for _, c_ in pred.top_k] == sorted([c_ for _, c_ in pred.top_k], reverse=True)
    assert rec.predict_features(x.astype(np.float64)).index == pred.index  # dtype is normalized, not rejected
    bad = x.copy()
    bad[3, 7] = np.nan
    for wrong in (np.zeros((29, 261)), np.zeros((30, 260)), np.zeros((1, 30, 261)), bad, np.full((30, 261), "a")):
        with pytest.raises(InputError):
            rec.predict_features(wrong)


def test_raw_clip_uses_exactly_the_training_preprocessing(rec):
    raw = _raw()
    arrays, st = preprocess_raw(raw, apply_profile(C2, "hands_pose_f32"))
    s, e = (int(v) for v in arrays["active_interval"])
    expected = assemble_features(resample_interval(arrays, rec.groups, s, e, 30), rec.groups)
    feats, _ = rec.features_from_raw(raw)
    assert feats.shape == (30, 261) and feats.dtype == np.float32 and np.array_equal(feats, expected)
    assert rec.predict_raw(raw).index == rec.predict_features(expected).index


@pytest.mark.parametrize("raw,reason", [(lambda: _raw(n=20, hands=lambda i: True), "too_few_decoded_frames"),
                                        (lambda: _raw(hands=lambda i: False), "no_active_signing_interval")])
def test_incomplete_or_handless_clips_are_rejected_with_the_training_reason(rec, raw, reason):
    p = rec.predict_raw(raw())
    assert p.status == "rejected" and p.label is None and p.reason == reason


def test_missing_hand_is_represented_like_training(rec):
    """Only the right hand present: left-hand block and presence column are zero, as in training."""
    raw = _raw()
    raw.observed[:, GROUP_INDEX["left_hand"]] = False
    raw.coords[:, SLICES["left_hand"]] = 0
    feats, _ = rec.features_from_raw(raw)
    assert np.all(feats[:, :64] == 0) and np.all(feats[:, 127] == 1)


# ---------------------------------------------------------------- streaming
def _stream(sr, spans, fps=30.0, start=0.0, **kw):
    """spans: list of (seconds, hands). Returns events."""
    evs, t = [], start
    for secs, hands in spans:
        for _ in range(int(round(secs * fps))):
            c, o, v = _frame(t, hands, **kw)
            ev = sr.push(t, c, o, v)
            if ev:
                evs.append(ev)
            t += 1 / fps
    return evs, t


def test_stream_yields_one_event_per_sign(rec):
    sr = StreamingRecognizer(rec, 640, 480)
    evs, _ = _stream(sr, [(1.0, False), (1.5, True), (1.0, False), (1.2, True), (1.0, False)])
    assert len(evs) == 2 and all(e.prediction.status == "ok" and not e.forced for e in evs)
    assert evs[0].frames >= 30 and evs[0].emitted == evs[0].prediction.label  # pass-through by default
    # the segment closes end_rest_s after the last hand frame, which lies inside the signing span
    assert 2.4 <= evs[0].last_hand_s <= 2.5 + 1e-9 and evs[0].end_s - evs[0].last_hand_s >= sr.end_rest_s - 1e-9
    assert sr.flush() is None


def test_stream_forced_cut_flush_and_rejections(rec):
    sr = StreamingRecognizer(rec, 640, 480, max_segment_s=2.0)
    evs, _ = _stream(sr, [(0.5, False), (3.0, True)])
    assert len(evs) == 1 and evs[0].forced
    tail = sr.flush()  # the rest of the long sign
    assert tail is not None
    short = StreamingRecognizer(rec, 640, 480, pre_roll_s=0.0)
    evs, _ = _stream(short, [(0.3, True), (0.5, False)])  # 9 hand frames + rest < 30 decoded frames
    assert len(evs) == 1 and evs[0].prediction.status == "rejected" and evs[0].emitted is None
    assert evs[0].prediction.reason == "too_few_decoded_frames"


def test_stream_handles_invalid_values_and_rejects_bad_input(rec):
    sr = StreamingRecognizer(rec, 640, 480)
    evs, t = _stream(sr, [(0.5, False), (1.2, True), (0.6, False)], nan_hand=True)  # NaN in the right hand
    assert len(evs) == 1 and evs[0].prediction.status in ("ok", "rejected")
    c, o, v = _frame(t)
    with pytest.raises(InputError):
        sr.push(t - 1.0, c, o, v)  # time going backwards
    with pytest.raises(InputError):
        sr.push(t + 1, c[:10], o, v)  # wrong shape
    with pytest.raises(ValueError):
        StreamingRecognizer(rec, 0, 480)


def test_stream_confidence_threshold_blocks_emission(rec):
    sr = StreamingRecognizer(rec, 640, 480, confidence_threshold=1.0)  # nothing is that confident
    evs, _ = _stream(sr, [(0.5, False), (1.5, True), (0.6, False)])
    assert len(evs) == 1 and evs[0].prediction.status == "low_confidence" and evs[0].emitted is None


# ---------------------------------------------------------------- post-processing units
def _p(label, conf=0.9, status="ok"):
    return Prediction(status, label, 0, conf, ((label, conf),))


def test_confidence_filter():
    f = ConfidenceFilter(0.5)
    assert f(_p("a", 0.6)).status == "ok" and f(_p("a", 0.4)).status == "low_confidence"
    assert f(Prediction("rejected", None, None, None)).status == "rejected"
    assert ConfidenceFilter()(_p("a", 0.01)).status == "ok"  # default: pass-through
    with pytest.raises(ValueError):
        ConfidenceFilter(1.5)


def test_smoother_passthrough_majority_and_no_repeats():
    s = Smoother()
    assert [s.update(_p(x)) for x in "aab"] == ["a", "a", "b"]  # pass-through keeps repeats
    s = Smoother(window=3, min_votes=2)
    out = [s.update(_p(x)) for x in "abaabbb"]
    assert out == [None, None, "a", None, None, "b", None]
    assert s.update(_p("c", status="low_confidence")) is None
    s.reset()
    assert s.update(_p("b")) is None and s.update(_p("b")) == "b"
    with pytest.raises(ValueError):
        Smoother(window=2, min_votes=3)


# ---------------------------------------------------------------- the real exported candidate (local only)
_CANDS = sorted(p for p in (PROJECT_ROOT / "models" / "candidates").glob("*") if (p / "best_model.pt").exists()) \
    if (PROJECT_ROOT / "models" / "candidates").exists() else []
CAND = _CANDS[0] if _CANDS else PROJECT_ROOT / "models" / "candidates" / "none"


@pytest.mark.skipif(not (CAND / "best_model.pt").exists(), reason="candidate not exported in this checkout")
def test_real_candidate_reproduces_source_checkpoint_on_test_sequences():
    import pandas as pd

    from asl.landmarks.pipeline import load_metadata
    from asl.landmarks.storage import available_configurations, store_groups
    from asl.models.train import load_checkpoint
    from asl.sequences.dataset import SequenceDataset
    from asl.sequences.labels import load_label_mapping

    rec = SignRecognizer.load(CAND / "best_model.pt", CAND / "class_mapping.json", C2, C3)
    src = PROJECT_ROOT / rec.info["source_experiment"]
    meta = json.loads((src / "metadata.json").read_text())
    ref = build_model(meta["model"], 261, 100, meta["neural_config"])
    load_checkpoint(src / "best.pt", ref)
    st = C3["stores"]["dev100_hp"]
    cf = available_configurations(C2["configurations"], store_groups(load_metadata(PROJECT_ROOT / st["landmarks_dir"])))
    ds = SequenceDataset(pd.read_parquet(PROJECT_ROOT / st["index_dir"] / "index.parquet"),
                         PROJECT_ROOT / st["landmarks_dir"] / "arrays", "test", 30, "hands_pose", cf,
                         load_label_mapping(PROJECT_ROOT / st["label_mapping"]))
    x = torch.stack([ds[i][0] for i in range(32)])
    with torch.no_grad():
        assert torch.equal(ref.eval()(x), rec.model(x))
    assert [rec.predict_features(ds[i][0].numpy()).label for i in range(5)] == \
        [rec.classes[int(i)] for i in ref(x[:5]).argmax(1)]


# ---------------------------------------------------------------- promotion is reversible and checked
def test_promote_and_restore_with_backup(tmp_path, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("promote", PROJECT_ROOT / "scripts" / "s4_05_promote_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    models = tmp_path / "models"
    cand = models / "candidates" / "x"
    cand.mkdir(parents=True)
    _export(cand)
    (cand / "export_report.json").write_text(json.dumps({"all_checks_pass": True}))
    (models / "best_model.pt").write_bytes(b"old weights")
    (models / "class_mapping.json").write_text("old mapping")
    monkeypatch.setattr(mod, "MODELS", models)
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    def run(*a):
        monkeypatch.setattr(sys, "argv", ["s4_05", *a])
        mod.main()

    run("--candidate", "models/candidates/x", "--dry-run")
    assert (models / "best_model.pt").read_bytes() == b"old weights"  # dry run changes nothing
    run("--candidate", "models/candidates/x")
    assert (models / "best_model.pt").read_bytes() == (cand / "best_model.pt").read_bytes()
    backups = sorted((models / "backups").iterdir())
    assert len(backups) == 1 and (backups[0] / "best_model.pt").read_bytes() == b"old weights"
    assert json.loads((backups[0] / "manifest.json").read_text())["files"].keys() == {"best_model.pt", "class_mapping.json"}
    run("--restore", str(backups[0].relative_to(tmp_path)))  # legacy files restore with a warning
    assert (models / "best_model.pt").read_bytes() == b"old weights"
    assert (models / "class_mapping.json").read_text() == "old mapping"
    assert len(list((models / "backups").iterdir())) == 2  # the restore backed up what it replaced
    (cand / "export_report.json").write_text(json.dumps({"all_checks_pass": False}))
    with pytest.raises(SystemExit):
        run("--candidate", "models/candidates/x")


# ---------------------------------------------------------------- live-evaluation helpers
def _ev(label, last_hand, status="ok", conf=0.8, top=None, emitted_end=None):
    from asl.inference.recognizer import SignEvent

    p = Prediction(status, label if status != "rejected" else None, 0 if label else None,
                   conf if status != "rejected" else None, tuple(top or ([(label, conf)] if label else [])),
                   "too_few_decoded_frames" if status == "rejected" else "")
    return SignEvent(p, label, last_hand - 1.0, last_hand + 0.4, 40, False, last_hand)


def test_assign_and_score_attempts():
    from asl.inference.evaluation import Attempt, assign_events, attempt_row, summarize

    att = [Attempt(0, "water", 3.0, 8.0), Attempt(1, "water", 11.0, 16.0), Attempt(2, "red", 19.0, 24.0),
           Attempt(3, "red", 27.0, 32.0)]
    evs = [(_ev("water", 5.0), 5.45), (_ev("wet", 13.0, top=[("wet", 0.6), ("water", 0.3)]), 13.5),
           (_ev("water", 14.5), 15.0),                      # second event in attempt 1 -> duplicate
           (_ev(None, 22.0, status="rejected"), 22.5),
           (_ev("red", 40.0), 40.5)]                         # outside every window
    orphans = assign_events(att, evs, grace_s=1.5)
    assert len(orphans) == 1
    for a, hands in zip(att, (10, 0, 5, 0)):
        a.frames, a.hand_frames = 10, hands
    rows = [attempt_row(a) for a in att]
    assert [r["outcome"] for r in rows] == ["correct", "wrong", "rejected", "missed"]
    assert rows[0]["delay_s"] == pytest.approx(0.45) and rows[1]["duplicates"] == 1 and rows[1]["in_top5"]
    assert rows[2]["reason"] == "too_few_decoded_frames" and rows[3]["predicted"] is None
    s = summarize(rows)
    assert s["accuracy_all_attempts"] == 0.25 and s["accuracy_when_answered"] == 0.5
    assert s["top5_rate_when_answered"] == 1.0 and s["attempts_with_duplicates"] == 1
    assert s["rejection_reasons"] == {"too_few_decoded_frames": 1} and s["per_sign_accuracy"] == {"red": 0.0, "water": 0.5}


def test_replay_source_paces_and_drops_like_a_live_camera():
    from asl.inference.evaluation import VideoReplaySource

    src = VideoReplaySource.__new__(VideoReplaySource)
    clock = {"t": 0.0}
    src.frames = [(i / 30, np.full((2, 2, 3), i, np.uint8)) for i in range(30)]
    src.clock, src.sleep = (lambda: clock["t"]), (lambda d: clock.__setitem__("t", clock["t"] + d))
    src.t0, src.i, src.dropped = 0.0, 0, 0
    ok, img, t = src.read()                                 # on time: frame 0, no sleep
    assert ok and t == 0.0 and img[0, 0, 0] == 0
    ok, img, t = src.read()                                 # early: waits for frame 1
    assert t == pytest.approx(1 / 30) and clock["t"] == pytest.approx(1 / 30) and src.dropped == 0
    clock["t"] += 0.2                                       # processing took 200 ms
    ok, img, t = src.read()                                 # newest due frame, older ones dropped
    assert img[0, 0, 0] == 7 and src.dropped == 5
    clock["t"] = 10.0
    assert src.read()[2] == pytest.approx(29 / 30) and src.read()[0] is False


def test_calibration_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cal", PROJECT_ROOT / "scripts" / "s6_04_confidence_calibration.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    conf = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.3])
    correct = np.array([True, True, True, False, True, False])
    assert cal.ece(np.full(4, 0.75), np.array([True, True, True, False])) == pytest.approx(0.0)  # calibrated
    assert cal.ece(np.full(4, 1.0), np.array([True, False, True, False])) == pytest.approx(0.5)  # overconfident
    rows = {r["threshold"]: r for r in cal.coverage_table(conf, correct, [0.0, 0.8])}
    assert rows[0.0]["coverage"] == 1.0 and rows[0.0]["accuracy_of_accepted"] == pytest.approx(4 / 6)
    assert rows[0.8]["coverage"] == 0.5 and rows[0.8]["accuracy_of_accepted"] == 1.0
    assert cal.suggest_threshold(conf, correct, 1.0) == pytest.approx(0.61)  # first threshold above the 0.6 error
    assert cal.suggest_threshold(conf, np.zeros(6, bool), 0.5) is None


# ---------------------------------------------------------------- regressions from the S01 webcam session
def test_grace_never_reaches_into_the_next_ready_phase():
    """Bug: an attempt's 1.5 s grace overlapped the next attempt's WAIT phase, so a sign made for
    the next prompt could be credited to the previous one; WAIT-phase signs were silently dropped."""
    from asl.inference.evaluation import Attempt, assign_events, attempt_row

    a0 = Attempt(0, "africa", 3.0, 8.0, ready_start_s=0.0)
    a1 = Attempt(1, "blind", 11.1, 16.1, ready_start_s=8.05)
    evs = [(_ev("pepper", 8.6), 9.0),      # inside a0's old grace, but during a1's WAIT
           (_ev("blind", 10.0), 10.4),     # WAIT phase of a1 (signed before GO)
           (_ev("blind", 13.0), 13.4)]     # a1's real answer
    orphans = assign_events([a0, a1], evs, grace_s=1.5)
    assert orphans == [] and a0.events == [] and len(a1.early_events) == 2 and len(a1.events) == 1
    r0, r1 = attempt_row(a0), attempt_row(a1)
    assert r0["outcome"] == "missed" and r1["outcome"] == "correct" and r1["early_events"] == 2
    assert (r1["ready_start_s"], r1["start_s"], r1["end_s"]) == (8.05, 11.1, 16.1)
    assert r1["early_labels"] == ["pepper", "blind"] and r1["started_before_go"] is False


S01 = PROJECT_ROOT / "reports" / "step6" / "webcam_eval" / "20261019T071836Z_camera_S01"


@pytest.mark.skipif(not (S01 / "events.jsonl").exists(), reason="S01 session report not present")
def test_s01_session_rescored_with_reconstructed_prompt_times():
    """The S01 report has no prompt times; the schedule (3 s WAIT + 5 s GO, first prompt ~1.6 s,
    ~0.05 s loop overrun per cycle) reproduces its recorded matching exactly. Re-scored with the fixed
    matching, the events that were credited through the overlapping grace move to the next prompt's
    WAIT phase - and accuracy is still 0 with no expected word in any top-5."""
    from asl.inference.evaluation import Attempt, assign_events, attempt_row, summarize
    from asl.inference.recognizer import SignEvent

    rows = [json.loads(l) for l in (S01 / "events.jsonl").read_text().splitlines()]
    evs = [(SignEvent(Prediction(r["status"], r["label"], None, r["confidence"], tuple(map(tuple, r["top_k"])), r["reason"]),
                      r["emitted"], r["start_s"], r["end_s"], r["frames"], r["forced"], r["last_hand_s"]), r["emitted_at_s"])
           for r in rows]
    signs = ["africa", "blind", "computer", "director", "forbid", "here", "more", "please", "shower", "tuesday"]
    t, att = 1.58, []
    for k, s in enumerate(x for x in signs for _ in range(2)):
        go = t + 3.0 + 0.03
        att.append(Attempt(k, s, go, go + 5.0, ready_start_s=t))
        t = go + 5.0 + 0.02
    orphans = assign_events(att, evs, grace_s=1.5)
    out = [attempt_row(a) for a in att]
    s = summarize(out)
    assert s["outcomes"].get("correct", 0) == 0 and s["top5_rate_when_answered"] == 0.0
    assert s["early_events_total"] >= 6                     # the 6 WAIT-phase events found in the analysis
    assert sum(a["events"] for a in out) + s["early_events_total"] + len(orphans) == len(evs)
    assert not any(any(l == a.expected for e, _ in a.early_events for l, _ in e.prediction.top_k) for a in att)


def test_camera_source_converts_to_rgb_and_does_not_invent_a_drop_count():
    """Bug: CameraSource reported dropped_frames = 0 although drops are not observable via OpenCV."""
    import cv2

    from asl.inference.evaluation import CameraSource

    class FakeCap:
        def __init__(self):
            self.props, self.n = {}, 0

        def set(self, prop, val):
            self.props[prop] = val
            return True

        def get(self, prop):
            return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

        def read(self):
            self.n += 1
            bgr = np.zeros((480, 640, 3), np.uint8)
            bgr[..., 0] = 255  # pure blue in BGR
            return (self.n <= 3), bgr

        def release(self):
            pass

    cap = FakeCap()
    src = CameraSource(0, 640, 480, 30, capture=cap)
    assert cap.props == {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FPS: 30}
    assert src.dropped is None and src.reported_fps == 30.0 and (src.width, src.height) == (640, 480)
    times = []
    for _ in range(3):
        ok, rgb, t = src.read()
        assert ok and rgb[0, 0].tolist() == [0, 0, 255]  # blue ends up in the B channel of RGB
        times.append(t)
    assert times == sorted(times) and src.read()[0] is False


def test_frame_diagnostics():
    from asl.inference.evaluation import frame_diagnostics

    t = np.arange(10) / 15.0                     # 15 fps
    obs = np.zeros((10, 4), bool)
    obs[:, 0] = True
    obs[2:6, 3] = True                           # right hand only in 4 frames
    obs[6:8, 2:4] = True                         # both hands in 2 frames
    dx = np.full(10, 0.2)
    d = frame_diagnostics(t, obs, dx)
    assert d["processed_fps"] == pytest.approx(15.0) and d["frame_interval_ms"]["share_over_50ms"] == 1.0
    assert d["any_hand_present"] == pytest.approx(0.6) and d["right_hand_only"] == pytest.approx(0.4)
    assert d["both_hands"] == pytest.approx(0.2) and d["left_hand_only"] == 0.0
    assert d["shoulder_width_over_frame_width_median"] == pytest.approx(0.2) and d["pose_present"] == 1.0


# ---------------------------------------------------------------- regressions from webcam session 2
def test_threaded_camera_returns_newest_frame_and_counts_skipped_frames():
    """Session 2: unthreaded reads waited for the next frame whenever a loop overran one camera
    period (43.5 ms vs 33.3 ms) -> every second frame skipped, invisibly. Threaded capture keeps the
    newest frame, never returns a frame twice, and counts the frames it had to skip."""
    import time as _time

    from asl.inference.evaluation import CameraSource

    class PacedCap:  # a 200 fps camera; frame value = frame number
        def __init__(self, n):
            self.i, self.n = 0, n

        def set(self, *a):
            return True

        def get(self, *a):
            return 200.0

        def read(self):
            _time.sleep(0.005)
            self.i += 1
            return (self.i <= self.n), np.full((4, 4, 3), self.i % 256, np.uint8)

        def release(self):
            pass

    src = CameraSource(0, capture=PacedCap(120), threaded=True)
    seen, times = [], []
    while True:
        ok, rgb, t = src.read()
        if not ok:
            break
        seen.append(int(rgb[0, 0, 0])), times.append(t)
        _time.sleep(0.015)  # slower consumer: ~3 camera frames per processed frame
    src.close()
    assert len(seen) > 10 and seen == sorted(seen) and len(set(seen)) == len(seen)  # newest, never repeated
    assert times == sorted(times) and src.dropped > 0
    assert src.captured == len(seen) + src.dropped  # every captured frame is either returned or counted


def test_cooldown_suppresses_continuations_but_is_off_by_default(rec):
    """A sign longer than max_segment_s is cut; its continuation starts one frame later. With
    cooldown_s > 0 the continuation is classified but not emitted; default 0 keeps old behaviour."""
    spans = [(0.5, False), (3.0, True), (1.0, False), (1.2, True), (1.0, False)]
    off = StreamingRecognizer(rec, 640, 480, max_segment_s=2.0)
    evs_off, _ = _stream(off, spans)
    on = StreamingRecognizer(rec, 640, 480, max_segment_s=2.0, cooldown_s=0.5)
    evs_on, _ = _stream(on, spans)
    assert [e.forced for e in evs_off] == [e.forced for e in evs_on] and evs_off[0].forced
    assert not any(e.suppressed for e in evs_off)                 # default: nothing suppressed
    cont = evs_on[1]                                               # the forced cut's continuation
    assert cont.suppressed and cont.emitted is None and cont.first_hand_s - evs_on[0].end_s < 0.5
    assert not evs_on[-1].suppressed and evs_on[-1].emitted is not None  # separate later sign still emitted
    with pytest.raises(ValueError):
        StreamingRecognizer(rec, 640, 480, cooldown_s=-1)


def test_first_hand_time_excludes_pre_roll(rec):
    sr = StreamingRecognizer(rec, 640, 480, pre_roll_s=0.5)
    evs, _ = _stream(sr, [(1.0, False), (1.5, True), (1.0, False)])
    e = evs[0]
    assert e.first_hand_s == pytest.approx(1.0, abs=1 / 30 + 1e-9)       # hands appear at 1.0 s
    assert e.start_s == pytest.approx(e.first_hand_s - 0.5, abs=1 / 30 + 1e-9)  # segment start includes pre-roll


def test_strict_scoring_excludes_segments_that_started_before_go():
    """Session 2: 13 of 17 answers had hands already up before GO (rest + sign mixed). Strict
    scoring answers with the first segment whose first hand frame is at/after GO; earlier segments are
    reported as pre_go; the previous first-event rule is kept as legacy_*."""
    from asl.inference.evaluation import Attempt, assign_events, attempt_row, summarize
    from asl.inference.recognizer import SignEvent

    def ev(label, first, last, suppressed=False):
        return SignEvent(Prediction("ok", label, 0, 0.7, ((label, 0.7),)), None if suppressed else label,
                         first - 0.5, last + 0.4, 40, False, last, first, suppressed)

    a0 = Attempt(0, "water", 3.0, 8.0, ready_start_s=0.0)
    a1 = Attempt(1, "red", 11.0, 16.0, ready_start_s=8.0)
    a2 = Attempt(2, "blue", 19.0, 24.0, ready_start_s=16.0)
    evs = [(ev("wet", 2.0, 4.0), 4.4), (ev("water", 5.0, 6.5), 6.9),     # a0: pre-GO then a clean answer
           (ev("red", 10.5, 12.0), 12.4),                                  # a1: only a pre-GO segment
           (ev("blue", 20.0, 21.0), 21.4), (ev("blue", 21.2, 22.0, suppressed=True), 22.4)]  # a2 + suppressed
    assign_events([a0, a1, a2], evs, 1.5)
    r0, r1, r2 = (attempt_row(a) for a in (a0, a1, a2))
    assert (r0["outcome"], r0["predicted"], r0["pre_go_events"], r0["legacy_outcome"]) == ("correct", "water", 1, "wrong")
    assert r1["outcome"] == "pre_go_only" and r1["legacy_outcome"] == "correct" and r1["started_before_go"]
    assert r2["outcome"] == "correct" and r2["duplicates"] == 0 and r2["suppressed_events"] == 1
    s = summarize([r0, r1, r2])
    assert s["accuracy_all_attempts"] == pytest.approx(2 / 3) and s["legacy_accuracy_all_attempts"] == pytest.approx(2 / 3)
    assert s["pre_go_events_total"] == 2 and s["suppressed_events_total"] == 1


def test_framing_advice_uses_training_band():
    from asl.inference.evaluation import framing_advice

    assert framing_advice(0.225).endswith("move closer")       # session 2
    assert framing_advice(0.371).endswith("OK") and framing_advice(0.5).endswith("move back")
    assert "not visible" in framing_advice(None) and "not visible" in framing_advice(float("nan"))


S02 = PROJECT_ROOT / "reports" / "step6" / "webcam_eval" / "20261019T075328Z_camera_S01"


@pytest.mark.skipif(not (S02 / "landmarks.npz").exists(), reason="session 2 landmarks not present")
def test_session2_replay_reproduces_live_events():
    """Live/offline parity on real webcam data: replaying the saved landmarks with the session's
    settings reproduces every recorded event (label, confidence, frames, timing)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("replay", PROJECT_ROOT / "scripts" / "s6_05_replay_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    events, _, attempts = mod.replay(S02, {})
    recorded = [json.loads(l) for l in (S02 / "events.jsonl").read_text().splitlines()]
    assert len(events) == len(recorded) == 39 and mod.matches_recorded(events, recorded)


# ---------------------------------------------------------------- regressions from session 3
def test_forced_cut_continuation_keeps_the_run_start(rec):
    """After a cut at max_segment_s with the hands still up, the next segment's first_hand_s is the
    frame after the cut; run_start_s keeps where the uninterrupted hand run began. A real rest
    (>= end_rest_s without hands) starts a new run."""
    sr = StreamingRecognizer(rec, 640, 480, max_segment_s=2.0)
    evs, _ = _stream(sr, [(0.5, False), (3.0, True), (1.0, False), (1.2, True), (1.0, False)])
    cut, cont, later = evs
    assert cut.forced and cut.run_start_s == cut.first_hand_s == pytest.approx(0.5, abs=1 / 30 + 1e-9)
    assert cont.first_hand_s > cut.end_s and cont.run_start_s == cut.run_start_s   # same signing run
    assert later.run_start_s == later.first_hand_s > cont.end_s + sr.end_rest_s    # new sign after a rest


def test_events_are_assigned_by_start_of_signing():
    """Session 3: a sign started inside a GO window but forced-cut past its end (and the tail after the
    cut) was keyed on its last hand frame and landed in the next prompt's WAIT phase - attempt 8 was
    scored 'missed' and 8 tails were reported as WAIT-phase signing. Session 2: a continuation of a hand
    run that began in WAIT was scored as a clean post-GO answer."""
    from asl.inference.evaluation import Attempt, assign_events, attempt_row
    from asl.inference.recognizer import SignEvent

    def ev(label, first, last, run=None, forced=False):
        return SignEvent(Prediction("ok", label, 0, 0.7, ((label, 0.7),)), label, first - 0.5, last + 0.4, 40,
                         forced, last, first, False, first if run is None else run)

    a0 = Attempt(0, "forbid", 3.0, 8.0, ready_start_s=0.0)
    a1 = Attempt(1, "here", 11.0, 16.0, ready_start_s=8.0)
    a2 = Attempt(2, "director", 19.0, 24.0, ready_start_s=16.0)
    evs = [(ev("church", 4.6, 8.6, forced=True), 8.7),         # started in a0's GO, ended in a1's WAIT
           (ev("church", 8.65, 9.5, run=4.6), 10.0),           # its tail after the cut
           (ev("here", 9.2, 10.4), 10.8),                       # genuinely signed during a1's WAIT
           (ev("here", 12.0, 13.5), 13.9),                      # a1's answer
           (ev("strong", 18.0, 22.0, forced=True), 22.1),       # a2: hands up since WAIT ...
           (ev("director", 22.05, 23.0, run=18.0), 23.4)]       # ... continuation after GO is not clean
    assert assign_events([a0, a1, a2], evs, 1.5) == []
    r0, r1, r2 = (attempt_row(a) for a in (a0, a1, a2))
    assert (r0["outcome"], r0["predicted"], r0["duplicates"]) == ("wrong", "church", 1)
    assert (r1["outcome"], r1["early_events"], r1["early_labels"]) == ("correct", 1, ["here"])
    assert r2["outcome"] == "pre_go_only" and r2["pre_go_events"] == 2 and r2["first_hand_s"] is None
    # events without a recorded first hand frame (older logs) keep the last-hand rule
    old = SignEvent(Prediction("ok", "x", 0, 0.7, (("x", 0.7),)), "x", 7.6, 9.0, 40, False, 8.6)
    a0b, a1b = Attempt(0, "x", 3.0, 8.0, ready_start_s=0.0), Attempt(1, "y", 11.0, 16.0, ready_start_s=8.0)
    assign_events([a0b, a1b], [(old, 9.0)], 1.5)
    assert a0b.events == [] and len(a1b.early_events) == 1


S03 = PROJECT_ROOT / "reports" / "step6" / "webcam_eval" / "20261019T084115Z_camera_S01"


@pytest.mark.skipif(not (S03 / "landmarks.npz").exists(), reason="session 3 landmarks not present")
def test_session3_replay_parity_and_rescoring():
    """Session 3 (threaded capture, gated GO): replay reproduces all 36 live events; re-scored by start
    of signing, attempt 8 is answered (wrong, 'church') instead of missed, the 9 forced-cut tails are
    duplicates of their own attempt, and only the 6 genuine WAIT-phase signs remain early events."""
    import importlib.util

    from asl.inference.evaluation import assign_events, attempt_row, summarize

    spec = importlib.util.spec_from_file_location("replay", PROJECT_ROOT / "scripts" / "s6_05_replay_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    events, _, attempts = mod.replay(S03, {})
    recorded = [json.loads(l) for l in (S03 / "events.jsonl").read_text().splitlines()]
    assert len(events) == len(recorded) == 36 and mod.matches_recorded(events, recorded)
    orphans = assign_events(attempts, events, 1.5)
    rows = [attempt_row(a) for a in attempts]
    s = summarize(rows)
    assert s["outcomes"] == {"correct": 13, "wrong": 7} and len(orphans) == 1   # the orphan: framing check
    assert (rows[8]["outcome"], rows[8]["predicted"]) == ("wrong", "church")
    assert s["early_events_total"] == 6 and s["attempts_with_duplicates"] == 9 and s["pre_go_events_total"] == 0


# ---------------------------------------------------------------- offline experiment scripts
def _load_script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_active_motion_crop_rule_and_no_overwrite(tmp_path, monkeypatch):
    """s6_06 (offline experiment): the initial rise and final drop faster than theta are removed, the
    no-crop control keeps the whole hand range, and an existing non-empty output folder is refused."""
    import math
    import sys

    m = _load_script("s6_06_analyze_active_motion_crop")
    vel = np.r_[np.full(5, -3.0), np.full(20, 0.2), np.full(4, 3.0)]   # rise, sign, drop (shoulder widths/s)
    hands = np.ones(31, bool)
    hands[0] = False                                                   # pre-roll frame without hands
    mo = {"a": 1, "b": 29, "vel": vel, "hands": hands}
    assert m.crop(mo, 1.0) == (6, 25)                    # first frame after the rise, last before the drop
    assert m.crop(mo, math.inf) == (1, 29)               # control: whole hand range
    assert m.crop(mo, 5.0) == (1, 29)                    # nothing faster than theta -> nothing removed
    assert m.sign_test(0, 2) == 0.5 and m.sign_test(0, 0) == 1.0
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("keep")
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["s6_06", "--out", "out"])
    with pytest.raises(SystemExit, match="not empty"):
        m.main()
    assert (tmp_path / "out" / "x").read_text() == "keep"
