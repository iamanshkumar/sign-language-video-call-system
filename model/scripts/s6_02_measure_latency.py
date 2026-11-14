"""Step 6.2: inference latency + live/offline parity on real test clips.

  SSL_CERT_FILE=/etc/ssl/cert.pem PYTHONPATH=src .venv/bin/python scripts/s6_02_measure_latency.py \
      --model-dir models/candidates/hands_pose_w30_bilstm_e100 --clips 20

1. Model only: one [30, 261] sequence, batch 1, after warm-up (asl.models.latency).
2. Real clips: N test-split clips of the model's store are downloaded to a temp dir (deleted at
   the end; the dataset is not re-downloaded). For each clip, every frame is run through MediaPipe
   Holistic (training settings; per-frame time measured), then:
   - offline parity: features from the inference path vs the stored training features of the same
     clip (SequenceDataset), and the prediction vs the experiment's stored test prediction;
   - streaming: frames are pushed into StreamingRecognizer with their timestamps; we time each
     push and the push that closes a segment (preprocessing + model).
Writes reports/step6/latency_<tag>.json and .md. The first clip's first 10 frames are warm-up.
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.inference import SignRecognizer, StreamingRecognizer
from asl.landmarks.extract import RawLandmarks, holistic_result_to_frame, make_holistic
from asl.landmarks.pipeline import hf_fetcher
from asl.landmarks.storage import available_configurations
from asl.models import latency as lat
from asl.sequences.dataset import SequenceDataset
from asl.sequences.labels import load_label_mapping

STEP3 = PROJECT_ROOT / "configs" / "step3_sequences.yaml"


def stats_ms(xs) -> dict:
    a = np.asarray(xs) * 1000.0
    return {"n": int(len(a)), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "max_ms": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    c1, c2, c3 = load_config(), load_yaml(STEP2_CONFIG), load_yaml(STEP3)
    inf = load_yaml(PROJECT_ROOT / "configs" / "inference.yaml")
    md = PROJECT_ROOT / args.model_dir
    rec = SignRecognizer.load(md / "best_model.pt", md / "class_mapping.json", c2, c3)
    info = rec.info
    torch.set_num_threads(4)
    hw = {"machine": platform.machine(), "processor": platform.processor(), "platform": platform.platform(),
          "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
          "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": "cpu"}

    # ---------------------------------------------------------------- 1. model only
    model_only = lat.model_latency(rec.model, rec.window, rec.feature_dim, 1, torch.device("cpu"), warmup=20, runs=500)
    api = lat.callable_latency(lambda: rec.predict_features(np.zeros((rec.window, rec.feature_dim), np.float32)),
                               1, 500, 20, "predict_features (validation + softmax + top-k), batch 1")

    # ---------------------------------------------------------------- 2. real clips
    st = c3["stores"][info["store"]]
    mapping = load_label_mapping(PROJECT_ROOT / st["label_mapping"])
    index = pd.read_parquet(PROJECT_ROOT / st["index_dir"] / "index.parquet")
    configs = available_configurations(c2["configurations"], info["preprocessing"]["stored_groups"])
    ds = SequenceDataset(index, PROJECT_ROOT / st["landmarks_dir"] / "arrays", "test", rec.window,
                         info["feature_config"], configs, mapping)
    stored_pred = pd.read_csv(PROJECT_ROOT / info["source_experiment"] / "predictions_test.csv").set_index("sample_id")
    rows = ds.rows.sample(n=min(args.clips, len(ds)), random_state=0)
    fetch = hf_fetcher(c1["dataset"]["repo_id"], c1["dataset"]["revision"], c1["probe"]["download_retries"])
    mp_frame, push_t, close_t, per_clip = [], [], [], []
    warm = 10
    with tempfile.TemporaryDirectory(dir=c1.path("raw_cache")) as tmp:
        import av
        for k, r in enumerate(rows.itertuples(index=False)):
            dest = Path(tmp) / f"{k}.mp4"
            got = fetch(r.repo_path, r.sha256, dest)
            if not got["download_ok"]:
                per_clip.append({"sample_id": r.sample_id, "error": got.get("download_error")})
                continue
            frames = []
            with av.open(str(dest)) as c:
                s = c.streams.video[0]
                fps = float(s.average_rate) if s.average_rate else 30.0
                w, h = s.codec_context.width, s.codec_context.height
                for i, fr in enumerate(c.decode(s)):
                    frames.append((fr.time if fr.time is not None else i / fps, fr.to_ndarray(format="rgb24")))
            dest.unlink()
            # MediaPipe per frame (fresh instance per clip, as in training)
            lm = []
            with make_holistic(c2["mediapipe"]) as hol:
                for i, (t, img) in enumerate(frames):
                    t0 = time.perf_counter()
                    res = hol.process(img)
                    dt = time.perf_counter() - t0
                    if not (k == 0 and i < warm):
                        mp_frame.append(dt)
                    lm.append((t, *holistic_result_to_frame(res)))
            # offline parity: whole clip through the inference path vs stored training features
            raw = RawLandmarks(np.array([x[0] for x in lm]), np.stack([x[1] for x in lm]), np.stack([x[2] for x in lm]),
                               np.stack([x[3] for x in lm]), int(sum(x[4] for x in lm)), w, h, "h264", fps, 0.0, 0.0)
            feats, stt = rec.features_from_raw(raw)
            i_ds = int(np.flatnonzero(ds.rows["sample_id"].to_numpy() == r.sample_id)[0])
            stored = ds[i_ds][0].numpy()
            p_off = rec.predict_raw(raw)
            # streaming: same landmarks pushed frame by frame
            sr = StreamingRecognizer(rec, w, h, **inf["segmentation"], **inf["postprocess"])
            events = []
            for t, cc, oo, vv, _ in lm:
                t0 = time.perf_counter()
                ev = sr.push(t, cc, oo, vv)
                dt = time.perf_counter() - t0
                (close_t if ev else push_t).append(dt)
                if ev:
                    events.append(ev)
            t0 = time.perf_counter()
            ev = sr.flush()
            if ev:
                close_t.append(time.perf_counter() - t0)
                events.append(ev)
            per_clip.append({
                "sample_id": r.sample_id, "word": r.word, "frames": len(frames), "fps": round(fps, 2),
                "features_identical_to_training": feats is not None and bool(np.array_equal(feats, stored)),
                "max_abs_feature_diff": None if feats is None else float(np.abs(feats - stored).max()),
                "offline_prediction": p_off.label, "stored_experiment_prediction": stored_pred.loc[r.sample_id, "pred_word"],
                "offline_equals_stored": p_off.label == stored_pred.loc[r.sample_id, "pred_word"],
                "streaming_segments": [{"label": e.prediction.label, "status": e.prediction.status,
                                        "confidence": e.prediction.confidence, "frames": e.frames,
                                        "reason": e.prediction.reason} for e in events]})
    ok = [p for p in per_clip if "error" not in p]
    seg_main = [p["streaming_segments"][0] if p["streaming_segments"] else None for p in ok]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": info["model"],
        "experiment_id": info.get("experiment_id"), "hardware": hw,
        "protocol": {"model_only": "random [1, 30, 261] input, 20 warm-up + 500 timed forward passes",
                     "mediapipe": f"Holistic {c2['mediapipe']['version']} model_complexity {c2['mediapipe']['model_complexity']}, "
                                  f"CPU, one process, per frame; first {warm} frames of the first clip excluded as warm-up",
                     "streaming": "StreamingRecognizer.push per frame on the clip's landmarks (MediaPipe time excluded); "
                                  "'segment close' = the push/flush that runs preprocessing + model",
                     "clips": f"{len(ok)} random dev100 test clips (seed 0), downloaded to a temp dir and deleted"},
        "model_only_forward": model_only, "predict_features_api": api,
        "mediapipe_per_frame": stats_ms(mp_frame), "push_per_frame_no_segment": stats_ms(push_t),
        "segment_close_preprocess_plus_model": stats_ms(close_t) if close_t else None,
        "mediapipe_max_fps_single_process": 1000.0 / stats_ms(mp_frame)["median_ms"],
        "parity": {"clips": len(ok), "features_identical_to_training": sum(p["features_identical_to_training"] for p in ok),
                   "offline_prediction_equals_experiment": sum(p["offline_equals_stored"] for p in ok),
                   "streaming_first_segment_equals_offline": sum(1 for p, s in zip(ok, seg_main) if s and s["label"] == p["offline_prediction"]),
                   "clips_with_one_segment": sum(1 for p in ok if len(p["streaming_segments"]) == 1),
                   "download_errors": len(per_clip) - len(ok)},
        "clips": per_clip}
    tag = args.tag or info.get("experiment_id", "model")
    out = PROJECT_ROOT / "reports" / "step6"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"latency_{tag}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("hardware", "model_only_forward", "predict_features_api", "mediapipe_per_frame",
                                               "push_per_frame_no_segment", "segment_close_preprocess_plus_model",
                                               "mediapipe_max_fps_single_process", "parity")}, indent=1, default=str))


if __name__ == "__main__":
    main()
