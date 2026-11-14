# Step 6 — Checkpoint export, promotion and real-time inference

Webcam evaluation procedure, simulation results, calibration and readiness: `docs/step6_webcam_evaluation.md`.
Application integration design (proposal): `docs/step7_integration_design.md`.

Status (updated 2026-11-14): candidate exported and validated; **not promoted**. `models/best_model.pt`
still holds the 50-epoch LSTM baseline, which was saved before the preprocessing contract existed and
has no `preprocessing` key, so `SignRecognizer.load("models")` (the default `model_dir` in
`configs/inference.yaml`) refuses it with `ModelLoadError`. Use
`--model-dir models/candidates/hands_pose_w30_bilstm_e100_s3` until the candidate is promoted with
`s4_05`. Inference module implemented, unit-tested and measured on recorded clips. Live webcam use has
been exercised through `s6_03_webcam_eval.py` (10 camera sessions, one signer, S01; see
`docs/step6_webcam_evaluation.md`). **Not integrated with the video-call app.** Isolated words only, no
sentence translation.

## 1. Export (no retraining) — `scripts/s4_04_export_model.py`

```bash
PYTHONPATH=src .venv/bin/python scripts/s4_04_export_model.py --experiment experiments/dev100_hp/hands_pose_w30_bilstm_e100_s3
```

Writes `models/candidates/<experiment_id>/{best_model.pt, class_mapping.json, export_report.json}`
(git-ignored). It refuses an existing candidate directory and only creates it if every check passes:

- before export: neural model selected on validation (`best.pt`), checkpoint reload reproduced
  val macro-F1, feature configuration stored in the store, groups and F = 261 match, class count
  and label-mapping hash match the store's mapping, window is a configured window, weights load
  strictly into the recorded architecture, checkpoint epoch = best epoch;
- after export: reloaded through the inference loader; logits identical to the source checkpoint
  on 64 real test sequences and 8 fixed random inputs; argmax on all 706 test sequences equal to the
  experiment's stored test predictions; `predict_features` agrees.

The exported `.pt` holds the weights, the architecture (model, hidden size, layers, dropout,
input dim, classes), window, feature groups, the label-mapping hash, source experiment, seed, and a
**preprocessing spec**: MediaPipe settings, 30 FPS resampling, rejection rules, gap handling,
storage profile and dtypes, feature configuration, feature-name hash, window, content region,
resampling. The loader refuses to run a model whose spec differs from the current configs.

Candidate `hands_pose_w30_bilstm_e100_s3` (BiLSTM, seed 3, best epoch 56): val 67.6 % / macro-F1 0.673,
test 64.0 % / 0.634. All 11 pre-export and 5 post-export checks passed (`export_report.json`).

## 2. Promotion (reversible) — `scripts/s4_05_promote_model.py`

```bash
# 1) inspect: prints current vs incoming SHA-256, changes nothing
PYTHONPATH=src .venv/bin/python scripts/s4_05_promote_model.py --candidate models/candidates/hands_pose_w30_bilstm_e100_s3 --dry-run
# 2) promote: backs up models/best_model.pt + class_mapping.json to models/backups/<UTC stamp>/ first
PYTHONPATH=src .venv/bin/python scripts/s4_05_promote_model.py --candidate models/candidates/hands_pose_w30_bilstm_e100_s3
# 3) undo (the command is printed by step 2)
PYTHONPATH=src .venv/bin/python scripts/s4_05_promote_model.py --restore models/backups/<stamp>
```

Only a candidate with `all_checks_pass` that loads through the inference loader is accepted; every
copy is checksum-verified and written atomically per file; a restore also backs up what it
replaces. The candidate's `class_mapping.json` is byte-identical to the tracked
`models/class_mapping.json`, so promotion changes only the (git-ignored) weights file.

## 3. Inference interface — `asl.inference`

```python
from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference import SignRecognizer, StreamingRecognizer

rec = SignRecognizer.load("models/candidates/hands_pose_w30_bilstm_e100_s3/best_model.pt",
                          "models/candidates/hands_pose_w30_bilstm_e100_s3/class_mapping.json",
                          load_yaml(STEP2_CONFIG), load_yaml(PROJECT_ROOT / "configs/step3_sequences.yaml"))
rec.predict_features(x)          # x: float [30, 261] built by the training pipeline -> Prediction
rec.predict_raw(raw)             # RawLandmarks of one clip -> Prediction (or status "rejected" + reason)
sr = StreamingRecognizer(rec, width, height)          # options: configs/inference.yaml
event = sr.push_rgb(frame_rgb, t_seconds)             # or sr.push(t, coords[543,3], observed[4], vis[33])
event = sr.flush()                                     # end of stream
# SignEvent: prediction, emitted, start_s, end_s, frames, forced, last_hand_s (end of signing)
```

- **Same representation as training.** Per frame: `holistic_result_to_frame` (moved out of the
  offline extractor, which now calls it too). Per segment: `preprocess_raw` → `resample_interval`
  (active interval → 30 frames) → `assemble_features` (`hands_pose`, 261 features in the documented
  order). Nothing is re-implemented. Verified: for 20 real test clips the inference path produced
  features **bit-identical** to the stored training features, and the same predictions as the
  experiment.
- **Segmentation (one prediction per sign).** A segment starts after `min_hand_run` (3, the training
  rule) consecutive frames with a hand and keeps `pre_roll_s` (0.5 s) of earlier frames; it ends after
  `end_rest_s` (0.4 s) without hands or is cut at `max_segment_s` (4 s). On the 20 test clips the
  streaming recognizer produced exactly one segment per clip, with the same label as the whole-clip
  prediction.
- **Safety.** `ModelLoadError` for missing/corrupt files, non-exported (legacy) checkpoints, class
  mapping or label-hash mismatch, preprocessing-spec mismatch, weights that do not fit the
  architecture. `InputError` for wrong shapes (e.g. 29 frames or 260 features), non-numeric data,
  NaN/Inf, non-increasing timestamps. A landmark group containing NaN/Inf is treated as not detected
  (same rule as offline extraction). Segments the training rules would reject (e.g. < 30 decoded
  frames, no active signing interval, hands missing too long) return `status="rejected"` with the
  training reason instead of a guess.
- **Optional post-processing** (`configs/inference.yaml`, off by default): confidence threshold
  (below → `status="low_confidence"`, not emitted) and majority-vote smoothing over the last N
  accepted predictions (emits a label once it has `min_votes`, never the same label twice in a row).
- CLI: `scripts/s6_01_run_inference.py --video clip.mp4 | --camera 0 [--model-dir ...]` prints one
  JSON line per segment. No saved run of `s6_01 --camera` exists; live camera capture has been run
  through `s6_03_webcam_eval.py` instead.

Known differences from training conditions: the live stream uses one Holistic instance for the
whole session (training used a fresh instance per clip); the segment boundaries come from the
segmenter rather than from the dataset's clip boundaries; a camera delivering fewer than ~30 frames
within a short segment can trigger the training `too_few_decoded_frames` rejection.

## 4. Latency (`scripts/s6_02_measure_latency.py`, `reports/step6/latency_hands_pose_w30_bilstm_e100_s3.json`)

Hardware: Apple M2 (8 cores, 8 GB), macOS 27.0.1, CPU only, PyTorch 2.14.1 with 4 threads, MediaPipe
Holistic 0.10.21 (model_complexity 1), one process, nothing else running.

| stage | protocol | median | p95 |
|---|---|---|---|
| model forward, batch 1 | random [1, 30, 261]; 20 warm-up + 500 timed runs | 1.68 ms | 1.80 ms |
| `predict_features` (validation + softmax + top-5) | same | 1.75 ms | 1.86 ms |
| MediaPipe Holistic per frame | 1,595 frames of 20 test clips; first 10 frames excluded as warm-up | 28.5 ms | 29.7 ms |
| `push` per frame (no segment end) | 1,586 frames | 0.007 ms | 0.009 ms |
| segment close: training preprocessing + model | 20 segments | 4.1 ms | 4.9 ms |

Reading: MediaPipe (~35 frames/s in a single process) dominates per-frame cost; the model is ~1.7 ms
per sign. A sign's label is available ≈ `end_rest_s` (0.4 s) + one frame + ~4 ms after the hands come
to rest. **Not measured:** live camera capture, frame decoding/transport inside the WebRTC call,
running alongside the browser/video call, and accuracy on live signers. These numbers alone do not
show real-time readiness of the full system.

Reproduce: `SSL_CERT_FILE=/etc/ssl/cert.pem PYTHONPATH=src .venv/bin/python scripts/s6_02_measure_latency.py --model-dir models/candidates/hands_pose_w30_bilstm_e100_s3 --clips 20`
(downloads 20 test clips, ~10 MB, to a temporary directory that is deleted afterwards).
