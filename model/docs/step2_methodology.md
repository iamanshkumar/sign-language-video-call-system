# Step 2 — Landmark preprocessing methodology (v2, approved)

Status: implemented and tested; **full-dataset extraction not started** (planned on a cloud
machine, environment to be decided). Config: `configs/step2_landmarks.yaml`.

## Pipeline per video

1. Download the file at the pinned dataset revision; verify its SHA-256.
2. Decode every frame with PyAV and run **one MediaPipe Holistic pass** (0.10.21, CPU,
   `model_complexity=1`, 468 face landmarks). A fresh Holistic instance per video.
3. **Uniform temporal resampling to 30 FPS**: target frames = round(N / fps_timestamps × 30);
   landmarks are linearly interpolated between neighbouring source frames when the group is
   present in both (nearest frame otherwise). No zero-padding, no frame repetition.
4. **Full-clip metadata** (kept unchanged): original FPS, frame count, duration, resolution,
   per-group missing frames, both-hands-missing intervals `[start, length]`, and the
   verdict of the old v1 whole-clip rule (`fullclip_rule_rejection`, comparison only).
5. **Active signing interval** (below) — recorded in metadata and in the array file.
6. **Rejection rules** (below). Rejected videos keep their frozen-split assignment; only
   the landmark metadata records `rejected` and `rejection_reason`.
7. **Missing landmarks**: interior gaps ≤ 5 frames are linearly interpolated; longer gaps
   and gaps at the clip edges are zero-filled with presence 0.
8. **Normalization**: coordinates converted to pixels (x·W, y·H, z·W), then
   `(coordinate − shoulder_center) / shoulder_distance` (midpoint of pose landmarks 11/12;
   2-D distance). Frames without valid shoulders (visibility < 0.5) reuse the last valid
   frame; leading frames use the first valid frame; no valid frame → rejection.
9. **Storage** (format v2, below). All frames are stored, including rest frames.

## Active signing interval

On the 30 FPS grid, a frame *has a hand* if the left or right hand is detected (raw
detections, before interpolation). A **meaningful** detection is a run of ≥ 3 consecutive
hand frames (`active_interval_min_hand_run`). The interval spans from the first frame of the
first meaningful run to the last frame of the last meaningful run (inclusive); gaps and
isolated blips inside it are part of it.

## Rejection rules (in order)

| reason | rule |
|---|---|
| `decode_or_mediapipe_error` | file cannot be decoded / processed |
| `too_few_decoded_frames` | < 30 decoded frames in the whole clip |
| `no_active_signing_interval` | no meaningful hand detection |
| `active_both_hands_missing_fraction` | inside the interval, both hands missing in > 30 % of frames |
| `active_both_hands_missing_run` | inside the interval, both hands missing > 15 consecutive frames |
| `normalization_no_valid_shoulders` | shoulders never valid |
| `invalid_values_unrepairable` | NaN/Inf after repair (incl. storage-dtype overflow) |

Resting frames before/after the interval are never counted as missing.

**Short active intervals are flagged, not rejected.** `short_active_interval = True` in the
metadata when the interval has < 30 frames (`short_active_interval_frames`). Such clips are
accepted if they pass the rules above. (A 30-frame rejection rule was briefly adopted and
then reverted: on the benchmark it rejected 116 of 400 clips, and the audit below shows the
flag mostly tracks clip length rather than detection quality.)

## Storage format v2 (one `.npz` per accepted video)

| key | shape | dtype |
|---|---|---|
| `pose` | [T, 33, 3] | float32 |
| `face` | [T, 468, 3] | float16 |
| `left_hand`, `right_hand` | [T, 21, 3] | float16 |
| `pose_vis` | [T, 33] | float32 |
| `flags` | [T, 4] | uint8 bits per group: 1 observed, 2 present, 4 interpolated |
| `center`, `scale` | [T, 3], [T] | float32 (pixel space; allows de-normalization) |
| `shoulder_measured` | [T] | uint8 |
| `active_interval` | [2] | int32 (first, last frame; −1, −1 if none) |

Mixed precision was approved after validation on the benchmark (52 M values): pure float16
failed the pre-registered criteria (pose landmarks reach 28 shoulder widths); float16 for
face/hands with float32 pose passed overflow, zero-fill, max error (0.49 px < 0.5), and the
model-input checks (z-scored and Random-Forest statistic features ≤ 0.009 std < 0.01); it
misses only the strict 99.9th-percentile criterion (0.21 px vs 0.1 px). Details:
`reports/step2/methodology_v2_report.md`.

## Configurations (selected from the same file)

| configuration | groups | features per frame |
|---|---|---|
| Hands | left_hand, right_hand | 128 |
| Hands + Pose | + pose (incl. visibility) | 261 |
| Hands + Pose + Face | + face | 1666 |

Per group: x, y, z per landmark + 1 presence flag (+ visibility for pose).
`asl.landmarks.layout.assemble_features` builds `[T, F]` float32; only needed groups are read.

## Benchmark under the current rules (400 Phase A videos, re-evaluated from stored detections)

Accepted 374 (93.5 %), of which 111 are flagged `short_active_interval`;
`active_both_hands_missing_fraction` 13; `too_few_decoded_frames` 9;
`active_both_hands_missing_run` 3; `no_active_signing_interval` 1. Measured storage
0.316 MB per video (all groups, mixed precision) → ≈ 32.0 GB for the full dataset
(26.5 GB compressed); hands 2.6 GB, hands + pose 7.3 GB if stored alone.

## Short-interval audit (benchmark, `scripts/s2_05_short_interval_audit.py`)

- 111 / 374 accepted clips (29.7 %) are short; median 23 active frames (0.77 s); 7 have < 10.
- Strongly tied to clip length (Spearman ρ = 0.72): short in 83 % of clips < 45 frames and
  76 % of 45-59-frame clips, vs 1.7 % of clips ≥ 90 frames.
- Not associated with label type (letters / words / multi-word, χ² p = 0.17) or FPS band
  (p = 0.74); weak association with filename source (p = 0.011, approximate, small cells).
- Classes: the benchmark has ~1 accepted clip per class (357 classes, 15 with ≥ 2), so
  per-class rates are not reliable; 107 classes have only short benchmark clips. A
  dataset-wide per-class audit requires the full extraction.
- Very short intervals (3-9 frames with no missing frames inside) should be spot-checked
  visually: they may be detection failures during fast motion rather than quick signs.
  Report: `reports/step2/short_interval_audit.md`.

## Safeguards

Frozen Step 1 split read via `load_frozen_split` (checksum-verified), never modified;
`check_split_preserved` verifies split/word/path of every processed sample. Resumable,
atomic writes; `s2_02` refuses > 1,000 videos without `--approved-large-run`.
