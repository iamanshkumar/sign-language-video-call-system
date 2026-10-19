# Step 5 — Subset vocabulary methodology

Status: vocabularies proposed (metadata only), **awaiting review**; no video downloaded, no
landmark extraction. Config: `configs/step5_vocabulary.yaml`. Code: `src/asl/data/vocabulary.py`,
`scripts/s5_01_select_vocabulary.py`. Report: `reports/step5/vocabulary_report.md`.

## Why a subset

The full vocabulary has 2,207 classes with ~35 training clips each, 968 classes that share
byte-identical clips with another label (24 entirely) and 26 letter classes made mostly of
augmented copies. A 2,207-way result would mix "the model fails" with "the labels cannot be
told apart from the data". A clean subset gives a baseline where train → validation → test
can actually show whether landmarks + LSTM work.

## Rules (a class failing any rule is excluded; all reasons are recorded)

| rule | threshold | evidence used |
|---|---|---|
| no byte-identical clip under another label | 0 clips | SHA-256 (manifest) |
| no augmented copies | 0 `*_aug_<n>` files | split `aug_source` |
| single-token label, no label-audit anomaly | — | label string, `label_audit.csv` |
| independent validation / test units | ≥ 6 each | split groups (SHA-256 group + aug family) |
| balanced split | train share 60–80 % | frozen split v1 |
| within-class duplicates | ≤ 10 % | SHA-256 |
| one timestamp batch | ≤ 50 % of the class | `<word>_<date>_<time>_<n>.mp4` names |
| filename collections | ≥ 2 | `source_proxy` (proxy for recording diversity, not signers) |
| homonym labels | deferred | lexical judgement only — listed in the config, needs manual inspection |

Ranking of the remaining pool (descending): min(val units, test units) → number of filename
collections → independent units → `sha256("42:<word>")` (deterministic tie-break). The dev
set is the top 100; the expanded set is the top 300 and contains the dev set.

## What the selection does not do

- **No re-splitting.** Each class keeps its frozen split v1 assignment, so all Step 1 leakage
  guarantees hold unchanged.
- **No merged labels.** Merge candidates are only recommended (report §6).
- **No claim of visual distinguishability.** Metadata cannot show it; per-class confusions
  after training will.
- **Not signer-independent.** There are no signer IDs. `numericid-WORD` filenames may come from
  ASL Citizen, which publishes participant IDs — a hypothesis to check before claiming
  signer independence.

## Using a vocabulary later

The class `index` in `configs/vocabulary_*.json` is the position in Unicode code-point order
within the subset; `full_vocabulary_index` points to `data/sequences/label_mapping.json`.
Training on a subset needs a subset label mapping (num_classes = 100 / 300) built from these
files — the next stage.

## Stage 2A — dev_100 landmark store (hands + pose)

### Storage profile `hands_pose_f32`

`configs/step2_landmarks.yaml → storage_profiles.hands_pose_f32`. Only *what is written*
changes; MediaPipe Holistic 0.10.21 settings, 30 FPS resampling, rejection rules, gap
handling and shoulder normalization are the Step 2 v2 methodology unchanged.

- Stored groups: `pose`, `left_hand`, `right_hand` — **no face**. (Holistic still computes the
  face mesh internally; dropping it saves storage, not CPU time.)
- All landmark coordinates, `pose_vis`, `center`, `scale` are **float32**. The v2 default stored
  hands as float16; at 100 classes the saving would be ~250 bytes/frame (~0.1 GB), not worth
  any precision loss. `flags` stay uint8 bit fields, `active_interval` int32.
- Feature configurations available from this store: `hands` (128), `hands_pose` (261).
  `hands_pose_face` needs a store that kept the face (re-extraction, only if an experiment
  shows face landmarks help).

### Why one `.npz` per video is kept

Each file holds the whole clip at 30 FPS (variable length, rest frames included) plus its
active interval. Fixed-length sequences are produced when a sample is loaded
(`SequenceDataset` → `resample_interval`), so one stored copy serves every window (30/45/60)
and every feature configuration. Materialising fixed-length `.npy` files per window would
triple the data, freeze one resampling choice, and drop the frames real-time inference must
also see. Per-video files also make extraction resumable (one atomic write per video).

### Hands + pose feature vector (F = 261), verified on the generated arrays

| columns | contents |
|---|---|
| 0–62 | left hand (subject's left): 21 landmarks × (x, y, z), interleaved x0, y0, z0, x1, … |
| 63 | left hand present (0/1) |
| 64–126 | right hand: 21 × (x, y, z) |
| 127 | right hand present (0/1) |
| 128–226 | pose: 33 × (x, y, z) |
| 227 | pose present (0/1) |
| 228–260 | pose visibility (33), multiplied by pose present |

Coordinates: pixel space (x·W, y·H, z·W) minus the shoulder midpoint (pose 11/12), divided by
the 2-D shoulder distance. Absent group → its coordinates (and visibility) are 0 and its
presence column is 0; interior gaps ≤ 5 frames are interpolated and marked present.
`hands` (128) = columns 0–127.

### Commands

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem   # python.org Python 3.12 has no CA store; macOS bundle
# Test A: 8 classes x (4 train, 1 val, 1 test) = 48 videos
PYTHONPATH=src .venv/bin/python scripts/s2_02_extract_landmarks.py --vocabulary configs/vocabulary_100.json \
    --classes 8 --per-split train=4,val=1,test=1 --profile hands_pose_f32 --out-dir data/landmarks/dev100_hp --workers 4
PYTHONPATH=src .venv/bin/python scripts/s3_04_prepare_store.py --landmarks-dir data/landmarks/dev100_hp \
    --index-dir data/sequences/dev100_testA --vocabulary configs/vocabulary_100.json
PYTHONPATH=src .venv/bin/python scripts/s2_10_verify_store.py --landmarks-dir data/landmarks/dev100_hp \
    --index-dir data/sequences/dev100_testA --vocabulary configs/vocabulary_100.json --tag dev100_testA --visualize 4
# Test B: the whole vocabulary (resumes; the 48 Test A videos are skipped)
PYTHONPATH=src caffeinate -i .venv/bin/python scripts/s2_02_extract_landmarks.py --vocabulary configs/vocabulary_100.json \
    --profile hands_pose_f32 --out-dir data/landmarks/dev100_hp --workers 4 --approved-large-run
PYTHONPATH=src .venv/bin/python scripts/s3_04_prepare_store.py --landmarks-dir data/landmarks/dev100_hp \
    --index-dir data/sequences/dev100 --vocabulary configs/vocabulary_100.json
PYTHONPATH=src .venv/bin/python scripts/s2_10_verify_store.py --landmarks-dir data/landmarks/dev100_hp \
    --index-dir data/sequences/dev100 --vocabulary configs/vocabulary_100.json --tag dev100 --visualize 6
```

`--estimate-only` prints videos / download / output / CPU time / workers without processing.
The 100-class label mapping is written once to `data/sequences/label_mapping_dev_100.json`
(indices = `configs/vocabulary_100.json` indices).

### Results (2026-10-02, Apple M2, 4 workers)

| | Test A (8 classes) | Test B (all 100 classes) |
|---|---|---|
| videos discovered / with a final record | 48 / 48 | 5,131 / 5,131 |
| accepted (array written) | 45 | 4,838 (94.3 %) |
| rejected by Step 2 rules | 3 | 293: both hands missing > 30 % of the active interval 150, < 30 decoded frames 101, both hands missing > 15 frames in a row 36, no active signing interval 6 |
| download failures / decode or MediaPipe errors | 0 / 0 (after the CA-bundle fix) | 0 / 0 |
| raw download (streamed, deleted) | 21 MB | 2.30 GB |
| `.npz` total / mean / median | 4.0 MB / 88 KB / 78 KB | 432 MB / 89 KB / 83 KB (1,084 bytes per 30 FPS frame) |
| wall time | 61 s | 98 min for the last 4,133 videos (+ 26 min before an interruption); ~5.5 s per video per worker, 95 % of it MediaPipe |
| verification (`reports/step2/dev100*_verification.md`) | 7/7 checks pass + overlay | 7/7 checks pass + overlay |

Accepted per split: train 3,407 / val 725 / test 706; every class is present in every split
(minimum 26 train, 4 val, 4 test per class after rejections — `bald` has 4 val, one class has 4 test).
Inside the active intervals (206,425 frames): left hand only missing 35.2 % (mostly one-handed
signs by right-handed signers), right hand only missing 9.0 %, both hands missing 0.5 %, pose
missing 0. Normalized shoulder midpoint max |error| 0.0, shoulder distance max |error − 1|
3.2e-8; the 975 frames without valid shoulders reuse the reference frame as specified.
This verifies the video → landmarks → `.npz` → Dataset pipeline only; no model has been trained.
