# Step 5 — Subset vocabulary methodology

Status (updated 2026-11-14): `dev_100` was adopted. Its videos were extracted (Stage 2A, store
`data/landmarks/dev100_hp/`, hands + pose) and all training since has used it (Stage 2B and the E0–E3
studies in `docs/evaluation_protocol.md`). `vocabulary_expanded.json` (300 words) is still a proposal:
not extracted, not trained on. The original status line read: "vocabularies proposed (metadata only),
awaiting review; no video downloaded, no landmark extraction". Config: `configs/step5_vocabulary.yaml`. Code: `src/asl/data/vocabulary.py`,
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

## Stage 2B — dev_100 training experiments (controlled, one change at a time)

Common setup (unless the row says otherwise): store `dev100_hp` (4,838 accepted videos; frozen
split v1 → train 3,407 / val 725 / test 706; 100 classes, `label_mapping_dev_100.json`),
`hands_pose` features (F = 261), active interval resampled to W frames, 2 recurrent layers,
hidden 128, dropout 0.3, AdamW lr 1e-3 / weight decay 1e-4, cross-entropy, batch 64,
ReduceLROnPlateau (×0.5, patience 3) and early stopping (patience 7) on validation macro-F1,
best-validation checkpoint, seed 42, CPU 4 threads, deterministic. Test is evaluated once,
after the selected checkpoint is reloaded. Rows: `reports/step4/results/experiment_results.csv`;
artifacts: `experiments/dev100_hp/<experiment_id>/`.

| experiment_id | change vs. previous best | best / run epochs | train acc | val acc | val macro-F1 | test acc | test macro-F1 | train − val acc | train time |
|---|---|---|---|---|---|---|---|---|---|
| `hands_pose_w30_lstm` | baseline, max 50 epochs | 49 / 50 (cap) | 84.7 % | 54.3 % | 0.539 | 53.7 % | 0.528 | 30.3 pts | 183 s |
| `hands_pose_w30_lstm_e100` | max 100 epochs | 71 / 78 | 97.1 % | 60.3 % | 0.600 | 61.2 % | 0.612 | 36.8 pts | 294 s |
| `hands_pose_w30_lstm_e100_do05` | dropout 0.5 | 76 / 83 | 90.9 % | 60.4 % | 0.599 | 59.5 % | 0.586 | 30.5 pts | 312 s |
| `hands_pose_w45_lstm_e100` | window 45 | 63 / 70 | 85.1 % | 52.6 % | 0.518 | 50.7 % | 0.500 | 32.5 pts | 311 s |
| **`hands_pose_w30_bilstm_e100`** | **BiLSTM** (vs `lstm_e100`) | 50 / 57 | 99.9 % | **66.8 %** | **0.666** | **64.7 %** | **0.645** | 33.1 pts | 305 s |

### BiLSTM experiment (2026-10-19)

- Change: `--model bilstm` only; `config.json` of the run is identical to `hands_pose_w30_lstm_e100`
  except `spec.model`. Classifier input = forward state after the last step ‖ backward state after
  the first step = 2 × 128 = 256 → `Linear(256, 100)`; 821,348 parameters (LSTM: 345,188).
- Validity checks: same index (hash unchanged since it was built), same train/val/test sample ids
  and labels in both runs, labels = `label_mapping_dev_100.json`, no split leakage
  (`check_index_leakage`), preflight `[30, 261]` float32 finite for every sample, reloaded
  checkpoint reproduces validation macro-F1. Only environment difference: macOS 27.0 → 27.0.1;
  re-running the first 3 epochs of `lstm_e100` reproduced its history bit-for-bit.
- Result: +6.5 pts validation accuracy (+0.066 macro-F1) and +3.5 pts test accuracy (+0.033 macro-F1).
  Paired on the same clips: test — 67 clips right only with BiLSTM vs 42 only with LSTM (exact
  McNemar p = 0.021); validation — 86 vs 39 (p < 0.001). Per-class test F1 median 0.667 (LSTM 0.615),
  no class at 0.
- Overfitting is not solved: train accuracy 99.9 %, train − val gap 33.1 pts (LSTM 36.8), train −
  test 35.1 pts (LSTM 35.9). Converged: LR halved at epochs 46 and 55, early stop at 57.
- Cost: inference 1.75 ms per sample (LSTM 0.93 ms), still negligible next to MediaPipe. A BiLSTM
  needs the whole window before predicting (non-causal), which matches the planned window-based
  real-time inference.
- Conclusion: BiLSTM is the new best dev_100 configuration by the pre-registered selection metric
  (validation macro-F1), and test agrees. Single seed; the gap to 100 % train accuracy shows the
  remaining problem is generalization (data per class / signer variation), not capacity.
  `models/best_model.pt` was **not** replaced (it still holds the 50-epoch baseline); promote with
  `s4_02 ... --export-dir` only after review.

Reproduce:

```bash
cd model
PYTHONPATH=src .venv/bin/python scripts/s4_02_run_experiment.py --store dev100_hp --model bilstm \
    --feature-config hands_pose --window 30 --max-epochs 100 --tag e100 \
    --notes "BiLSTM only (vs hands_pose_w30_lstm_e100); everything else identical"
```
(`s4_02` refuses to overwrite an existing experiment directory; to re-run, use a new `--tag`.)

### Seed study: LSTM vs BiLSTM (2026-10-19)

Both architectures were trained with seeds 42 (existing runs), 1, 2, 3 and 4; the seed changes weight
initialisation, dropout masks and training shuffle order only (each run's `config.json` equals its
seed-42 run apart from `seed`; checked by the script). Every run selects its checkpoint on validation
macro-F1; the test split is evaluated once per run and never used for any choice. Full tables:
`reports/step5/seed_study.md` (generated by `scripts/s5_02_seed_study.py`).

| mean ± std (min–max), 5 seeds | LSTM | BiLSTM |
|---|---|---|
| validation macro-F1 | 0.594 ± 0.033 (0.562–0.640) | 0.662 ± 0.010 (0.651–0.673) |
| test accuracy | 57.5 ± 3.8 % (52.3–61.2) | 63.9 ± 1.0 % (62.2–64.7) |
| test macro-F1 | 0.570 ± 0.043 (0.514–0.612) | 0.634 ± 0.010 (0.618–0.644) |
| train accuracy | 92.4 ± 7.0 % | 99.1 ± 1.3 % |
| train − val accuracy gap | 32.4 ± 4.2 pts | 32.5 ± 1.1 pts |

- BiLSTM is better in all 5 seed pairs on validation macro-F1, test accuracy and test macro-F1
  (mean paired difference: +0.068 val macro-F1, +6.3 pts test accuracy, +0.064 test macro-F1) and is
  far less seed-sensitive (test-accuracy std 1.0 vs 3.8 pts). The single-seed comparison (61.2 % vs
  64.7 %) understated the gap: seed 42 was the best of the five LSTM runs.
- Statistics (limits stated in the report): exact two-sided sign test p = 0.0625 for every metric -
  the smallest value possible with 5 pairs, so it cannot reach 0.05; paired t-tests give p = 0.009–0.014
  but assume normal per-seed differences that 5 values cannot verify. Per-seed exact McNemar tests on
  the 706 test clips favour BiLSTM in every pair (p from 0.021 to 1e-8); they measure clip-level, not
  seed-level, variation. Conclusion: a consistent improvement on this split, not a proven
  significant one; the split is not signer-independent.
- Overfitting is unchanged: the train − validation gap is ~32 pts for both architectures.
- Run note: `hands_pose_w30_bilstm_e100_s3` spent 22 min (instead of ~12 s) in its post-training
  evaluation. Not reproducible: its checkpoint evaluates the full test split in 0.22 s like the other
  seeds, has no subnormal weights and reproduces its stored predictions; metrics are unaffected
  (training time 343 s is measured separately).

Candidate selection rule (validation only): the BiLSTM run with the highest validation macro-F1 -
`hands_pose_w30_bilstm_e100_s3` (0.673; test 64.0 % / 0.634, train 99.97 %). It is exported as a
candidate (`models/candidates/hands_pose_w30_bilstm_e100_s3/`), **not** promoted; see
`docs/step6_inference.md`.

Reproduce (from `model/`; each command refuses to overwrite an existing run):

```bash
for m in bilstm lstm; do for s in 1 2 3 4; do
  PYTHONPATH=src .venv/bin/python scripts/s4_02_run_experiment.py --store dev100_hp --model $m \
      --feature-config hands_pose --window 30 --max-epochs 100 --seed $s --tag e100_s$s
done; done
PYTHONPATH=src .venv/bin/python scripts/s5_02_seed_study.py
```
