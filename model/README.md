# Sign Language Translator Video Calling System — ML pipeline

Isolated ASL **word** recognition from video (not sentence-level translation).

## Current status (2026-11-14)

The pipeline runs end to end on the 100-word development vocabulary (`dev_100`): data → landmarks →
sequences → training → export → streaming inference. It is **not connected to the video-call app**.

| Status | What |
|---|---|
| Implemented and tested | Steps 1–3 (manifest, frozen split, MediaPipe landmarks, sequences); Step 4 training (LSTM, BiLSTM, Random Forest); temporal augmentation and feature ablation; export/promotion; `asl.inference` (contract-checked loading, streaming segmentation, post-processing); E0–E3 evaluation and paper scripts. `pytest`: 202 passed, 3 skipped on a fresh clone (the skips need git-ignored local artifacts: the exported candidate and webcam recordings); 205 passed with them present. |
| Implemented, tested only through helpers | The CLIs `s6_01`–`s6_06` (no direct tests). `s6_03` has been run with a real webcam (10 sessions, 180 attempts, one signer, S01). |
| Partial | Deployment: the validated candidate `models/candidates/hands_pose_w30_bilstm_e100_s3/` has **not been promoted**. `models/best_model.pt` still holds the older 50-epoch LSTM, which predates the preprocessing contract and is refused by `SignRecognizer.load`, so the default `model_dir: models` in `configs/inference.yaml` does not load. Pass `--model-dir models/candidates/hands_pose_w30_bilstm_e100_s3`. E1 and E2 runs cannot be exported (ablations are refused by `s4_04`; E1 was never exported). |
| Prepared, not run | Cloud extraction of the full-vocabulary val/test splits (`docs/cloud_extraction_runbook.md`). |
| Planned, not implemented | Inference service (FastAPI/WebSocket), caption delivery over an `RTCDataChannel`, and any integration with the Next.js call app (`docs/step7_integration_design.md`). |

Latest saved results (dev_100, hands + pose, window 30; 5 seeds; sources in brackets):

- Exported candidate (BiLSTM, seed 3): test accuracy 0.640, macro-F1 0.634
  [`models/candidates/.../export_report.json`].
- Final one-time test, 5-seed means: BiLSTM baseline 0.639; E1 (+ temporal augmentation) 0.661;
  E2 (E1 + upper-body pose) 0.678 [`reports/paper/final_test/seed_mean.csv`]. Baseline test numbers were
  computed during development, before the protocol in `docs/evaluation_protocol.md` was frozen.
- Webcam replay set, one signer, 10 of the 100 signs, 92 core attempts: candidate top-1 0.402, top-5 0.587
  [`reports/e0/webcam_replay_set/summary_by_model.csv`].
- The models fit the training split almost perfectly (candidate train accuracy 0.9997), and the split is
  not signer-independent.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Python 3.12 is required (MediaPipe, needed in Step 2, has no 3.14 wheels).

## Step 1 pipeline

Run in order from the project root (`.venv/bin/python scripts/...`):

| Script | What it does | Outputs |
|---|---|---|
| `s1_01_build_manifest.py` | Lists the HF repo at the pinned commit, reconciles it with `Aslense Dataset.csv`, audits SHA-256 duplicate content and labels. No video download. | `data/metadata/manifest.parquet`, `manifest_summary.json`, `duplicate_content_groups.csv`, `label_audit.csv` |
| `s1_02a_select_eda_sample.py` | Seeded class-balanced sample (5,000 videos, ≥2 per class) for Representative Sample EDA. | `data/metadata/eda_sample/sample.csv`, `sample_class_counts.csv` |
| `s1_02b_probe_eda_sample.py` | Probes **only the sample**: batch download → SHA-256 check → full PyAV decode → delete. Resumable; retries HTTP 429. | `data/metadata/eda_sample/probe_parts/`, `sample_probe_results.parquet` |
| `s1_03_eda_report.py` | Exact metadata statistics + Representative Sample EDA (video properties), figures, report. | `reports/step1/` |
| `s1_04_make_split.py` | Creates the group-aware stratified 70/15/15 split from the full manifest **once** and freezes it (refuses to overwrite). | `data/splits/split_v1*.csv`, `split_v1.lock.json` |
| `s1_05_verify_split.py` | Checksum, leakage, reproducibility and stratification checks. Non-zero exit on failure. | `reports/step1/split_verification.json` |

Tests: `.venv/bin/python -m pytest`.

Configuration: `configs/step1_data.yaml` (dataset revision, seed, ratios, probe settings).

## Rules applied in Step 1

The split contains every reconciled file (CSV duplicate row and missing file
excluded). The indivisible split unit is a SHA-256 duplicate group joined with
augmentation families (`<source>.mp4` + `<source>_<word>_aug_<n>.mp4`).
Video-level rules (undecodable, < 30 decoded frames) and landmark rules are
applied in Step 2 as flags on the frozen split, never by re-splitting.

`data/metadata/probe_parts/` holds a partial (16,800-file, part_1/part_10 only)
full-dataset probe that was abandoned; it is not used for statistics.

## Step 2 (Phase A: benchmark only)

MediaPipe **0.10.21** (legacy `solutions.holistic`, CPU) is pinned: mediapipe 1.0.1's
Tasks API aborts on this Mac ("Metal service unavailable", even with the CPU delegate).
Model for reference/Tasks API is not needed by the pipeline; `models/` is git-ignored.

| Script | What it does |
|---|---|
| `s2_01_select_benchmark.py` | Seeded benchmark sample from the frozen TRAIN split → `data/landmarks/benchmark/benchmark_sample.csv` |
| `s2_02_extract_landmarks.py` | Resumable Holistic extraction (download → 30 FPS → rules → gap fill → shoulder normalization → `.npz`) |
| `s2_03_benchmark_report.py` | Phase A (v1) throughput/rejection/storage report → `reports/step2/benchmark_report.md` (historical) |
| `s2_04_reevaluate_benchmark.py` | Methodology v2 on the stored Phase A extractions: active-interval rules, storage variants, float16 validation → `reports/step2/methodology_v2_report.md` |
| `s2_05_short_interval_audit.py` | Class/category audit of short active intervals on the benchmark → `reports/step2/short_interval_audit.md` |

Methodology v2 (approved; full description in `docs/step2_methodology.md`): hand rules
are evaluated only inside the **active signing interval** (first → last run of ≥3
consecutive frames with a detected hand); intervals shorter than 30 frames are only
flagged (`short_active_interval`), not rejected. Rest frames are kept in the arrays and full-clip
statistics are preserved in the metadata. Storage: float16 face/hands, float32 pose,
uint8 presence flags. `s2_02` refuses
runs over 1,000 videos without `--approved-large-run`; `--from-split` reads the frozen split.

Configuration: `configs/step2_landmarks.yaml`. One Holistic pass stores all groups once;
`hands`, `hands_pose`, `hands_pose_face` are feature selections (`asl.landmarks.layout.assemble_features`).
Rejected videos stay in the frozen split; their status is in the landmark metadata only.

## Step 3 (sequence preparation)

Methodology: `docs/step3_methodology.md`. Config: `configs/step3_sequences.yaml`.
Each accepted video's **active signing interval** is uniformly resampled (linear
interpolation, endpoints included) to exactly 30 / 45 / 60 frames, per configuration
(`hands` 128, `hands_pose` 261, `hands_pose_face` 1666 features). Sequences are generated
lazily from the landmark `.npz` files via a per-store `index.parquet`; nothing is copied.

| Script | What it does |
|---|---|
| `s2_06_convert_benchmark_to_v2.py` | Re-packs the Phase A benchmark extractions into the approved v2 mixed-precision format → `data/landmarks/benchmark_v2_mixed/` (no MediaPipe) |
| `s3_01_build_label_mapping.py` | Label mapping from the frozen split vocabulary → `data/sequences/label_mapping.json` |
| `s3_02_build_sequence_index.py` | Sequence index (split from the frozen split, leakage checks) → `data/sequences/<store>/index.parquet` |
| `s3_03_sequence_report.py` | End-to-end Dataset/DataLoader checks + `reports/step3/sequence_preparation_report.md` |

```python
from asl.sequences.dataset import SequenceDataset, make_dataloaders
ds = SequenceDataset(index, arrays_dir, "train", 45, "hands_pose", configurations, label_mapping)
x, y = ds[0]          # float32 [45, 261], int64
```

Environment note: something on this Mac periodically sets the macOS `hidden` flag on
files in `.venv/.../site-packages`, and Python 3.12.13 then skips the editable-install
`.pth`, so `import asl` fails. Tests set `pythonpath = ["src"]`; run scripts with
`PYTHONPATH=src .venv/bin/python scripts/...`.

## Step 4 (training and evaluation infrastructure)

Methodology: `docs/step4_methodology.md`. Config: `configs/step4_training.yaml`.
LSTM / BiLSTM (2 × 128, dropout 0.30, AdamW 1e-3 / 1e-4, ≤ 50 epochs by default, early stopping 7,
ReduceLROnPlateau 0.5 / 3, selection on validation macro-F1) and a Random Forest on
temporal statistics (mean / std / min / max). Test data is evaluated only after the
selected checkpoint is reloaded. The planned 18-experiment matrix on the full 2,207-word vocabulary
was **not run**; development moved to the `dev_100` vocabulary (Step 5). The dev_100 runs
(`experiments/dev100_hp/`, `reports/step4/results/experiment_results.csv`) mostly use a 100-epoch
budget (`--max-epochs 100`, tag `e100`); every e100 run stopped early. `reports/step4/experiment_summary.md` and
`reports/step4/results/experiments.json` were generated on 2026-10-02, before those runs, and are out of
date; `experiment_results.csv` is current.

`s4_02_run_experiment.py` options added for the dev_100 studies: `--seed`, `--augment temporal`
(training-only, `src/asl/sequences/augment.py`), `--drop-pose-landmarks` (feature ablation; such runs
cannot be exported) and `--no-test-eval` (test is kept for final candidates).

| Script | What it does |
|---|---|
| `s4_01_smoke_test.py` | Synthetic end-to-end smoke test (2,207-way models) → `reports/step4/smoke/` |
| `s4_02_run_experiment.py` | Runs exactly ONE experiment (`--model --feature-config --window --store`) |
| `s4_03_report.py` | Summary, curves, confusion figures → `reports/step4/` |

## Step 5 (subset vocabulary — dev_100 adopted)

Methodology: `docs/step5_vocabulary.md`. Config: `configs/step5_vocabulary.yaml`. The selection uses
metadata only; the dev_100 videos were then extracted (hands + pose) and used for all training since.
`vocabulary_expanded.json` (300 words) has not been extracted or trained on.

| Script | What it does |
|---|---|
| `s5_01_select_vocabulary.py` | Duplicate/problem-class analysis and rule-based selection → `configs/vocabulary_100.json`, `configs/vocabulary_expanded.json` (300, superset), `reports/step5/` |
| `s2_02_extract_landmarks.py --vocabulary … --profile hands_pose_f32` | Extract only the vocabulary's videos, hands + pose, float32 (no face) → `data/landmarks/dev100_hp/` |
| `s3_04_prepare_store.py --vocabulary …` | Sequence index with the 100-class mapping `data/sequences/label_mapping_dev_100.json` |
| `s2_10_verify_store.py` | Integrity, feature layout (F = 261), missing landmarks, normalization, Dataset windows, splits, storage, landmark overlay → `reports/step2/<tag>_verification.md` |

Exact commands and the hands + pose feature layout: `docs/step5_vocabulary.md` (Stage 2A).
Training experiments and the LSTM/BiLSTM seed study: `docs/step5_vocabulary.md` (Stage 2B),
`scripts/s5_02_seed_study.py` → `reports/step5/seed_study.md`.

## Step 6 (export, promotion, inference)

Methodology and measurements: `docs/step6_inference.md`. Config: `configs/inference.yaml`.

| Script | What it does |
|---|---|
| `s4_04_export_model.py` | Export a trained experiment as a validated candidate (`models/candidates/<id>/`), no retraining |
| `s4_05_promote_model.py` | Reversible promotion of a candidate to `models/best_model.pt` (backup + restore) |
| `s6_01_run_inference.py` | Streaming isolated-sign recognition on a video file or webcam |
| `s6_02_measure_latency.py` | Model / MediaPipe / segment latency and live-vs-training feature parity on test clips |
| `s6_03_webcam_eval.py` | Guided webcam evaluation with a human signer (or `--video` simulation replay) → `reports/step6/webcam_eval/` |
| `s6_05_replay_session.py` | Replay a recorded session's landmarks offline (parity check, re-scoring, setting variants) → `<session>/replay_<tag>/` |
| `s6_04_confidence_calibration.py` | Calibration (ECE) and confidence-threshold coverage on validation/test → `reports/step6/calibration_*.json` |

Webcam evaluation procedure, session analyses and readiness: `docs/step6_webcam_evaluation.md`;
integration design (proposal, not implemented): `docs/step7_integration_design.md`.

Promotion status: the candidate has not been promoted (see "Current status"). Until it is, run
inference with `--model-dir models/candidates/hands_pose_w30_bilstm_e100_s3`.

## Evaluation protocol and paper scripts (E0–E3)

Protocol: `docs/evaluation_protocol.md` (frozen 2026-10-23). Scripts read saved predictions and
sessions; they do not train.

| Script | What it does |
|---|---|
| `e0_01_bootstrap_ci.py`, `e0_02_webcam_replay_set.py`, `e0_03_robustness.py` | Confidence intervals, frozen webcam replay set, robustness perturbations → `reports/e0/` |
| `e1_01_compare.py`, `e2_01_compare.py` | Temporal augmentation (E1) and upper-body pose (E2) decisions → `reports/e1/`, `reports/e2/` |
| `e3_01_final_test.py` | One-time final test of baseline → E1 → E2 → `reports/paper/final_test/` |
| `pp_01_postprocessing_replay.py`, `pp_02_hands_only_compare.py` | Post-processing replay on webcam sessions; hands-only comparison → `reports/paper/` |
| `paper_01`–`paper_05` | Paper tables, pipeline figure, claim verification, webcam details, precision/recall |

```python
from asl.inference import SignRecognizer, StreamingRecognizer
```

With the python.org Python 3.12 build, downloads need `export SSL_CERT_FILE=/etc/ssl/cert.pem`.

## Remote extraction (full-vocabulary val/test) — prepared, not yet run

The dev_100 store (train, val and test) was extracted locally. The tooling below is for the full
2,207-word splits and has only been exercised in a dry run (`data/landmarks/runs/dryrun_valtest_m2/`).

Runbook: `docs/cloud_extraction_runbook.md`. `s2_07_prepare_extraction.py` (plan + shards +
fingerprint), `s2_08_extract_shard.py` (one shard, refuses on fingerprint mismatch),
`s2_09_merge_verify.py` (merge + integrity), `s3_04_prepare_store.py` (Step 3 for any store),
`scripts/cloud/` (setup, shard runner, bundle). Needs Linux x86_64 (no MediaPipe ARM wheel).

## Using the split in later steps

```python
from asl.config import load_config
from asl.data.split import load_frozen_split
split = load_frozen_split(load_config().path("splits"), "v1")  # verifies the checksum
```

## Known limitations

- Isolated words only; no continuous signing.
- No signer IDs → the split is not signer-independent.
- Many labels are synonyms that share byte-identical videos (see the EDA report).
- Only byte-identical duplicates are detected; near-duplicates are not.
- Webcam evidence comes from one signer and 10 signs; recognition quality for other signers is unknown.
- No inference service or video-call integration exists yet.
