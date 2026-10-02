# Sign Language Translator Video Calling System — ML pipeline

Isolated ASL **word** recognition from video (not sentence-level translation).
Current status: **Step 1 — Dataset + EDA** (validation, EDA, frozen split).

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
LSTM / BiLSTM (2 × 128, dropout 0.30, AdamW 1e-3 / 1e-4, ≤ 50 epochs, early stopping 7,
ReduceLROnPlateau 0.5 / 3, selection on validation macro-F1) and a Random Forest on
temporal statistics (mean / std / min / max). Test data is evaluated only after the
selected checkpoint is reloaded. **The 18-experiment matrix has not been run.**

| Script | What it does |
|---|---|
| `s4_01_smoke_test.py` | Synthetic end-to-end smoke test (2,207-way models) → `reports/step4/smoke/` |
| `s4_02_run_experiment.py` | Runs exactly ONE experiment (`--model --feature-config --window --store`) |
| `s4_03_report.py` | Summary, curves, confusion figures → `reports/step4/` |

## Remote extraction (val/test) — prepared, not yet run

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
