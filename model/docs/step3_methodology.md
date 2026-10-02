# Step 3 — Sequence preparation methodology

Status: implemented and tested on the converted benchmark store and synthetic fixtures.
No model training, no full-dataset extraction. Config: `configs/step3_sequences.yaml`.

## 1. Inputs (what Step 2 provides)

Inspected from `docs/step2_methodology.md` and `src/asl/landmarks/`:

| item | representation |
|---|---|
| landmark store | one `.npz` per **accepted** video (format v2): `pose` [T,33,3] float32, `face` [T,468,3] float16, `left_hand`/`right_hand` [T,21,3] float16, `pose_vis` [T,33] float32, `flags` [T,4] uint8 (1 observed, 2 present, 4 interpolated), `center`, `scale`, `shoulder_measured`, `active_interval` [2] int32 |
| time axis | all T frames of the clip on the **30 FPS grid**, including rest frames before/after the sign |
| `active_interval` | `[start, end]`, **inclusive** frame indices into that 30 FPS grid; `(-1, -1)` if none (such videos are rejected, so never stored). Also in metadata as `active_start_frame` / `active_end_frame`; length L = end − start + 1 |
| rejected videos | metadata row with `rejected=True`, `rejection_reason`, no array file; they keep their frozen-split assignment |
| short intervals | `short_active_interval=True` when L < 30; **accepted** (flag only) |
| configurations | `hands` (128 features/frame), `hands_pose` (261), `hands_pose_face` (1666), built by `asl.landmarks.layout.assemble_features` (x, y, z per landmark + 1 presence flag per group + pose visibility) |

## 2. Content region

The temporal content of a sample is its **active signing interval** `[start, end]`.
Rest frames outside it (hands at rest, out of the frame) are not used for sequences; they
remain in the stored arrays.

## 3. Sequence construction (one sequence per video, per window, per configuration)

For a window W ∈ {30, 45, 60}, the active interval is **uniformly resampled** to exactly W
frames on the temporal axis:

    p_k = start + k · (L − 1) / (W − 1),   k = 0 … W − 1      (first and last active frame included)

For each landmark group at position p_k with neighbours i = ⌊p_k⌋, j = i + 1, a = p_k − i:

- group PRESENT in both i and j → linear interpolation `(1 − a)·x_i + a·x_j` (coordinates
  and pose visibility), presence 1;
- otherwise → value and presence of the nearest frame (j if a ≥ 0.5, else i).

This is the same rule Step 2 uses for 30 FPS resampling. Integer positions reproduce the
stored frame exactly. Values are computed in float32 from the mixed-precision storage
(float16 → float32 cast before interpolation). The `interpolated`/`observed` bits follow
the nearest frame. Feature order and semantics are unchanged: the resampled arrays are
passed to `assemble_features`.

| active interval L | W = 30 | W = 45 | W = 60 |
|---|---|---|---|
| L ≥ 60 | downsample | downsample | downsample (identity if L = 60) |
| 45 ≤ L ≤ 59 | downsample | downsample (identity if 45) | upsample |
| 30 ≤ L ≤ 44 | downsample (identity if 30) | upsample | upsample |
| L < 30 (accepted, `short_active_interval`) | upsample | upsample | upsample |
| rejected in Step 2 | **excluded** (no array; counted in the report with its reason) | | |

- **No zero-padding, no random selection, no frame repetition.** Upsampling places new
  samples *between* stored frames and interpolates them; it never copies a frame. (Where a
  group is absent in a neighbouring frame, the nearest frame's value/presence is used for
  that group only — the Step 2 semantics of a zero-filled, presence-0 gap are preserved.)
- **Short active intervals are kept** and upsampled. The stretch factor W / L is recorded
  per sample (max 60 / 3 = 20× on the benchmark) and reported. This is temporal
  resampling of real landmark trajectories, not frame duplication — but for very short
  intervals most output frames are interpolated (see §8).
- Every accepted video with L ≥ 2 can produce every window (Step 2 guarantees L ≥ 3), so
  no accepted sample is unable to produce a window.

## 4. Labels

- Vocabulary = all distinct `word` values of the **frozen Step 1 split** (2,207 classes),
  sorted by Unicode code point; index = position in that order (0 … 2206).
- Built once from the frozen split, never from a split subset or from accepted samples;
  classes with no accepted sample keep their index.
- Artifact: `data/sequences/label_mapping.json` (mapping, vocabulary SHA-256, frozen-split
  checksum). The builder refuses to overwrite a different mapping.
- Original string labels are kept in the sequence index next to the integer.

## 5. Storage

No sequence arrays are materialized. A **sequence index** (`data/sequences/<store>/index.parquet`)
has one row per accepted video: sample id, word, label index, split (from the frozen
split), duplicate/augmentation group ids, array file, active interval, L, and per-window
stretch factors. Sequences are generated lazily from the landmark `.npz` files (only the
groups needed by the configuration are read). `SequenceDataset(preload=True)` can cache
generated sequences in RAM for small configurations. One storage copy of each video
serves all 3 windows × 3 configurations.

## 6. Dataset / DataLoader

`SequenceDataset(index, split, window, configuration, label_mapping)` returns
`(sequence float32 [W, F], label int64)`; `dataset.metadata(i)` returns sample id, word,
split, window, configuration, source array file and active interval. DataLoaders: batch
size, workers and seed from config; train shuffled with a seeded `torch.Generator`;
validation/test never shuffled; no augmentation.

## 7. Split and leakage

The frozen Step 1 split is authoritative: each sequence takes the split of its source
video from the frozen split file (the landmark metadata must agree, otherwise the build
fails). Only one sequence per video per window exists, and a Dataset is built for one
split, so windows of one video cannot be spread across splits. The index build fails if a
sample id, a SHA-256 duplicate group or a split group (duplicate + augmentation family)
appears in more than one split, or if a label is unknown.

## 8. Decisions requiring approval

1. **Time normalization (main decision).** Resampling the active interval to W frames
   makes every training sequence span the whole sign regardless of its real duration.
   W is therefore a *temporal resolution*, not a fixed real-time duration (a 0.4 s sign and
   a 3 s sign both become W frames). This follows the original plan ("uniformly sample the
   required number of frames across its temporal duration"), but it differs from the
   real-time sliding windows planned for streaming inference (30 frames = 1 s at 30 FPS),
   so train-time and stream-time inputs will have different speed statistics.
   *Alternative B:* fixed real-time windows of W stored frames centred on the active
   interval (cropping long signs, extending into real rest frames for short ones;
   resampling only when the whole clip is shorter than W).
2. **Extreme upsampling of short intervals.** Kept as decided in Step 2; an optional cap on
   the stretch factor (or a minimum L per window) could be added later — not applied now.
