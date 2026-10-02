# Step 4 — Model training and evaluation methodology

Status: infrastructure implemented and tested; smoke test (synthetic) and one train-only
benchmark pipeline check completed. **The experiment matrix has not been run.**
Config: `configs/step4_training.yaml`. Code: `src/asl/models/`. Scripts: `scripts/s4_0*.py`.

## 1. Inputs (unchanged from Steps 1–3)

Sequences come from Step 3 exactly as produced (`SequenceDataset`): the active signing
interval uniformly resampled to W ∈ {30, 45, 60} frames, features per configuration
`hands` 128, `hands_pose` 261, `hands_pose_face` 1666. Labels: the Step 3 mapping
(2,207 classes, `data/sequences/label_mapping.json`); no new encoder. Splits: the frozen
Step 1 split via the Step 3 index. Short active intervals are used as provided; their
active length and stretch factor (W / L) are written next to every prediction.

## 2. Experiment matrix

18 neural experiments = 3 feature configurations × 3 windows × {LSTM, BiLSTM}, plus the
Random Forest baseline. IDs are deterministic: `<feature_config>_w<W>_<model>`, e.g.
`hands_w30_lstm`, `hands_pose_face_w60_bilstm`, `hands_w45_rf`. The RF is implemented
for every configuration × window (9 runs) because its statistics are computed from the
W-frame sequences; **whether to run all 9 or one per configuration needs approval.**

## 3. Random Forest baseline (temporal statistics)

For a sequence X ∈ ℝ^{W×F}: per feature f, mean, population standard deviation (ddof = 0),
minimum and maximum over the W frames, concatenated as [means | stds | mins | maxs] → 4F
values (hands 512, hands + pose 1044, hands + pose + face 6664). The dimension does not
depend on W. Presence columns are included (their mean is the fraction of frames in which
the group is present). Nothing is fitted for this transformation, so no information can
flow from validation/test. `RandomForestClassifier(n_estimators=300, max_depth=None,
min_samples_leaf=1, max_features="sqrt", class_weight=None, n_jobs=1, random_state=42)`,
trained on TRAIN only, evaluated on VALIDATION and then TEST. No hyperparameter search.

## 4. LSTM and BiLSTM

Input [B, T, F] → `nn.LSTM(F, 128, num_layers=2, dropout=0.30, batch_first=True)` →
representation → dropout 0.30 → `Linear(·, 2207)` → logits (no softmax).

- **LSTM** (unidirectional): representation = final hidden state of the top layer (128).
- **BiLSTM** (`bidirectional=True`): concatenation of the top layer's forward state after
  the last step and backward state after the first step (2 × 128 = 256), so the
  classifier input is 256.

Parameters (measured): hands 0.55 M (LSTM) / 1.23 M (BiLSTM); hands + pose 0.62 M / 1.36 M;
hands + pose + face 1.34 M / 2.80 M. The 2,207-way output layer is 0.28 M (LSTM) / 0.57 M (BiLSTM).

## 5. Training and model selection

AdamW (lr 1e-3, weight decay 1e-4), CrossEntropyLoss, batch size 64 (32 only if the
device runs out of memory; recorded), at most 50 epochs. After every epoch: validation
loss and macro-F1. `ReduceLROnPlateau(mode=max, factor=0.5, patience=3)` steps on
validation macro-F1 (**the monitored metric was not specified by the project — approval
needed**). The checkpoint (`best.pt`) is saved whenever validation macro-F1 strictly
improves (ties keep the earlier epoch); training stops after 7 epochs without improvement.
History per epoch: learning rate, training loss, training macro-F1 (from the
training-mode predictions made during the epoch), validation loss, validation macro-F1.
No feature scaling beyond Step 2's shoulder normalization; no augmentation.

## 6. Test procedure

The training function never receives the test data. After training ends, the selected
checkpoint is **reloaded from disk**, validation metrics are recomputed (and must equal
the selection value), and only then is the test split evaluated — once. The test split is
never used for early stopping, scheduling, checkpointing, preprocessing or choosing a
configuration. Choosing the best of the 18 configurations must use validation results.

A store without validation data is refused unless `--allow-no-validation` is given
explicitly; such a run has no model selection (the final epoch is saved as `final.pt`)
and reports **train-fit** metrics only, labelled as not measuring generalization.

## 7. Metrics

Accuracy, macro and weighted precision / recall / F1 (`zero_division=0`), correct and
incorrect counts, per-class precision / recall / F1 / support, top confusions, and the
full 2,207 × 2,207 confusion matrix (sparse `.npz`; figures show the evaluated classes).
Macro averages are over the classes occurring in y_true or y_pred (sklearn default);
classes absent from both have undefined precision/recall. The vocabulary size (2,207),
the number of classes in y_true and the number evaluated are recorded with every result.

## 8. Latency

Measured separately from training, after 20 warm-up runs, over 200 timed runs, on the
experiment's device (CPU here; `torch.cuda`/`torch.mps` synchronization when used):

1. **Model inference**: forward pass on an already available tensor, batch 1 (single
   sample) and batch 64 (batched); mean, median, p95 latency and throughput (samples/s).
2. **End-to-end without MediaPipe**: per sample, load the `.npz`, resample the active
   interval, assemble features (+ statistics for RF) and run the model (50 samples).

MediaPipe extraction (Step 2: ≈ 15.5 frames/s per CPU worker on the M2) and buffering are
not included; no claim about real-time captioning is made from these numbers.

## 9. Reproducibility

Seeds for Python, NumPy and PyTorch (42), `torch.use_deterministic_algorithms(True,
warn_only=True)`, CPU by default on the M2 (4 threads), seeded DataLoader shuffling.
Recorded per experiment: seed, Python/PyTorch/NumPy versions, platform, device (GPU name
if any), all hyperparameters, batch size used, epochs, best epoch, best validation
macro-F1, timings. Experiments are never overwritten; results are appended to
`reports/step4/results/experiment_results.csv`. Verified: repeating an experiment gives
identical history and metrics; a reloaded checkpoint reproduces validation macro-F1.

## 10. Benchmark limitations

- The only real Step 3 data is the benchmark store: 374 accepted videos, **all in TRAIN**
  (the Step 2 benchmark was drawn from the train split), 357 of 2,207 classes, ~1 video
  per class. It has **no validation or test videos**, so no real validation/test metric
  exists yet. Re-splitting it would break the frozen split, so it is not done.
- The real run is a pipeline check (data loading, training, checkpointing, latency), not
  a result. Models always have 2,207 outputs; benchmark numbers say nothing about
  2,207-class performance.
- Meaningful experiments need landmark extraction of validation and test videos (and,
  for the full matrix, the full dataset) — a Step 2 run requiring approval.
