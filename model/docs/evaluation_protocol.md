# Evaluation protocol for the paper (E0, frozen 2026-10-23)

Applies to every model result reported from now on. Changes to this document need explicit
approval and are recorded in the change log at the end.

## 1. Data splits and their roles
- **Store:** `dev100_hp` (4,838 accepted videos, 100 classes, frozen split v1: train 3,407 / val 725 /
  test 706), `hands_pose` features (F = 261), window 30.
- **Train:** fitting only.
- **Validation:** every development decision (architecture, features, augmentation, regularization,
  epochs, thresholds). Checkpoints are selected on validation macro-F1.
- **Test:** final reporting only. **Test-use log:** before E0, test metrics were computed and reported
  for 13 dev100 runs (`reports/step4/results/experiment_results.csv`); none was used for a choice. From
  E0 on, test inference is run only for the final candidates named in §5, once, after all choices are
  fixed. E0 computes confidence intervals from the predictions saved by those 13 runs (no new test
  inference).
- **Limitation (must be stated in the paper):** the split is not signer-independent. There are no
  signer IDs; the only grouping is 4 filename collections (2,917 / 1,146 / 708 / 67 clips), all present
  in every split, which is too coarse to act as a signer proxy. Test accuracy is therefore an estimate
  for signers/recordings like those in the dataset, not for unseen signers.

## 2. Metrics and uncertainty
- Primary offline metric: macro-F1 (`asl.models.metrics.classification_metrics`, macro over the union
  of true and predicted labels). Also reported: accuracy, weighted-F1, per-class F1, top-5 where
  available, expected calibration error.
- Uncertainty: (a) seed variation — 5 seeds (42, 1, 2, 3, 4), mean ± sample SD and range; (b) clip
  variation — 95 % percentile bootstrap over clips, resampled within each true class (2,000 resamples,
  seed 0); for a seed-mean the same
  clip resample is applied to all seeds. Paired differences between two models use the same resamples
  (paired bootstrap) and, per seed, exact McNemar tests.
- No p-value is reported without the number of units it is based on (seeds, clips, attempts, signers).

## 3. Decision rule for "the model improved" (offline)
A change is accepted only if, against the current reference (BiLSTM, hands_pose, w30, 5 seeds):
1. mean validation macro-F1 over 5 seeds improves by **≥ 0.02** (≈ 2 × the reference seed SD 0.010);
2. it is better in **≥ 4 of 5** paired seeds;
3. no class has validation F1 = 0 in the seed-mean model set.
Train–validation gap and calibration are secondary and cannot justify a change on their own.
One change per experiment; the experiment and its comparison are written down before training.

**3a. Non-inferiority rule (simplifying changes only).** A change that removes inputs or complexity
(e.g. a feature ablation), decided before training to be judged for non-inferiority, is accepted against
its paired reference if: (1) the mean validation macro-F1 change over the 5 paired seeds is **≥ −0.01**;
(2) the lower bound of the paired, class-stratified 95 % bootstrap interval of that mean change is
**≥ −0.02**; (3) no class has validation F1 = 0. Meeting §3 as well counts as an improvement. Robustness
and webcam replay results are secondary and descriptive and cannot rescue a change that fails (1)–(3).
Current reference for E2: E1 (temporal augmentation, seeds 42, 1, 2, 3, 4).

## 4. Webcam development replay set
- Built by `scripts/e0_02_webcam_replay_set.py` from saved session landmarks; one segment per attempt,
  frozen as frame ranges (the segmenter does not depend on the classifier), so every model is evaluated
  on identical inputs and comparisons are paired (exact McNemar on attempts).
- Status: **development data** — one signer (S01), already inspected, protocol varied between sessions.
  It may support a decision as secondary evidence only and never yields a reported real-world accuracy.
- Core subset: sessions recorded with threaded capture at ≥ 25 fps (S3, S4, 5a, 5b, 5c-1, 5c-2);
  S2 (15 fps) and 5d (24 fps) are listed but excluded from the core subset; attempts excluded by the
  objective session rules (signing during WAIT, pre-GO-only) are flagged.

## 5. Final evaluation (after development is closed)
- Offline: final candidate(s) and the reference, 5 seeds each, test evaluated once, with §2 intervals.
- Real world: a new recording set of ≥ 5 signers who were not involved in development (consent
  required), fixed setup (camera device and format logged, ≥ 25 fps, framing "OK"), protocol fixed before
  recording; intervals by signer-level bootstrap; per-signer results reported.
- Optional cross-dataset test on a signer-labelled public dataset for shared glosses, subject to its
  licence.

## 6. Robustness diagnostics (E0, validation only, predeclared)
Purpose: decide which training intervention to test first; not a reported result. Each perturbation is
applied to the stored arrays inside the active interval before the unchanged resampling and feature
assembly; models: the 5 BiLSTM and 5 LSTM seed runs. Fixed parameters:

| id | perturbation | webcam motivation |
|---|---|---|
| hips_hidden | hip (23, 24) visibility → 0 | S3 hips 0.015 vs training 0.26 |
| hips_visible | hip visibility → 0.95 | 5c-2 hips 0.935 |
| hold_start | first frame repeated for +50 % of the interval | raise / pause before signing |
| hold_end | last frame repeated for +50 % | holding the final position |
| repeat_x2 | interval played twice | repeated movement |
| drop_minor_hand | hand with fewer observed frames removed | one hand not detected |
| hand_dropout_20 | each hand removed in 20 % of frames (seed 0) | intermittent detection |
| mirror | x → −x, left/right hands and pose pairs swapped | left-hand-dominant signing |
| rotate_10 | x, y rotated by +10° about the shoulder centre | camera/body roll |
| noise_002 | Gaussian noise σ = 0.02 shoulder widths on present coordinates (seed 0) | landmark jitter |

Report per perturbation: accuracy and macro-F1 drop versus clean (paired bootstrap 95 % CI), mean over
the 5 BiLSTM seeds and per seed. Clean results must reproduce the saved validation predictions exactly.
Interpretation rule: the first training experiment targets the perturbation with the largest
consistent drop (CI excluding 0 in ≥ 4/5 BiLSTM seeds) **that corresponds to a measured webcam
mismatch**; ties are broken by implementation risk. Mirror is reported but is not a mismatch measured in
the webcam data.

## Change log
- 2026-10-23: created (E0).
- 2026-10-23: §2 bootstrap made class-stratified before any result was interpreted: the unstratified
  bootstrap put macro-F1 point estimates near the top of their own intervals (e.g. 0.634 in [0.586, 0.647])
  because resampling ~7 clips per class distorts class supports. Applies to all models alike.
- 2026-10-27: §3a non-inferiority rule added (approved) before E2 (upper-body pose ablation) was run; E1
  became the reference for E2. No E2 result existed when this was written.
