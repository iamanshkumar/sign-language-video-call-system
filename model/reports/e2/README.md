# E2 — upper-body pose ablation on top of E1 (2026-10-27)

One change versus the reference E1 (BiLSTM, hands_pose, w30, temporal augmentation, seeds 42, 1, 2, 3, 4):
the 40 feature columns of pose landmarks 23–32 (hips, knees, ankles, heels, feet; x, y, z and visibility;
columns 197–226 and 251–260) are removed for every split, giving 221 features instead of 261. Judged by the
non-inferiority rule (docs/evaluation_protocol.md §3a), added and approved before the runs. Runs:
`experiments/dev100_hp/hands_pose_w30_bilstm_e100_augtemp_upperpose[_s1.._s4]`; logs `logs/`; comparison
`comparison/` (`scripts/e2_01_compare.py`). **No test inference.** Not exportable (refused by design: the
inference pipeline does not support the reduced feature set).

## Motivation (measured before E2, read-only)
- Shuffling landmarks 23–32 across validation clips cost E1 5.4 pts accuracy (range 3.9–8.1): the model
  used them, although knees/ankles are almost never visible (visibility 0.02 / 0.003) and ASL signing
  happens above the waist. These 40 features alone predicted the class at 7.2 % (chance 1 %).
- Hip visibility varies on the webcam (0.015–0.94) versus 0.26 in training.

## Validity
- Each E2 `config.json` equals its E1 config apart from `feature_ablation`; feature_dim 221; dropped columns
  as specified; no test files; `test_evaluated = false`; results rows have empty test columns.
- Clean validation inputs (columns removed) reproduce every E2 run's saved predictions.
- All 654 pre-existing files unchanged except the append-only results table (original bytes are an exact
  prefix; 5 rows added). 193 tests pass.

## Rule §3a (validation macro-F1, paired with E1)
| seed | E1 | E2 | change (paired 95 % CI) |
|---|---|---|---|
| 42 | 0.678 | 0.703 | +0.025 (−0.002, +0.052) |
| 1 | 0.705 | 0.722 | +0.017 (−0.008, +0.043) |
| 2 | 0.694 | 0.694 | +0.000 (−0.028, +0.027) |
| 3 | 0.683 | 0.710 | +0.027 (+0.002, +0.052) |
| 4 | 0.694 | 0.676 | −0.018 (−0.045, +0.009) |
| **mean** | **0.691** | **0.701** | **+0.010 (−0.002, +0.023)** |

- Mean change ≥ −0.01: met (+0.010). CI lower bound ≥ −0.02: met (−0.002). No class at F1 = 0: met.
  **Non-inferiority met.**
- Improvement rule §3 (≥ +0.02, ≥ 4/5 seeds): **not met** (+0.010; 4/5 seeds better, sign test p = 0.375).
  E2 is not shown to be better than E1; it is shown not to be worse by more than the margin.

## Secondary (descriptive)
- Overfitting (mean of 5 seeds): train 99.6 → 99.0 %, val 69.5 → 70.5 %, gap 30.1 → 28.6 pts.
- Per class: 58 better, 42 worse (median +0.012, IQR −0.028 to +0.043). Gains: BLIND 0.64→0.79, COUSIN,
  GRANDFATHER, BUY, RED; losses: MINUTE 0.69→0.56, BALD, GOVERNMENT, GRADUATE, STUDENT.
- Robustness (accuracy drop vs clean): hips_hidden / hips_visible 0.0 (by construction; E1 −0.9 / −3.5);
  hold_start −7.3, hold_end −7.6, repeat_x2 −7.0, drop_minor_hand −31.1, mirror −18.2, rotate_10 −5.3 —
  otherwise similar to E1.
- Webcam development replay set (92 core attempts, one signer): top-1 43.9 → 43.7 % (per seed −6.5 to +7.6
  pts; pooled 29 attempts gained vs 30 lost; no seed significant); top-5 66.5 → 69.6 %. By sign: TUESDAY
  44→56 %, BLIND 100→70 %; AFRICA, COMPUTER, FORBID stay at ~0 %.

## Conclusion
Removing the lower-body landmarks is non-inferior on validation (point estimate +0.010) and removes a
non-linguistic, mostly invisible input channel the model had been using, with no measurable cost on the
webcam development set. It does not meet the improvement rule. By the decision recorded before the run,
upper-body pose (221 features) becomes the representation for subsequent experiments. Offline
validation results do not establish real-world improvement; the persistent webcam failures (AFRICA,
COMPUTER, FORBID) are unchanged and remain a signer/variant limitation.
