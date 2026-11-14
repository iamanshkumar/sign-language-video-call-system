# E1 — temporal augmentation (2026-10-23)

One change versus the BiLSTM reference (hands_pose, w30, 100-epoch budget, seeds 42, 1, 2, 3, 4): during
training only, with probability 0.5 per clip, one temporal operation is applied to the active interval
before the unchanged resampling (`src/asl/sequences/augment.py`): hold at start or end (10–50 % of the
interval), tempo warp (local speed 2/3–3/2, interpolated like the training resampling), or one extra
cycle of a sub-span whose end returns to its start. Labels are never changed; validation, test and
webcam inputs are untouched. Runs: `experiments/dev100_hp/hands_pose_w30_bilstm_e100_augtemp[_s1.._s4]`,
logs `logs/`, comparison `comparison/` (`scripts/e1_01_compare.py`). **No test inference** (`--no-test-eval`).

## Validity
- Each E1 `config.json` equals its reference apart from `train_augmentation` and `evaluate_test`; no test
  files; `test_evaluated = false`; results rows have empty test columns.
- Augmentation realism on all 3,407 training clips (`augmentation_realism_check.csv`): holds and
  repetition never create a frame-to-frame hand movement larger than the largest one already in the clip
  (gap-aware); tempo exceeds 1.5× the clip's own maximum in 0.3 % of clips (interpolation across existing
  detection gaps). Repetition finds a valid loop in 92 % of clips (lowest class: STUDENT 36 %).
- No class pair in the vocabulary differs only by repetition or tempo, so the operations cannot turn one
  class into another. Operation counts are recorded per run (about 50 % unchanged, ~12 % each operation).
- All 522 pre-existing experiment/model/report/config/index files unchanged except the append-only
  `reports/step4/results/experiment_results.csv`, whose original bytes are an exact prefix (5 rows added).
- Clean validation inputs reproduce every E1 run's saved predictions. 187 tests pass.

## Decision rule (docs/evaluation_protocol.md §3) — validation macro-F1
| seed | reference | E1 | gain (paired 95 % CI) |
|---|---|---|---|
| 42 | 0.666 | 0.678 | +0.012 (−0.012, +0.036) |
| 1 | 0.651 | 0.705 | +0.055 (+0.029, +0.083) |
| 2 | 0.670 | 0.694 | +0.024 (−0.003, +0.050) |
| 3 | 0.673 | 0.683 | +0.010 (−0.014, +0.034) |
| 4 | 0.652 | 0.694 | +0.042 (+0.017, +0.068) |
| **mean** | **0.662** | **0.691** | **+0.029 (+0.017, +0.041)** |

- Gain ≥ 0.02: **met** (+0.029). Better in ≥ 4/5 seeds: **met** (5/5; exact sign test p = 0.0625, the minimum
  possible with 5 pairs). No class with validation F1 = 0: **met**. **All predefined criteria met.**
- Per seed, the clip-level interval excludes 0 for seeds 1 and 4 only; the gain is mostly a shift of the
  weaker seeds (E1 seed SD 0.011, reference 0.010; range 0.678–0.705 vs 0.651–0.673).
- The validation set also selected each checkpoint (both arms identically), so absolute values are
  slightly optimistic; the paired comparison is not affected by that in one direction.

## Overfitting (mean of 5 seeds)
| | train acc | val acc | train − val |
|---|---|---|---|
| reference | 99.1 % | 66.6 % | 32.5 pts |
| E1 | 99.6 % | 69.5 % | 30.1 pts |
The gap shrinks by 2.4 pts because validation rises; train accuracy stays ~100 %. Overfitting is not solved.

## Per class (validation F1, mean over seeds)
72 classes improve, 26 get worse, 2 unchanged; median change +0.026 (IQR −0.001 to +0.066). Largest gains:
BAD 0.59→0.84, DOUBLE 0.36→0.53, PERSON 0.56→0.72, NURSE 0.25→0.41. Largest losses: WET 0.70→0.55,
SUN 0.70→0.57, VOTE 0.81→0.72, CHURCH 0.47→0.38. Weakest classes remain NO (0.27), CHURCH, PRINT.

## Robustness (validation; accuracy drop vs clean, mean of 5 seeds)
| perturbation | reference | E1 |
|---|---|---|
| hold_start | −17.1 | −6.8 |
| hold_end | −16.9 | −7.6 |
| repeat_x2 | −10.9 | −8.0 |
| tempo-unrelated: drop_minor_hand / mirror / rotate_10 | −29.2 / −17.4 / −6.0 | −30.8 / −18.2 / −5.6 |
| hips_visible | −3.0 | −3.5 |
The targeted sensitivities fall sharply; the untargeted ones are unchanged (expected — and partly
by construction, since the hold/repeat perturbations resemble the training operations).

## Webcam development replay set (one signer, 92 core attempts; secondary evidence)
| seed | reference top-1 | E1 top-1 | E1 top-5 (ref) | E1-only / ref-only correct | exact McNemar p |
|---|---|---|---|---|---|
| 42 | 39.1 % | 37.0 % | 62.0 % (63.0) | 6 / 8 | 0.79 |
| 1 | 40.2 % | 43.5 % | 68.5 % (64.1) | 9 / 6 | 0.61 |
| 2 | 46.7 % | 47.8 % | 65.2 % (64.1) | 7 / 6 | 1.00 |
| 3 | 40.2 % | 47.8 % | 70.7 % (58.7) | 9 / 2 | 0.065 |
| 4 | 45.7 % | 43.5 % | 66.3 % (65.2) | 3 / 5 | 0.73 |
Mean top-1 42.4 % → 43.9 % (+1.5 pts); no seed shows a significant paired difference. By sign (pooled
seeds): BLIND 60→100 %, DIRECTOR 0→20 %, TUESDAY 36→44 %; AFRICA, COMPUTER, FORBID stay at 0 %.
The persistent webcam failures are not fixed by temporal augmentation.

## Conclusion
E1 meets every predefined offline criterion on validation (+0.029 mean macro-F1, 5/5 seeds, no zero
class) and removes most of the model's sensitivity to holds; webcam evidence is neutral to slightly
positive and not significant. Not done (by design): test evaluation, model selection on test, promotion.
