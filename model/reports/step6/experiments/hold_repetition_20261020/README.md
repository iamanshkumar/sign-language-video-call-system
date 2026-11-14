# Holds and repeated movement — exploratory offline measurements (2026-10-20)

Descriptive analysis of the 40 strictly scored answer segments of webcam sessions S3
(`20261019T084115Z_camera_S01`) and S4 (`20261020T045903Z_camera_S01`) against the training clips of the
same 10 signs. No crop rule, no model or pipeline change; thresholds come from the training data, not from
these 40 attempts. The method was fixed in the analysis script before it was run; it was not
written up separately beforehand. Analysis script: scratch-only (not in the repository).
Files: `per_attempt.csv` (both threshold settings), `training_reference.csv` (train-split clips of the 10
signs), `summary.json`.

## Method
- Same space for both: webcam segments rebuilt from `landmarks.npz` exactly as in
  `scripts/s6_06_analyze_active_motion_crop.py`, then the training preprocessing (`preprocess_raw`):
  30 fps grid, shoulder-normalized coordinates, active interval. Training clips: the stored dev100 arrays
  (train split; 334 clips of the 10 signs; thresholds from all 3,407 train clips).
- Per frame inside the active interval, per hand (observed in both consecutive frames): position speed =
  speed of the 21-landmark centroid; articulation speed = mean speed of landmarks relative to the
  centroid (hand-shape change). Max over the two hands; 5-frame centered smoothing.
- Low-motion frame: position speed < Tp and articulation speed < Ta, with (Tp, Ta) = the 10th or 25th
  percentile of all training frames (both reported): p10 = (0.267, 0.171), p25 = (0.677, 0.360) units/s.
- Hold: a run of low-motion frames of ≥ 0.3 s. hold_frac = hold time / active duration (also the share of
  the model's 30 resampled frames that fall in holds); move_s = duration − hold time.
- Bursts: peaks of position speed with prominence ≥ the median training frame speed (1.47 units/s),
  ≥ 0.2 s apart.
- Each webcam value is also expressed as a percentile among the training clips of the same sign.
- Correct vs wrong: Mann–Whitney U (unadjusted), and a within-sign permutation test (correctness shuffled
  within each sign, 10,000 permutations) because outcome is strongly tied to sign identity.

## Measured (p25 thresholds; p10 in summary.json, same direction)
| | training (train split) | S3 | S4 |
|---|---|---|---|
| active duration, median | 1.23 s | 4.03 s | 2.88 s |
| hold fraction, median | 0.00 | 0.11 | 0.09 |
| longest hold, median | 0.00 s | 0.43 s | 0.30 s |
| moving time, median | 1.20 s | 3.48 s | 2.73 s |
| bursts, median | 1 | 1 | 0 |

- Holds ≥ 0.3 s are essentially absent from training clips of these signs (median 0 at both thresholds)
  but occur in webcam segments, mostly BLIND (hold fraction 0.55–0.97), MORE (0–0.67), AFRICA (0.33–0.63).
- Moving time alone is 2–3× the training duration, so holds are not the main part of the gap: the
  movement itself takes longer (slower and/or repeated).
- More bursts than in training for the same sign: HERE (4–6 vs training median 1), COMPUTER (1–4 vs 0),
  FORBID S3 (2–3 vs 1).

## Association with correct vs wrong (22 correct, 18 wrong)
- Holds: correct median 0.14 vs wrong 0.10 (p = 0.30); within-sign p = 0.11, in the direction of *more*
  holds in correct answers. The longest holds (BLIND, MORE) were 8/8 correct.
- Duration and moving time: no clear difference (p = 0.58, 0.11).
- Bursts: wrong answers had more (median 1 vs 0, p < 0.001 unadjusted), but this disappears within signs
  (permutation p = 0.44). It comes from COMPUTER, HERE, FORBID (always or mostly wrong, many bursts) vs
  BLIND, MORE, PLEASE (always correct, no bursts). Within the 4 signs with mixed outcomes (AFRICA,
  DIRECTOR, HERE, TUESDAY) burst counts of correct and wrong attempts are the same or overlap.

## Limitations
- One signer, 40 attempts, 10 signs; outcome is confounded with sign identity; 24 unadjusted tests.
- The burst measure misses slow cyclic movement: PLEASE shows 3–4 height cycles in the crop-experiment
  plots but 0 bursts here. Repetition counts are therefore incomplete.
- Low-motion thresholds are generic percentiles of training speeds; signs with a static hand shape at a
  location (BLIND) count as holds by construction.
- Forced-cut segments are measured as classified (not merged with their continuation).

## Assessment
The data do not show that holds hurt recognition (if anything the reverse), and the apparent link
between repeated movement and errors is explained by which signs were attempted. The measurements do show
webcam signing is 2–3× longer than training even when moving, with extra repetitions for some signs
(HERE, COMPUTER). Whether repetition itself causes errors can only be separated from sign difficulty
by varying it within the same sign: a controlled comparison (same signs, single vs repeated movement,
randomized order) would be the follow-up, if pursued.
