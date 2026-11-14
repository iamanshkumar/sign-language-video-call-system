# E0 — evaluation baseline (2026-10-23)

Protocol: `docs/evaluation_protocol.md`. No training, no checkpoint or pipeline change, no new test
inference. All 432 pre-existing experiment, model, report, config and index files re-verified by SHA-256.

| part | script | output |
|---|---|---|
| confidence intervals (saved predictions) | `scripts/e0_01_bootstrap_ci.py` | `bootstrap/` |
| frozen webcam development replay set | `scripts/e0_02_webcam_replay_set.py` | `webcam_replay_set/` |
| robustness diagnostics (validation only) | `scripts/e0_03_robustness.py` | `robustness/` |
Shared helpers `scripts/e0_common.py`; tests `tests/test_e0.py`.

## 1. Offline results with intervals (5 seeds; 95 % class-stratified clip bootstrap)
Point estimates reproduce every saved metric exactly.

| split | model | accuracy mean ± SD | clip CI of seed mean | macro-F1 mean ± SD | clip CI |
|---|---|---|---|---|---|
| val | LSTM | 60.0 ± 3.2 % | 57.4–62.7 % | 0.594 ± 0.033 | 0.561–0.615 |
| val | BiLSTM | 66.6 ± 0.9 % | 63.9–69.4 % | 0.662 ± 0.010 | 0.629–0.685 |
| test | LSTM | 57.5 ± 3.8 % | 54.8–60.3 % | 0.570 ± 0.043 | 0.537–0.592 |
| test | BiLSTM | 63.9 ± 1.0 % | 61.1–66.6 % | 0.634 ± 0.010 | 0.600–0.656 |

BiLSTM − LSTM on test: +6.4 pts accuracy (CI 4.9–7.7), +0.064 macro-F1 (0.049–0.078); positive in every
seed (per-seed CIs exclude 0 on test; on validation seed 4's does not). Caveats: not signer-independent;
test was reported for 13 runs before E0.

## 2. Webcam development replay set (one signer; secondary evidence only)
140 attempts from 8 recordings, 134 with a segment; core subset (≥ 25 fps, objective exclusions applied)
92 attempts. Validity: the candidate reproduces the strict answer of every segment and every live answer
of the sessions scored live with the strict rules; its source run equals the candidate.

| model (5 seeds) | core top-1 mean (min–max) | core top-5 mean |
|---|---|---|
| LSTM | 29.8 % (23.9–39.1) | 57.2 % |
| BiLSTM | 42.4 % (39.1–46.7) | 63.0 % |

Paired on the same 92 attempts, BiLSTM is right where LSTM is wrong more often in every seed
(exact McNemar p from 0.40 to 8e-6). The candidate (BiLSTM seed 3) is a median seed here (40.2 %).
Webcam top-1 is ~20 pts below offline test accuracy.

## 3. Robustness (validation, BiLSTM mean of 5 seeds; drop vs clean 66.6 %)
| perturbation | Δ accuracy (CI) | all 5 seeds' CI < 0 | measured on the webcam? |
|---|---|---|---|
| drop_minor_hand | −29.2 (−31.4, −27.1) | yes | **no**: two-handed signs show both hands as often as training (ratio 0.92–1.37) |
| mirror | −17.4 (−20.0, −14.9) | yes | **no**: one-handed signs right-dominant in both (webcam 75–83 % right-only, training 57–86 %) |
| hold_start | −17.1 (−19.0, −15.2) | yes | **yes, in kind**: holds ≥ 0.3 s absent in training, median 9–11 % of webcam segments (up to 97 %) |
| hold_end | −16.9 (−18.9, −15.0) | yes | as above |
| repeat_x2 | −10.9 (−12.3, −9.5) | yes | **partly**: extra repetitions measured for HERE, COMPUTER |
| hand_dropout_20 | −8.9 (−10.5, −7.3) | yes | no (two-hand presence matches) |
| rotate_10 | −6.0 (−7.5, −4.5) | yes | no: shoulder-line angle matches (median 0.2° vs −0.5°) |
| hips_visible | −3.0 (−3.9, −2.0) | yes | yes, one session (5c-2 hip visibility 0.94 vs training 0.26) |
| hips_hidden | −0.5 (−1.0, 0.0) | no | yes (S3 0.015) — no effect |
| noise_002 | +0.1 | no | — |
LSTM shows the same ordering. Perturbation magnitudes are fixed stress levels, not webcam-calibrated.

## 4. Decision (protocol §6 rule)
The largest consistent drop that corresponds to a measured webcam mismatch is **temporal: holds
(−17 pts) and repetition (−11 pts)**. Hand loss and mirroring are larger sensitivities but are not measured
mismatches for this signer (they matter for the future multi-signer test). Hip visibility is measured but
small (−3 pts). Counter-evidence to weigh: on the webcam set, holds were not associated with errors within
signs, raise/lower cropping did not help, and the repetition protocol was inconclusive; the perturbations
are stronger than typical webcam holds. First training experiment proposed: temporal augmentation
(E1, needs approval).
