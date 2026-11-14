# Paper tables (generated from saved reports by scripts/paper_01_tables.py)

## T1. Dataset (dev100 store)

| classes | accepted clips | train / val / test | train clips per class (median, min–max) | signer-independent split |
|---|---|---|---|---|
| 100 | 4838 | 3407 / 725 / 706 | 33 (26–45) | no (no signer IDs) |

Source: `data/sequences/dev100/index.parquet`

## T2. LSTM vs BiLSTM, hands + pose, 30 frames, 5 seeds (test values were reported during development)

| split | model | accuracy % (mean ± SD) | 95 % CI | macro-F1 (mean ± SD) | 95 % CI  |
|---|---|---|---|---|---|
| val | LSTM | 60.0 ± 3.2 | 57.4–62.6 | 0.594 ± 0.033 | 0.561–0.615 |
| val | BiLSTM | 66.6 ± 0.9 | 63.9–69.4 | 0.662 ± 0.010 | 0.629–0.685 |
| test | LSTM | 57.5 ± 3.8 | 54.8–60.3 | 0.570 ± 0.043 | 0.537–0.592 |
| test | BiLSTM | 63.9 ± 1.0 | 61.1–66.6 | 0.634 ± 0.010 | 0.600–0.656 |
| val | BiLSTM − LSTM (paired) | +6.6 pts | +5.3 to +7.9 | +0.068 | +0.054 to +0.083 |
| test | BiLSTM − LSTM (paired) | +6.3 pts | +4.9 to +7.7 | +0.064 | +0.049 to +0.078 |

Source: `reports/e0/bootstrap/seed_mean.csv`; `reports/e0/bootstrap/bilstm_minus_lstm.csv`

## T3. Landmark representation, VALIDATION, 5 paired seeds

| setting | representation | features | val macro-F1 (mean ± SD) | paired change (95 % CI) | train − val acc (pts) |
|---|---|---|---|---|---|
| BiLSTM, no augmentation | hands + full pose | 261 | 0.662 ± 0.010 | reference | 32.5 |
| BiLSTM, no augmentation | hands only | 128 | 0.685 ± 0.015 | +0.023 (+0.007 to +0.040) | 30.4 |
| BiLSTM + temporal augmentation | hands + full pose | 261 | 0.691 ± 0.011 | reference | 30.1 |
| BiLSTM + temporal augmentation | hands + upper-body pose (0–22) | 221 | 0.701 ± 0.017 | +0.010 (-0.002 to +0.023) | 28.6 |
| — | hands + pose + face | – | not evaluated | face landmarks not extracted for dev100 | – |

Source: `reports/paper/hands_only/comparison/summary.json`; `reports/paper/hands_only/comparison/per_seed.csv`; `reports/e2/comparison/summary.json`; `reports/e2/comparison/per_seed.csv`; `configs/step2_landmarks.yaml`

## T4. Temporal resampling length (EXPLORATORY: single seed, LSTM; not a sliding-window study)

| frames per sign | model | seed | val macro-F1 | test macro-F1 (reported in development) | model latency b=1 (ms, training run) |
|---|---|---|---|---|---|
| 30 | LSTM | 42 | 0.600 | 0.612 | 0.93 |
| 45 | LSTM | 42 | 0.518 | 0.500 | 1.29 |

Source: `reports/step4/results/experiment_results.csv`

## T5. Development ladder, VALIDATION (paired changes) with the test column from the one-time final evaluation

| model | val macro-F1 (mean ± SD) | val accuracy % | train − val (pts) | paired change (95 % CI) | test acc / macro-F1 |
|---|---|---|---|---|---|
| Baseline BiLSTM | 0.662 ± 0.010 | 66.6 | 32.5 | reference | 63.9 % / 0.634 (reported in development) |
| E1: + temporal augmentation | 0.691 ± 0.011 | 69.5 | 30.1 | +0.028 (+0.017 to +0.041); 5/5 seeds; §3 met | 66.1 % / 0.658 (one-time final evaluation) |
| E2: E1 + upper-body pose | 0.701 ± 0.017 | 70.5 | 28.6 | +0.010 vs E1 (-0.002 to +0.023); 4/5 seeds; §3a non-inferiority met, §3 not met | 67.8 % / 0.675 (one-time final evaluation) |

Source: `reports/e1/comparison/per_seed.csv`; `reports/e1/comparison/summary.json`; `reports/e2/comparison/per_seed.csv`; `reports/e2/comparison/summary.json`; `reports/e0/bootstrap/seed_mean.csv`

## T5b. TEST, one-time final evaluation after development closed (706 clips, 5 seeds; class-stratified bootstrap)

| model | test accuracy % (mean ± SD) | 95 % CI | test macro-F1 (mean ± SD) | 95 % CI  |
|---|---|---|---|---|
| Baseline BiLSTM (saved predictions, reported in development) | 63.9 ± 1.0 | 61.1–66.6 | 0.634 ± 0.010 | 0.600–0.656 |
| E1: + temporal augmentation | 66.1 ± 1.8 | 63.5–69.0 | 0.658 ± 0.019 | 0.625–0.681 |
| E2: E1 + upper-body pose (221 features) | 67.8 ± 1.6 | 65.3–70.6 | 0.675 ± 0.017 | 0.643–0.698 |
| paired e1 - baseline | +2.3 pts | +1.1 to +3.6 | +0.024 (5/5 seeds better) | +0.012 to +0.038 |
| paired e2 - e1 | +1.6 pts | +0.4 to +2.8 | +0.016 (5/5 seeds better) | +0.004 to +0.030 |
| paired e2 - baseline | +3.9 pts | +2.5 to +5.3 | +0.041 (5/5 seeds better) | +0.026 to +0.057 |

Source: `reports/paper/final_test/seed_mean.csv`; `reports/paper/final_test/paired.csv`; `reports/paper/final_test/summary.json`

## T6. Latency and throughput (CPU, macOS-27.0.1-arm64-arm-64bit, 8 GB, torch threads 4)

| component | median ms | p95 ms | note |
|---|---|---|---|
| MediaPipe Holistic per frame | 28.5 | 29.7 | max 35.1 frames/s single process |
| BiLSTM forward, batch 1 | 1.68 | 1.80 | dedicated benchmark, 500 runs |
| LSTM forward, batch 1 (5 training runs) | 0.89 | – | measured inside training runs (median of runs) |
| BiLSTM forward, batch 1 (5 training runs) | 1.72 | – | measured inside training runs (median of runs) |
| streaming push per frame | 0.007 | 0.009 |  |
| segment close: preprocessing + model | 4.1 | 4.9 | 20 dataset clips |
| live capture S2 | 66.6 (frame interval) | 70.2 | 15.0 processed fps; threaded=False |
| live capture S3 | 33.5 (frame interval) | 38.1 | 29.8 processed fps; threaded=True |
| live capture S4 | 33.4 (frame interval) | 34.6 | 29.9 processed fps; threaded=True |

Source: `reports/step6/latency_hands_pose_w30_bilstm_e100_s3.json`; `reports/step4/results/experiment_results.csv`; `reports/step6/webcam_eval/<session>/summary.json`

## T7. Webcam development replay set — ONE SIGNER, 92 core attempts, frozen segments (not generalization evidence)

| model (5 seeds) | top-1 % mean (min–max) | top-5 % mean |
|---|---|---|
| LSTM | 29.8 (23.9–39.1) | 57.2 |
| BiLSTM (baseline) | 42.4 (39.1–46.7) | 63.0 |
| E1 | 43.9 (37.0–47.8) | 66.5 |
| E2 | 43.7 (35.9–51.1) | 69.6 |

Source: `reports/e0/webcam_replay_set/summary_by_model.csv`; `reports/e1/comparison/webcam_replay_paired.csv`; `reports/e2/comparison/webcam_replay_paired.csv`

## T8. Live streaming sessions — ONE SIGNER, baseline BiLSTM candidate (S3 re-scored by replay with the final scoring rules)

| session | protocol | attempts | correct | missed | attempts with duplicates | answers forced-cut (live) | median delay rest-closed s (live) | processed fps | dropped frames |
|---|---|---|---|---|---|---|---|---|---|
| S3 | session 3: threaded capture, gated GO | 20 | 13 | 0 | 9 | 11 | 0.46 | 29.8 | 28/5300 |
| S4 | session 4: controlled protocol - hands o | 20 | 9 | 0 | 2 | 5 | 0.44 | 29.9 | 36/5052 |

Source: `reports/step6/webcam_eval/20261019T084115Z_camera_S01/replay_firsthand/summary.json`; `reports/step6/webcam_eval/20261020T045903Z_camera_S01/summary.json`; `…/attempts.csv`

## T9. Streaming post-processing replay — ONE SIGNER, 92 core attempts, baseline BiLSTM candidate

| cooldown s | confidence threshold | attempts | correct | top-1 % | coverage % | precision of answers % | duplicate emissions | missed (no segment) | spurious outside attempts |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 92 | 37 | 40.2 | 100.0 | 40.2 | 14 | 0 | 3 |
| 0.5 | 0.0 | 92 | 37 | 40.2 | 100.0 | 40.2 | 0 | 0 | 2 |
| 1.0 | 0.0 | 92 | 37 | 40.2 | 100.0 | 40.2 | 0 | 0 | 2 |
| 0.0 | 0.4 | 92 | 35 | 38.0 | 76.1 | 50.0 | 8 | 0 | 2 |
| 0.0 | 0.57 | 92 | 30 | 32.6 | 54.3 | 60.0 | 5 | 0 | 1 |
| 0.0 | 0.7 | 92 | 28 | 30.4 | 43.5 | 70.0 | 4 | 0 | 0 |

Source: `reports/paper/postprocessing/summary_by_setting.csv`; `reports/paper/postprocessing/summary.json`

## T10. Confidence calibration, baseline candidate (BiLSTM seed 3); test check was run during development

| split | accuracy % | ECE (15 bins) | mean confidence |
|---|---|---|---|
| val | 67.6 | 0.093 | 0.768 |
| test | 64.0 | 0.119 | 0.757 |
| threshold 0.57 (chosen on val, 80 % precision target) on test | 77.7 of accepted | – | coverage 74.9 % |

Source: `reports/step6/calibration_hands_pose_w30_bilstm_e100_s3.json`

## T11. Robustness diagnostics, VALIDATION, accuracy change vs clean (pts, mean of 5 BiLSTM seeds)

| perturbation | baseline | E1 | E2 |
|---|---|---|---|
| hips_hidden | -0.5 | -0.9 | +0.0 |
| hips_visible | -3.0 | -3.5 | +0.0 |
| hold_start | -17.1 | -6.8 | -7.3 |
| hold_end | -16.9 | -7.6 | -7.6 |
| repeat_x2 | -10.9 | -8.0 | -7.0 |
| drop_minor_hand | -29.2 | -30.8 | -31.1 |
| hand_dropout_20 | -8.9 | -7.8 | -6.5 |
| mirror | -17.4 | -18.2 | -18.2 |
| rotate_10 | -6.0 | -5.6 | -5.3 |
| noise_002 | +0.1 | -0.0 | -0.1 |

Source: `reports/e0/robustness/by_model.csv`; `reports/e1/comparison/robustness_e1_vs_ref.csv`; `reports/e2/comparison/robustness_e2.csv`

## T12. Test-set use disclosure (no test result was used to choose a model)

| use | count | source |
|---|---|---|
| test metrics computed for development runs | 13 | reports/step4/results/experiment_results.csv (rows with test values) |
| confidence-threshold check on test (candidate s3) | 1 | reports/step6/calibration_hands_pose_w30_bilstm_e100_s3.json |
| latency/parity script: predictions on 20 test clips (equality check only) | 1 | reports/step6/latency_hands_pose_w30_bilstm_e100_s3.json |
| simulated live replay of dataset test clips | 3 | reports/step6/webcam_eval/*_video_replay_dataset-test-clip |
| one-time final test evaluation of E1 and E2 (5 seeds each), after development closed | 1 | reports/paper/final_test/ |
| hands-only test inference | 0 | metadata.json test_evaluated = false |

Source: `as listed`

## Figures

- `F2_val_macro_f1_paired_seeds.png`
- `F3_latency_breakdown.png`
- `F4_threshold_coverage_precision.png`
- `F5_coverage_accuracy_dataset_vs_webcam.png`
