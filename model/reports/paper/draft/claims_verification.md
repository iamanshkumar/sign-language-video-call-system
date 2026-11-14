# Claims verification

Draft: `reports/paper/draft/sections.md`. 61 claims, 0 failed.

| result | claim | text expected in the draft (recomputed from source) | source (under reports/) |
|---|---|---|---|
| PASS | classes / clips | 100 ASL word classes and 4,838 accepted video clips | index.parquet |
| PASS | split | 3,407 training, 725 validation and 706 test | index.parquet |
| PASS | clips per class | median of 33 training clips per class (range 26–45) | index.parquet |
| PASS | BiLSTM test acc | 63.9 ± 1.0 % (95 % CI 61.1–66.6) | e0/bootstrap/seed_mean.csv |
| PASS | LSTM test acc | 57.5 ± 3.8 % for the LSTM | e0/bootstrap/seed_mean.csv |
| PASS | test macro-F1 both | 0.634 ± 0.010 and 0.570 ± 0.043 | e0/bootstrap/seed_mean.csv |
| PASS | paired test acc | +6.3 percentage points in accuracy (95 % CI +4.9 to +7.7) | e0/bootstrap/bilstm_minus_lstm.csv |
| PASS | paired test F1 | +0.064 in macro-F1 (+0.049 to +0.078) | e0/bootstrap/bilstm_minus_lstm.csv |
| PASS | hands-only means | from 0.662 to 0.685 | paper/hands_only/comparison/summary.json |
| PASS | hands-only change | +0.023 (95 % CI +0.007 to +0.040) | paper/hands_only/comparison/summary.json |
| PASS | hands-only per class | 59 classes improved and 41 declined | paper/hands_only/comparison/summary.json |
| PASS | hands-only seeds | all five seeds higher | paper/hands_only/comparison/summary.json |
| PASS | hands-only largest declines | VOTE, REGISTER and PLEASE | paper/hands_only/comparison/per_class_val_f1.csv |
| PASS | E2 val means | from 0.691 to 0.701, a paired change of +0.010 (95 % CI -0.002 to +0.023) | e2/comparison/summary.json |
| PASS | E2 non-inferiority met / improvement not met | met the predefined non-inferiority rule | e2/comparison/summary.json |
| PASS | E2 per-seed range | from -0.018 to +0.027, with four of five seeds higher | e2/comparison/val_change_paired.csv |
| PASS | w45 vs w30 | 0.518 against 0.600 | step4/results/experiment_results.csv |
| PASS | E1 val | from 0.662 to 0.691, a paired gain of +0.028 (95 % CI +0.017 to +0.041) | e1/comparison/summary.json |
| PASS | E1 all seeds / rule met | all five seeds improved | e1/comparison/summary.json |
| PASS | E1 test acc | 66.1 ± 1.8 % for E1 (95 % CI 63.5–69.0) | paper/final_test/seed_mean.csv |
| PASS | E2 test acc | 67.8 ± 1.6 % for E2 (65.3–70.6) | paper/final_test/seed_mean.csv |
| PASS | baseline test acc (reused) | against 63.9 ± 1.0 % for the baseline | paper/final_test/seed_mean.csv |
| PASS | test macro-F1 | 0.658 ± 0.019 for E1 and 0.675 ± 0.017 for E2, against 0.634 ± 0.010 | paper/final_test/seed_mean.csv |
| PASS | E2-baseline | +3.9 percentage points in accuracy (95 % CI +2.5 to +5.3) and +0.041 in macro-F1 (+0.026 to +0.057) | paper/final_test/paired.csv |
| PASS | E1-baseline | +2.3 points (+1.1 to +3.6) | paper/final_test/paired.csv |
| PASS | E2-E1 test | +1.6 points (+0.4 to +2.8) | paper/final_test/paired.csv |
| PASS | E2-E1 discussion | +1.6 points of accuracy and +0.016 macro-F1 | paper/final_test/paired.csv |
| PASS | all seeds each step | Each step was better in all five seeds | paper/final_test/paired.csv |
| PASS | E2 per class test | for 66 classes and reduced it for 34 | paper/final_test/per_class.csv |
| PASS | gaps | from 32.5 points (baseline) to 30.1 (E1) and 28.6 (E2) | e1/e2 per_seed.csv |
| PASS | hold_start | from -17.1 to -6.8 points | robustness |
| PASS | repeat | from -10.9 to -8.0 points | robustness |
| PASS | hand loss / mirror | (-29.2 points for the baseline) and to mirroring (-17.4) | robustness |
| PASS | hips | (-3.0 to 0.0 points) | robustness |
| PASS | mediapipe | median of 28.5 ms per frame (p95 29.7 ms) | latency json |
| PASS | max fps | about 35 frames per second | latency json |
| PASS | bilstm ms | median of 1.68 ms per sign | latency json |
| PASS | training-run latencies | LSTM took 0.89 ms and the BiLSTM 1.72 ms | experiment_results.csv |
| PASS | segment close | median of 4.1 ms | latency json |
| PASS | push | bookkeeping 0.007 ms | latency json |
| PASS | fps | processing to 15.0 frames per second | session summaries |
| PASS | fps threaded | raised this to 29.8–29.9 | session summaries |
| PASS | delay | median of 0.44–0.46 s | attempts.csv |
| PASS | webcam top1 | 29.8 % for the LSTM, 42.4 % for the BiLSTM baseline, 43.9 % for E1 and 43.7 % for E2 | replay set / e1 / e2 |
| PASS | webcam top5 | 57.2 %, 63.0 %, 66.5 % and 69.6 % | replay set / e1 / e2 |
| PASS | no significant seed | no statistically significant difference | webcam_replay_paired.csv |
| PASS | webcam attempts | 92 webcam attempts | replay set |
| PASS | sessions | 13 and 9 of 20 attempts were correct | session summaries |
| PASS | session dups | 9 and 2 attempts produced a duplicate | session summaries |
| PASS | session missed | no sign was missed | session summaries |
| PASS | pp default | 37 attempts were correct and 14 attempts produced one duplicate emission each | postprocessing |
| PASS | pp cooldown | removed all 14 duplicate emissions | postprocessing |
| PASS | threshold 0.4 | 76.1 % of attempts were answered with 50.0 % precision | postprocessing |
| PASS | threshold 0.57 | 54.3 % with 60.0 % | postprocessing |
| PASS | threshold 0.7 | 43.5 % with 70.0 % | postprocessing |
| PASS | pp no missed | No sign was missed under any setting | postprocessing |
| PASS | calibration test | 77.7 % precision at 74.9 % coverage | calibration json |
| PASS | duplicate causes | of the 14 duplicate emissions in the replay below, 13 were such continuations and one was a second burst of movement starting 0.57 s | paper/draft/webcam_details/summary.json |
| PASS | never recognised | AFRICA and COMPUTER were never recognised by any of the 20 models | paper/draft/webcam_details/summary.json |
| PASS | forbid once | FORBID was recognised once (by one E2 seed) in 240 model-attempts | paper/draft/webcam_details/summary.json |
| PASS | test uses | 13 training runs and three auxiliary checks | experiment_results.csv; T12 |
