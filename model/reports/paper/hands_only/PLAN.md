# Hands-only representation (paper Objective 1) — plan, written before the runs (2026-10-27)

- Runs: BiLSTM, feature_config `hands` (left + right hand: 2 x (21 x 3 + 1) = 128 features), window 30,
  100-epoch budget, seeds 42, 1, 2, 3, 4, no augmentation, `--no-test-eval`.
  Experiment ids `hands_w30_bilstm_e100[_s1.._s4]` in `experiments/dev100_hp/`.
- Comparison: paired by seed with the existing hands + pose BiLSTM runs `hands_pose_w30_bilstm_e100[_s1.._s4]`
  (identical configuration apart from the feature configuration).
- Reported (descriptive, validation only): mean validation macro-F1 and accuracy per representation, the
  paired mean difference (hands - hands_pose) with a class-stratified 95 % bootstrap interval, number of seeds
  in each direction, train - validation gap. No adoption decision depends on it; test is not evaluated.
