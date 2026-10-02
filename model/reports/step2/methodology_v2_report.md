# Step 2 - methodology v2 re-evaluation of the Phase A benchmark

Re-evaluated from the stored Phase A extractions (no MediaPipe re-run): 400 attempted videos, 391 with stored landmarks.

## Decisions

| rule | outcome counts |
|---|---|
| v1 (whole clip) | {'both_hands_missing_fraction': 357, 'accepted': 26, 'too_few_decoded_frames': 9, 'both_hands_missing_run': 8} |
| v2 (active signing interval) | {'accepted': 374, 'active_both_hands_missing_fraction': 13, 'too_few_decoded_frames': 9, 'active_both_hands_missing_run': 3, 'no_active_signing_interval': 1} |

v2 acceptance: **93.5%**; accepted clips with a short active interval (< 30 frames; flagged `short_active_interval`, not rejected): 111.

Active frames (accepted): {'count': 374.0, 'mean': 40.3, 'std': 25.33, 'min': 3.0, '25%': 28.0, '50%': 36.0, '75%': 45.0, 'max': 252.0}

Both hands missing inside the interval, % (accepted): {'count': 374.0, 'mean': 1.9, 'std': 4.76, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 0.0, 'max': 28.38}

Rest frames before: {'count': 390.0, 'mean': 17.31, 'std': 15.25, 'min': 0.0, '25%': 8.0, '50%': 13.0, '75%': 24.0, 'max': 177.0}

Rest frames after: {'count': 390.0, 'mean': 26.91, 'std': 16.99, 'min': 0.0, '25%': 18.0, '50%': 24.0, '75%': 33.0, 'max': 159.0}

## Storage variants (measured on the stored benchmark extractions)

Full-dataset estimates use the v2 acceptance rate (93.5%).

| variant | MB/video | MB/video compressed | full GB | full GB compressed | hands GB | hands+pose GB | hands+pose+face GB | precision |
|---|---|---|---|---|---|---|---|---|
| float32 | 0.581 | 0.503 | 59.0 | 51.1 | 4.8 | 9.5 | 59.0 | reference |
| float16 | 0.293 | 0.239 | 29.7 | 24.3 | 2.6 | 5.0 | 29.7 | FAIL |
| float16_pose_float32 | 0.316 | 0.261 | 32.0 | 26.5 | 2.6 | 7.3 | 32.0 | FAIL |

## Precision validation (pre-registered criteria)

Criteria: {'max_abs_value_below': 60000.0, 'p999_error_px_below': 0.1, 'max_error_px_below': 0.5, 'max_zscore_feature_error_below': 0.01, 'max_stat_feature_rel_error_below': 0.01}

### float16

- values checked 52,023,933; error px {'mean': 0.021347249966849516, 'p99': 0.13066428570688005, 'p999': 0.24279928291978822, 'max': 0.7722632893564878}
- per group: {'pose': {'max_abs_value': 28.457393646240234, 'p999_px': 0.42694187302374087, 'max_px': 0.7722632893564878}, 'face': {'max_abs_value': 4.146161079406738, 'p999_px': 0.21711167028115597, 'max_px': 0.4905374561349163}, 'left_hand': {'max_abs_value': 3.83638858795166, 'p999_px': 0.13161323833975466, 'max_px': 0.2827667012461461}, 'right_hand': {'max_abs_value': 3.691145420074463, 'p999_px': 0.13288305900100084, 'max_px': 0.2932588000840042}}
- downstream: {'hands': {'max_zscore_feature_error': 0.004116023425012827, 'max_stat_feature_rel_error': 0.0039035738445818424}, 'hands_pose': {'max_zscore_feature_error': 0.013658168725669384, 'max_stat_feature_rel_error': 0.012828021310269833}, 'hands_pose_face': {'max_zscore_feature_error': 0.013658168725669384, 'max_stat_feature_rel_error': 0.012828021310269833}}
- checks: {'no_overflow': True, 'zero_fill_preserved': True, 'p999_error_px': False, 'max_error_px': False, 'zscore_features': False, 'rf_statistics': False} → **FAIL**

### float16_pose_float32

- values checked 52,023,933; error px {'mean': 0.018661595859056346, 'p99': 0.11347246118384646, 'p999': 0.21429136673244437, 'max': 0.4905374561349163}
- per group: {'pose': {'max_abs_value': 28.457393646240234, 'p999_px': 0.0, 'max_px': 0.0}, 'face': {'max_abs_value': 4.146161079406738, 'p999_px': 0.21711167028115597, 'max_px': 0.4905374561349163}, 'left_hand': {'max_abs_value': 3.83638858795166, 'p999_px': 0.13161323833975466, 'max_px': 0.2827667012461461}, 'right_hand': {'max_abs_value': 3.691145420074463, 'p999_px': 0.13288305900100084, 'max_px': 0.2932588000840042}}
- downstream: {'hands': {'max_zscore_feature_error': 0.004116023425012827, 'max_stat_feature_rel_error': 0.0039035738445818424}, 'hands_pose': {'max_zscore_feature_error': 0.004116023425012827, 'max_stat_feature_rel_error': 0.0039035738445818424}, 'hands_pose_face': {'max_zscore_feature_error': 0.009085999801754951, 'max_stat_feature_rel_error': 0.008258175104856491}}
- checks: {'no_overflow': True, 'zero_fill_preserved': True, 'p999_error_px': False, 'max_error_px': True, 'zscore_features': True, 'rf_statistics': True} → **FAIL**

