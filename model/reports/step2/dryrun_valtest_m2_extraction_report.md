# Step 2 extraction run `dryrun_valtest_m2`

Planned 12 videos; records 12; integrity OK. Store: `data/landmarks/dryrun_valtest_m2_store/`. Hosts: ['{"cpus": 8, "hostname_hash": "07539ad3198f", "machine": "arm64", "platform": "macOS-26.6.2-arm64-arm-64bit", "python": "3.12.13"}']

| split | videos | accepted | accepted % | short active | classes | MB | rejections |
|---|---|---|---|---|---|---|---|
| test | 6 | 6 | 100.0 | 2 | 6 | 2.2 | {} |
| val | 6 | 6 | 100.0 | 4 | 6 | 1.6 | {} |

Decode errors: 0; unreferenced arrays from interrupted batches (ignored): 0.

Parity vs M2 benchmark extraction: {"videos": 3, "same_decision": 3, "same_interval": 3, "summary": {"pose_max_abs_diff": {"count": 3.0, "mean": 0.0, "std": 0.0, "min": 0.0, "25%": 0.0, "50%": 0.0, "75%": 0.0, "max": 0.0}, "pose_mean_abs_diff": {"count": 3.0, "mean": 0.0, "std": 0.0, "min": 0.0, "25%": 0.0, "50%": 0.0, "75%": 0.0, "max": 0.0}, "face_max_abs_diff": {"count": 3.0, "mean": 0.00049, "std": 0.0, "min": 0.00049, "25%": 0.00049, "50%": 0.00049, "75%": 0.00049, "max": 0.00049}, "face_mean_abs_diff": {"count": 3.0, "mean": 0.0, "std": 0.0, "min": 0.0, "25%": 0.0, "50%": 0.0, "75%": 0.0, "max": 0.0}, "left_hand_max_abs_diff": {"count": 3.0, "mean": 8e-05, "std": 0.00014, "min": 0.0, "25%": 0.0, "50%": 0.0, "75%": 0.00012, "max": 0.00024}, "left_hand_mean_abs_diff": {"count": 3.0, "mean": 0.0, "std": 0.0, "min": 0.0, "25%": 0.0, "50%": 0.0, "75%": 0.0, "max": 0.0}, "right_hand_max_abs_diff": {"count": 3.0, "mean": 0.0, "std": 0.0, "min": 0.0, "25%": 0.0, "50%": 0.0, "75%": 0.0, "max": 0.0}, "right_hand_mean_abs_diff": {"count": 3.0, "mean": 0.0, "std": 0.0, "min": 0.0, "25%": 0.0, "50%": 0.0, "75%": 0.0, "max": 0.0}, "observed_flag_agreement": {"count": 3.0, "mean": 1.0, "std": 0.0, "min": 1.0, "25%": 1.0, "50%": 1.0, "75%": 1.0, "max": 1.0}}}

