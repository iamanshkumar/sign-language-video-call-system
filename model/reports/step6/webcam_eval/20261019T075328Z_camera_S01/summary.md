# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | africa | 0.58 | 3.62 | missed | None |  | False | None | [] |  | 0 | 0.17 | no segment detected |
| 1 | africa | 8.69 | 11.75 | wrong | grandmother | 0.41 | True | True | ['africa'] | 0.44 | 1 | 0.52 |  |
| 2 | blind | 16.82 | 19.88 | wrong | fish | 0.492 | True | True | [] | 0.44 | 0 | 0.84 |  |
| 3 | blind | 24.92 | 27.99 | correct | blind | 0.78 | True | True | ['nose', None] | 0.44 | 0 | 0.56 |  |
| 4 | computer | 33.05 | 36.06 | wrong | church | 0.487 | False | True | ['blind'] | 0.04 | 0 | 1.00 |  |
| 5 | computer | 41.12 | 44.12 | wrong | church | 0.656 | False | True | ['day'] | 0.04 | 1 | 0.91 |  |
| 6 | director | 49.12 | 52.19 | wrong | strong | 0.506 | True | True | [] | 0.04 | 1 | 0.92 |  |
| 7 | director | 57.25 | 60.32 | correct | director | 0.728 | True | True | [] | 0.04 | 1 | 0.83 |  |
| 8 | forbid | 65.39 | 68.45 | wrong | day | 0.466 | False | True | [] | 0.04 | 0 | 0.96 |  |
| 9 | forbid | 73.45 | 76.45 | missed | None |  | False | None | ['school'] |  | 0 | 0.63 | no segment detected |
| 10 | here | 81.45 | 84.45 | wrong | forbid | 0.394 | False | True | ['church', None] | 0.5 | 1 | 0.81 |  |
| 11 | here | 89.52 | 92.55 | wrong | scared | 0.945 | True | True | [] | 0.5 | 1 | 0.49 |  |
| 12 | more | 97.60 | 100.66 | missed | None |  | False | None | ['scared'] |  | 0 | 0.61 | no segment detected |
| 13 | more | 105.69 | 108.75 | correct | more | 0.989 | True | False | ['more', None] | 0.24 | 0 | 0.77 |  |
| 14 | please | 113.76 | 116.82 | correct | please | 0.957 | True | True | [] | 0.24 | 0 | 0.81 |  |
| 15 | please | 121.89 | 124.96 | correct | please | 0.959 | True | True | ['please'] | 0.48 | 0 | 0.58 |  |
| 16 | shower | 129.99 | 132.99 | wrong | sun | 0.846 | True | True | [] | 0.04 | 1 | 0.77 |  |
| 17 | shower | 137.99 | 141.06 | correct | shower | 0.755 | True | False | ['shower'] | 0.04 | 0 | 0.91 |  |
| 18 | tuesday | 146.09 | 149.09 | correct | tuesday | 0.42 | True | True | [None] | 0.17 | 0 | 0.73 |  |
| 19 | tuesday | 154.09 | 157.09 | correct | tuesday | 0.358 | True | False | ['aunt'] | 0.17 | 0 | 0.68 |  |

```json
{
 "attempts": 20,
 "outcomes": {
  "missed": 3,
  "wrong": 9,
  "correct": 8
 },
 "accuracy_all_attempts": 0.4,
 "accuracy_when_answered": 0.47058823529411764,
 "top5_rate_when_answered": 0.7647058823529411,
 "attempts_with_duplicates": 7,
 "early_events_total": 15,
 "answers_started_before_go": 14,
 "rejection_reasons": {},
 "delay_s": {
  "n": 17,
  "median": 0.17208529100753367,
  "p95": 0.5029046920128166
 },
 "confidence_correct_median": 0.7676856815814972,
 "confidence_wrong_median": 0.4918232858181,
 "mean_hand_visible_fraction": 0.7244446404709561,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "laptop, arm's length, upper body and both hands in frame",
 "notes": "session 2 after evaluation fixes",
 "model_dir": "models/candidates/hands_pose_w30_bilstm_e100_s3",
 "model": "hands_pose_w30_bilstm_e100_s3",
 "settings": {
  "pre_roll_s": 0.5,
  "end_rest_s": 0.4,
  "max_segment_s": 4.0,
  "confidence_threshold": 0.0,
  "smoothing_window": 1,
  "smoothing_min_votes": 1
 },
 "frame_size": [
  640,
  360
 ],
 "frames": 2433,
 "dropped_frames": null,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "input_diagnostics": {
  "frames": 2433,
  "frame_interval_ms": {
   "median": 66.6272289963672,
   "p95": 70.1582480862271,
   "share_over_50ms": 0.993421052631579
  },
  "processed_fps": 15.00887872816269,
  "pose_present": 0.9958898479243732,
  "any_hand_present": 0.6292642827784628,
  "both_hands": 0.2655158240854912,
  "left_hand_only": 0.04356761200164406,
  "right_hand_only": 0.32018084669132757,
  "shoulder_width_over_frame_width_median": 0.2247197926044464,
  "read_wait_ms_median": 23.0745000008028,
  "loop_ms_median": 66.6272289963672
 },
 "training_reference": {
  "shoulder_width_over_frame_width_median": 0.371,
  "inside_active_interval": {
   "frames_inside_active_interval": 206425,
   "left_hand_missing_only": 72697,
   "right_hand_missing_only": 18675,
   "both_hands_missing": 1090,
   "pose_missing": 0
  }
 },
 "capture_fps_median": 15.00887872816269,
 "mediapipe_ms": {
  "median": 31.397625018144026,
  "p95": 32.30969179421663
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-19T07:53:28+00:00"
}
```
