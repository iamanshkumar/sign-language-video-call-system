# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | shower | 6.24 | 9.26 | wrong | sun | 0.671 | True | False | [] | 0.46 | 0 | 0.65 |  |
| 1 | more | 14.28 | 17.28 | correct | more | 0.995 | True | False | [] | 0.43 | 0 | 0.57 |  |
| 2 | forbid | 22.30 | 25.30 | wrong | graduate | 0.322 | False | False | [] | 0.21 | 0 | 0.72 |  |
| 3 | africa | 30.30 | 33.34 | wrong | bank | 0.347 | False | False | [] | 0.48 | 0 | 0.61 |  |
| 4 | computer | 38.35 | 41.36 | wrong | long | 0.75 | False | False | [] | 0.03 | 1 | 0.81 |  |
| 5 | africa | 46.37 | 49.37 | wrong | history | 0.367 | False | False | [] | 0.45 | 0 | 0.55 |  |
| 6 | please | 54.38 | 57.40 | correct | please | 0.942 | True | False | [] | 0.49 | 0 | 0.64 |  |
| 7 | shower | 62.40 | 65.44 | wrong | sun | 0.753 | True | False | [] | 0.45 | 0 | 0.59 |  |
| 8 | here | 70.44 | 73.44 | wrong | japan | 0.514 | True | False | [] | 0.46 | 0 | 0.64 |  |
| 9 | tuesday | 78.44 | 81.47 | wrong | yes | 0.463 | True | False | [] | 0.37 | 0 | 0.67 |  |
| 10 | more | 86.47 | 89.50 | correct | more | 0.995 | True | False | [] | 0.43 | 0 | 0.59 |  |
| 11 | computer | 94.50 | 97.77 | wrong | diamond | 0.689 | False | False | ['long'] | 0.04 | 1 | 0.84 |  |
| 12 | here | 102.77 | 105.80 | wrong | forbid | 0.4 | False | False | [] | 0.49 | 0 | 0.55 |  |
| 13 | tuesday | 110.82 | 113.82 | correct | tuesday | 0.221 | True | False | [] | 0.04 | 1 | 0.73 |  |
| 14 | forbid | 118.84 | 121.86 | wrong | school | 0.602 | False | False | [] | 0.44 | 0 | 0.49 |  |
| 15 | please | 126.87 | 129.89 | correct | please | 0.987 | True | False | [] | 0.44 | 0 | 0.49 |  |

```json
{
 "attempts": 16,
 "outcomes": {
  "wrong": 11,
  "correct": 5
 },
 "accuracy_all_attempts": 0.3125,
 "accuracy_when_answered": 0.3125,
 "top5_rate_when_answered": 0.5625,
 "attempts_with_duplicates": 3,
 "early_events_total": 1,
 "pre_go_events_total": 0,
 "suppressed_events_total": 0,
 "attempts_with_hands_up_before_go": 0,
 "legacy_accuracy_all_attempts": 0.3125,
 "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
 "rejection_reasons": {},
 "delay_s": {
  "n": 16,
  "median": 0.4386151455109939,
  "p95": 0.48806892699940363
 },
 "confidence_correct_median": 0.9871814846992493,
 "confidence_wrong_median": 0.5144686102867126,
 "mean_hand_visible_fraction": 0.6343597338264164,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "closer, upper body in frame, hands rest out of view",
 "notes": "session 5c: condition REPEATED (3 cycles), repetition protocol v1",
 "model_dir": "models/candidates/hands_pose_w30_bilstm_e100_s3",
 "model": "hands_pose_w30_bilstm_e100_s3",
 "settings": {
  "pre_roll_s": 0.5,
  "end_rest_s": 0.4,
  "max_segment_s": 4.0,
  "confidence_threshold": 0.0,
  "smoothing_window": 1,
  "smoothing_min_votes": 1,
  "cooldown_s": 0.0
 },
 "frame_size": [
  640,
  360
 ],
 "frames": 4025,
 "dropped_frames": 28,
 "camera_fps_measured": 29.85183573525832,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "camera_frames_captured": 4054,
 "capture_options": {
  "threaded": true,
  "preview_every": 3,
  "hands_down_s": 0.5,
  "max_wait_s": 6.0
 },
 "framing_at_end": "framing: shoulders 20% of width (training 26-42%) - move closer",
 "input_diagnostics": {
  "frames": 4025,
  "frame_interval_ms": {
   "median": 33.33152100094594,
   "p95": 38.74653133971151,
   "share_over_50ms": 0.027087475149105367
  },
  "processed_fps": 30.00163118783629,
  "pose_present": 1.0,
  "any_hand_present": 0.39279503105590063,
  "both_hands": 0.1565217391304348,
  "left_hand_only": 0.031055900621118012,
  "right_hand_only": 0.20521739130434782,
  "shoulder_width_over_frame_width_median": 0.19756311178207397,
  "read_wait_ms_median": 0.05337499896995723,
  "loop_ms_median": 33.33152100094594
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
 "capture_fps_median": 30.00163118783629,
 "processed_fps_median": 30.00163118783629,
 "mediapipe_ms": {
  "median": 28.057458985131234,
  "p95": 28.35058358614333
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-20T06:41:02+00:00"
}
```
