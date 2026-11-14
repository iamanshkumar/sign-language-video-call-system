# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | here | 6.07 | 9.08 | wrong | japan | 0.446 | True | False | [] | 0.5 | 0 | 0.58 |  |
| 1 | shower | 14.10 | 17.11 | correct | shower | 0.549 | True | False | [] | 0.43 | 0 | 0.63 |  |
| 2 | tuesday | 22.13 | 25.13 | correct | tuesday | 0.263 | True | False | ['religion'] | 0.4 | 0 | 0.71 |  |
| 3 | please | 30.13 | 33.17 | correct | please | 0.997 | True | False | [] | 0.45 | 0 | 0.61 |  |
| 4 | forbid | 38.20 | 41.20 | wrong | graduate | 0.526 | False | False | [] | 0.43 | 0 | 0.52 |  |
| 5 | shower | 46.20 | 49.23 | correct | shower | 0.52 | True | False | [] | 0.47 | 0 | 0.56 |  |
| 6 | more | 54.23 | 57.24 | correct | more | 0.986 | True | False | [] | 0.47 | 0 | 0.60 |  |
| 7 | forbid | 62.27 | 65.27 | wrong | day | 0.519 | False | False | [] | 0.47 | 0 | 0.53 |  |
| 8 | africa | 70.27 | 73.30 | wrong | history | 0.244 | False | False | [] | 0.46 | 0 | 0.50 |  |
| 9 | computer | 78.30 | 81.48 | wrong | church | 0.55 | False | False | ['long'] | 0.47 | 0 | 0.72 |  |
| 10 | africa | 86.48 | 89.48 | wrong | pepper | 0.54 | False | False | [] | 0.44 | 0 | 0.49 |  |
| 11 | here | 94.50 | 98.08 | wrong | japan | 0.577 | True | False | ['here'] | 0.44 | 0 | 0.51 |  |
| 12 | more | 103.10 | 106.13 | correct | more | 0.995 | True | False | [] | 0.44 | 0 | 0.71 |  |
| 13 | computer | 111.14 | 114.17 | wrong | long | 0.472 | False | False | [] | 0.03 | 1 | 0.75 |  |
| 14 | please | 119.18 | 122.18 | correct | please | 0.978 | True | False | [] | 0.47 | 0 | 0.55 |  |
| 15 | tuesday | 127.18 | 130.18 | correct | tuesday | 0.254 | True | False | [] | 0.44 | 0 | 0.63 |  |

```json
{
 "attempts": 16,
 "outcomes": {
  "wrong": 8,
  "correct": 8
 },
 "accuracy_all_attempts": 0.5,
 "accuracy_when_answered": 0.5,
 "top5_rate_when_answered": 0.625,
 "attempts_with_duplicates": 1,
 "early_events_total": 3,
 "pre_go_events_total": 0,
 "suppressed_events_total": 0,
 "attempts_with_hands_up_before_go": 0,
 "legacy_accuracy_all_attempts": 0.5,
 "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
 "rejection_reasons": {},
 "delay_s": {
  "n": 16,
  "median": 0.4476011669903528,
  "p95": 0.4810093120104284
 },
 "confidence_correct_median": 0.7630394101142883,
 "confidence_wrong_median": 0.5224958062171936,
 "mean_hand_visible_fraction": 0.6008668903803133,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "closer, upper body in frame, hands rest out of view",
 "notes": "session 5a: condition SINGLE (one movement cycle), repetition protocol v1",
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
 "frames": 3988,
 "dropped_frames": 48,
 "camera_fps_measured": 29.675985121049973,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "camera_frames_captured": 4038,
 "capture_options": {
  "threaded": true,
  "preview_every": 3,
  "hands_down_s": 0.5,
  "max_wait_s": 6.0
 },
 "framing_at_end": "framing: shoulders 21% of width (training 26-42%) - move closer",
 "input_diagnostics": {
  "frames": 3988,
  "frame_interval_ms": {
   "median": 33.364958013407886,
   "p95": 36.946429396630265,
   "share_over_50ms": 0.005517933283170303
  },
  "processed_fps": 29.971564765588635,
  "pose_present": 0.9849548645937813,
  "any_hand_present": 0.37612838515546637,
  "both_hands": 0.16123370110330992,
  "left_hand_only": 0.024072216649949848,
  "right_hand_only": 0.19082246740220662,
  "shoulder_width_over_frame_width_median": 0.20990033447742462,
  "read_wait_ms_median": 0.07845798972994089,
  "loop_ms_median": 33.364958013407886
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
 "capture_fps_median": 29.971564765588635,
 "processed_fps_median": 29.971564765588635,
 "mediapipe_ms": {
  "median": 28.045396000379696,
  "p95": 28.374203950806987
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-20T06:35:35+00:00"
}
```
