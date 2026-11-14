# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | africa | 5.72 | 8.74 | wrong | grandfather | 0.679 | False | False | [] | 0.53 | 0 | 0.43 |  |
| 1 | africa | 13.75 | 16.76 | wrong | bank | 0.719 | False | False | [] | 0.44 | 0 | 0.48 |  |
| 2 | blind | 21.79 | 26.19 | correct | blind | 0.749 | True | False | ['government'] | 0.44 | 0 | 0.64 |  |
| 3 | blind | 31.19 | 34.21 | correct | blind | 0.805 | True | False | [] | 0.43 | 0 | 0.51 |  |
| 4 | computer | 39.22 | 42.22 | wrong | church | 0.43 | False | False | [] | 0.07 | 0 | 0.75 |  |
| 5 | computer | 47.23 | 50.27 | wrong | summer | 0.2 | False | False | [] | 0.18 | 0 | 0.78 |  |
| 6 | director | 55.28 | 58.28 | wrong | awake | 0.498 | False | False | [] | 0.44 | 0 | 0.57 |  |
| 7 | director | 63.29 | 66.30 | wrong | strong | 0.556 | True | False | [] | 0.44 | 0 | 0.54 |  |
| 8 | forbid | 71.32 | 74.34 | wrong | church | 0.39 | False | False | [] | 0.04 | 1 | 0.81 |  |
| 9 | forbid | 79.36 | 82.40 | wrong | school | 0.977 | False | False | [] | 0.44 | 0 | 0.37 |  |
| 10 | here | 87.40 | 90.43 | correct | here | 0.981 | True | False | [] | 0.44 | 0 | 0.60 |  |
| 11 | here | 95.44 | 98.44 | wrong | scared | 0.283 | True | False | [] | 0.44 | 0 | 0.57 |  |
| 12 | more | 103.44 | 106.44 | correct | more | 0.977 | True | False | [] | 0.44 | 0 | 0.61 |  |
| 13 | more | 111.45 | 114.48 | correct | more | 0.977 | True | False | [] | 0.44 | 0 | 0.63 |  |
| 14 | please | 119.49 | 122.49 | correct | please | 0.996 | True | False | [] | 0.34 | 0 | 0.75 |  |
| 15 | please | 127.50 | 130.50 | correct | please | 0.988 | True | False | [] | 0.43 | 0 | 0.55 |  |
| 16 | shower | 135.50 | 138.53 | correct | shower | 0.917 | True | False | [] | 0.44 | 0 | 0.60 |  |
| 17 | shower | 143.54 | 146.58 | correct | shower | 0.476 | True | False | [] | 0.44 | 0 | 0.54 |  |
| 18 | tuesday | 151.59 | 154.61 | wrong | yes | 0.277 | True | False | [] | 0.04 | 1 | 0.81 |  |
| 19 | tuesday | 159.61 | 162.62 | wrong | yes | 0.404 | False | False | [] | 0.44 | 0 | 0.49 |  |

```json
{
 "attempts": 20,
 "outcomes": {
  "wrong": 11,
  "correct": 9
 },
 "accuracy_all_attempts": 0.45,
 "accuracy_when_answered": 0.45,
 "top5_rate_when_answered": 0.6,
 "attempts_with_duplicates": 2,
 "early_events_total": 1,
 "pre_go_events_total": 0,
 "suppressed_events_total": 0,
 "attempts_with_hands_up_before_go": 0,
 "legacy_accuracy_all_attempts": 0.45,
 "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
 "rejection_reasons": {},
 "delay_s": {
  "n": 20,
  "median": 0.43713445849425625,
  "p95": 0.44851429374248264
 },
 "confidence_correct_median": 0.9772516489028931,
 "confidence_wrong_median": 0.4304455816745758,
 "mean_hand_visible_fraction": 0.6019191003083619,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "closer, upper body in frame, hands rest out of view",
 "notes": "session 4: controlled protocol - hands out during WAIT, sign once after GO, lower hands immediately",
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
  480
 ],
 "frames": 5014,
 "dropped_frames": 36,
 "camera_fps_measured": 29.97386381580979,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "camera_frames_captured": 5052,
 "capture_options": {
  "threaded": true,
  "preview_every": 3,
  "hands_down_s": 0.5,
  "max_wait_s": 6.0
 },
 "framing_at_end": "framing: shoulders 28% of width (training 26-42%) - OK",
 "input_diagnostics": {
  "frames": 5014,
  "frame_interval_ms": {
   "median": 33.44462500535883,
   "p95": 34.6337579889223,
   "share_over_50ms": 0.0015958507879513266
  },
  "processed_fps": 29.90017079993482,
  "pose_present": 1.0,
  "any_hand_present": 0.3721579577183885,
  "both_hands": 0.15496609493418428,
  "left_hand_only": 0.01814918228958915,
  "right_hand_only": 0.19904268049461507,
  "shoulder_width_over_frame_width_median": 0.27696657180786133,
  "read_wait_ms_median": 0.07574999472126365,
  "loop_ms_median": 33.44462500535883
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
 "capture_fps_median": 29.90017079993482,
 "processed_fps_median": 29.90017079993482,
 "mediapipe_ms": {
  "median": 28.57058348308783,
  "p95": 29.573545485618524
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-20T04:59:03+00:00"
}
```
