# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | please | 5.58 | 8.63 | correct | please | 0.995 | True | False | [] | 0.45 | 0 | 0.48 |  |
| 1 | shower | 13.64 | 16.66 | wrong | sun | 0.875 | True | False | [] | 0.45 | 0 | 0.68 |  |
| 2 | computer | 21.67 | 24.70 | wrong | diamond | 0.79 | False | False | [] | 0.03 | 1 | 0.81 |  |
| 3 | forbid | 29.71 | 32.76 | wrong | school | 0.421 | False | False | [] | 0.46 | 0 | 0.53 |  |
| 4 | tuesday | 37.78 | 40.79 | wrong | sun | 0.269 | True | False | [] | 0.44 | 0 | 0.42 |  |
| 5 | here | 45.83 | 48.88 | correct | here | 0.477 | True | False | [] | 0.45 | 0 | 0.64 |  |
| 6 | forbid | 53.91 | 56.91 | wrong | school | 0.523 | False | False | [] | 0.45 | 0 | 0.52 |  |
| 7 | here | 61.91 | 64.95 | wrong | new | 0.449 | False | False | [] | 0.45 | 0 | 0.57 |  |
| 8 | more | 69.95 | 72.96 | correct | more | 0.752 | True | False | [] | 0.45 | 0 | 0.52 |  |
| 9 | africa | 78.00 | 81.00 | wrong | history | 0.365 | False | False | [] | 0.45 | 0 | 0.53 |  |
| 10 | more | 86.01 | 89.03 | correct | more | 0.892 | True | False | [] | 0.45 | 0 | 0.55 |  |
| 11 | shower | 94.04 | 97.07 | wrong | sun | 0.663 | True | False | [] | 0.45 | 0 | 0.64 |  |
| 12 | tuesday | 102.13 | 105.13 | wrong | government | 0.231 | True | False | [] | 0.46 | 0 | 0.61 |  |
| 13 | please | 110.13 | 113.17 | correct | please | 0.977 | True | False | [] | 0.44 | 0 | 0.54 |  |
| 14 | africa | 118.20 | 121.20 | wrong | history | 0.432 | False | False | [] | 0.45 | 0 | 0.59 |  |
| 15 | computer | 126.20 | 129.26 | wrong | diamond | 0.844 | False | False | [] | 0.04 | 0 | 0.77 |  |

```json
{
 "attempts": 16,
 "outcomes": {
  "correct": 5,
  "wrong": 11
 },
 "accuracy_all_attempts": 0.3125,
 "accuracy_when_answered": 0.3125,
 "top5_rate_when_answered": 0.5625,
 "attempts_with_duplicates": 1,
 "early_events_total": 0,
 "pre_go_events_total": 0,
 "suppressed_events_total": 0,
 "attempts_with_hands_up_before_go": 0,
 "legacy_accuracy_all_attempts": 0.3125,
 "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
 "rejection_reasons": {},
 "delay_s": {
  "n": 16,
  "median": 0.45018500002333894,
  "p95": 0.4561918332474306
 },
 "confidence_correct_median": 0.8924892544746399,
 "confidence_wrong_median": 0.44929346442222595,
 "mean_hand_visible_fraction": 0.5875,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "closer, upper body in frame, hands rest out of view",
 "notes": "session 5d: condition SINGLE (one movement cycle), repetition protocol v1",
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
 "frames": 3227,
 "dropped_frames": 17,
 "camera_fps_measured": 24.007277171677256,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "camera_frames_captured": 3245,
 "capture_options": {
  "threaded": true,
  "preview_every": 3,
  "hands_down_s": 0.5,
  "max_wait_s": 6.0
 },
 "framing_at_end": "framing: shoulders 20% of width (training 26-42%) - move closer",
 "input_diagnostics": {
  "frames": 3227,
  "frame_interval_ms": {
   "median": 41.88891701051034,
   "p95": 47.33726025006035,
   "share_over_50ms": 0.008989460632362058
  },
  "processed_fps": 23.872663018456414,
  "pose_present": 1.0,
  "any_hand_present": 0.36132630926557174,
  "both_hands": 0.14564611093895258,
  "left_hand_only": 0.02107220328478463,
  "right_hand_only": 0.19460799504183451,
  "shoulder_width_over_frame_width_median": 0.20447039604187012,
  "read_wait_ms_median": 10.097000020323321,
  "loop_ms_median": 41.88891701051034
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
 "capture_fps_median": 23.872663018456414,
 "processed_fps_median": 23.872663018456414,
 "mediapipe_ms": {
  "median": 28.138791996752843,
  "p95": 28.49936680868268
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-20T06:46:45+00:00"
}
```
