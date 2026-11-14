# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | shower | 5.55 | 8.55 | wrong | sun | 0.761 | True | False | [] | 0.44 | 1 | 0.73 |  |
| 1 | more | 13.57 | 16.62 | correct | more | 0.515 | True | False | [] | 0.23 | 0 | 0.69 |  |
| 2 | forbid | 21.62 | 24.62 | wrong | school | 0.673 | False | False | [] | 0.44 | 0 | 0.58 |  |
| 3 | africa | 29.62 | 32.65 | wrong | history | 0.355 | False | False | ['pepper'] | 0.47 | 0 | 0.67 |  |
| 4 | computer | 37.67 | 40.68 | wrong | church | 0.337 | False | False | [] | 0.13 | 0 | 0.58 |  |
| 5 | africa | 45.68 | 48.72 | wrong | history | 0.317 | False | False | [] | 0.44 | 0 | 0.55 |  |
| 6 | please | 53.75 | 57.02 | correct | please | 0.605 | True | False | [] | 0.45 | 0 | 0.59 |  |
| 7 | shower | 62.02 | 65.27 | wrong | sun | 0.803 | True | False | [] | 0.45 | 0 | 0.55 |  |
| 8 | here | 70.32 | 73.34 | wrong | impossible | 0.169 | False | False | [] | 0.44 | 0 | 0.62 |  |
| 9 | tuesday | 78.35 | 81.37 | wrong | yes | 0.259 | False | False | [] | 0.17 | 0 | 0.66 |  |
| 10 | more | 86.38 | 89.75 | correct | more | 0.857 | True | False | [] | 0.47 | 0 | 0.53 |  |
| 11 | computer | 94.77 | 97.78 | wrong | chocolate | 0.746 | False | False | [] | 0.04 | 1 | 0.77 |  |
| 12 | here | 102.79 | 105.82 | correct | here | 0.862 | True | False | [] | 0.44 | 0 | 0.56 |  |
| 13 | tuesday | 110.82 | 113.82 | wrong | yes | 0.151 | False | False | [] | 0.47 | 0 | 0.62 |  |
| 14 | forbid | 118.83 | 121.85 | wrong | church | 0.954 | False | False | [] | 0.46 | 0 | 0.62 |  |
| 15 | please | 126.85 | 129.88 | wrong | who | 0.162 | True | False | [] | 0.47 | 0 | 0.57 |  |

```json
{
 "attempts": 16,
 "outcomes": {
  "wrong": 12,
  "correct": 4
 },
 "accuracy_all_attempts": 0.25,
 "accuracy_when_answered": 0.25,
 "top5_rate_when_answered": 0.4375,
 "attempts_with_duplicates": 2,
 "early_events_total": 1,
 "pre_go_events_total": 0,
 "suppressed_events_total": 0,
 "attempts_with_hands_up_before_go": 0,
 "legacy_accuracy_all_attempts": 0.25,
 "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
 "rejection_reasons": {},
 "delay_s": {
  "n": 16,
  "median": 0.44134837499586865,
  "p95": 0.47134111449850025
 },
 "confidence_correct_median": 0.7309483885765076,
 "confidence_wrong_median": 0.34611403942108154,
 "mean_hand_visible_fraction": 0.617760067114094,
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
  480
 ],
 "frames": 4052,
 "dropped_frames": 19,
 "camera_fps_measured": 30.00296547240931,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "camera_frames_captured": 4073,
 "capture_options": {
  "threaded": true,
  "preview_every": 3,
  "hands_down_s": 0.5,
  "max_wait_s": 6.0
 },
 "framing_at_end": "framing: shoulders 18% of width (training 26-42%) - move closer",
 "input_diagnostics": {
  "frames": 4052,
  "frame_interval_ms": {
   "median": 33.59999999520369,
   "p95": 37.62975001882296,
   "share_over_50ms": 0.0004937052579609973
  },
  "processed_fps": 29.761904766153194,
  "pose_present": 1.0,
  "any_hand_present": 0.4178183613030602,
  "both_hands": 0.16683119447186576,
  "left_hand_only": 0.03923988153998026,
  "right_hand_only": 0.21174728529121423,
  "shoulder_width_over_frame_width_median": 0.17537140846252441,
  "read_wait_ms_median": 0.08747899846639484,
  "loop_ms_median": 33.59999999520369
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
 "capture_fps_median": 29.761904766153194,
 "processed_fps_median": 29.761904766153194,
 "mediapipe_ms": {
  "median": 27.721374994143844,
  "p95": 28.151320660253987
 },
 "events_outside_attempts": 2,
 "created_utc": "2026-10-20T06:43:53+00:00"
}
```
