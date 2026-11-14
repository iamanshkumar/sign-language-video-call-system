# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tuesday | 5.62 | 8.64 | wrong | wednesday | 0.244 | True | False | [] | 0.29 | 0 | 0.74 |  |
| 1 | here | 13.67 | 16.67 | wrong | impossible | 0.377 | False | False | [] | 0.47 | 0 | 0.50 |  |
| 2 | forbid | 21.68 | 24.68 | wrong | long | 0.407 | False | False | [] | 0.47 | 0 | 0.49 |  |
| 3 | please | 29.71 | 32.71 | correct | please | 0.992 | True | False | [] | 0.46 | 0 | 0.63 |  |
| 4 | tuesday | 37.71 | 40.74 | wrong | yes | 0.362 | True | False | [] | 0.46 | 0 | 0.43 |  |
| 5 | computer | 45.74 | 48.77 | wrong | long | 0.379 | True | False | [] | 0.03 | 1 | 0.83 |  |
| 6 | africa | 53.77 | 56.77 | wrong | history | 0.365 | False | False | [] | 0.44 | 0 | 0.53 |  |
| 7 | more | 61.82 | 64.87 | correct | more | 0.993 | True | False | [] | 0.45 | 0 | 0.59 |  |
| 8 | here | 69.87 | 72.89 | wrong | scared | 0.486 | False | False | [] | 0.43 | 0 | 0.71 |  |
| 9 | shower | 77.90 | 80.93 | wrong | sun | 0.832 | True | False | [] | 0.47 | 0 | 0.64 |  |
| 10 | more | 85.95 | 88.96 | correct | more | 0.988 | True | False | [] | 0.45 | 0 | 0.28 |  |
| 11 | forbid | 93.99 | 96.99 | wrong | cookie | 0.402 | False | False | [] | 0.43 | 0 | 0.63 |  |
| 12 | computer | 102.02 | 105.03 | wrong | church | 0.471 | False | False | [] | 0.03 | 1 | 0.85 |  |
| 13 | please | 110.04 | 113.06 | correct | please | 0.989 | True | False | [] | 0.47 | 0 | 0.61 |  |
| 14 | shower | 118.07 | 121.09 | wrong | sun | 0.779 | True | False | [] | 0.46 | 0 | 0.71 |  |
| 15 | africa | 126.11 | 129.11 | wrong | history | 0.397 | False | False | [] | 0.47 | 0 | 0.61 |  |

```json
{
 "attempts": 16,
 "outcomes": {
  "wrong": 12,
  "correct": 4
 },
 "accuracy_all_attempts": 0.25,
 "accuracy_when_answered": 0.25,
 "top5_rate_when_answered": 0.5625,
 "attempts_with_duplicates": 2,
 "early_events_total": 0,
 "pre_go_events_total": 0,
 "suppressed_events_total": 0,
 "attempts_with_hands_up_before_go": 0,
 "legacy_accuracy_all_attempts": 0.25,
 "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
 "rejection_reasons": {},
 "delay_s": {
  "n": 16,
  "median": 0.45353683349094354,
  "p95": 0.4725950829961221
 },
 "confidence_correct_median": 0.9909105002880096,
 "confidence_wrong_median": 0.3993333429098129,
 "mean_hand_visible_fraction": 0.6131836870427474,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "closer, upper body in frame, hands rest out of view",
 "notes": "session 5b: condition REPEATED (3 cycles), repetition protocol v1",
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
 "frames": 4013,
 "dropped_frames": 36,
 "camera_fps_measured": 30.004887710065066,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "camera_frames_captured": 4051,
 "capture_options": {
  "threaded": true,
  "preview_every": 3,
  "hands_down_s": 0.5,
  "max_wait_s": 6.0
 },
 "framing_at_end": "framing: shoulders 20% of width (training 26-42%) - move closer",
 "input_diagnostics": {
  "frames": 4013,
  "frame_interval_ms": {
   "median": 33.37812550307717,
   "p95": 38.83201060525607,
   "share_over_50ms": 0.025174476570289133
  },
  "processed_fps": 29.959741145673647,
  "pose_present": 1.0,
  "any_hand_present": 0.3755295290306504,
  "both_hands": 0.15175679043109894,
  "left_hand_only": 0.0224271118863693,
  "right_hand_only": 0.20134562671318215,
  "shoulder_width_over_frame_width_median": 0.19519028067588806,
  "read_wait_ms_median": 0.05549998604692519,
  "loop_ms_median": 33.37812550307717
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
 "capture_fps_median": 29.959741145673647,
 "processed_fps_median": 29.959741145673647,
 "mediapipe_ms": {
  "median": 28.05091699701734,
  "p95": 28.38700019638054
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-20T06:38:12+00:00"
}
```
