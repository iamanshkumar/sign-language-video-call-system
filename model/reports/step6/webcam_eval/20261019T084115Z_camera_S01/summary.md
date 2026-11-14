# Live evaluation - REAL-WORLD webcam session

| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | africa | 5.61 | 10.35 | correct | africa | 0.367 | True | False | ['africa'] | 0.47 | 0 | 0.50 |  |
| 1 | africa | 15.38 | 18.39 | correct | africa | 0.559 | True | False | ['fish'] | 0.47 | 0 | 0.68 |  |
| 2 | blind | 23.42 | 29.05 | correct | blind | 0.556 | True | False | ['blind'] | 0.44 | 0 | 0.48 |  |
| 3 | blind | 34.08 | 37.09 | correct | blind | 0.731 | True | False | [] | 0.03 | 0 | 0.84 |  |
| 4 | computer | 42.12 | 48.38 | wrong | church | 0.727 | False | False | ['blind', 'church'] | 0.04 | 0 | 0.83 |  |
| 5 | computer | 53.42 | 56.45 | wrong | church | 0.857 | False | False | ['brother'] | 0.03 | 0 | 0.85 |  |
| 6 | director | 61.45 | 65.18 | correct | director | 0.483 | True | False | ['church', 'shower'] | 0.45 | 0 | 0.69 |  |
| 7 | director | 70.18 | 73.22 | wrong | awake | 0.507 | True | False | [] | 0.47 | 0 | 0.71 |  |
| 8 | forbid | 78.22 | 81.25 | missed | None |  | False | None | [] |  | 0 | 0.64 | no segment detected |
| 9 | forbid | 86.27 | 89.31 | wrong | church | 0.295 | False | False | ['church'] | 0.04 | 0 | 0.82 |  |
| 10 | here | 94.33 | 98.42 | wrong | impossible | 0.415 | False | False | ['paint', 'here'] | 0.03 | 0 | 0.77 |  |
| 11 | here | 103.45 | 106.47 | wrong | scared | 0.375 | False | False | ['much'] | 0.04 | 0 | 0.81 |  |
| 12 | more | 111.49 | 114.52 | correct | more | 0.829 | True | False | ['boat'] | 0.43 | 0 | 0.65 |  |
| 13 | more | 119.52 | 122.55 | correct | more | 0.892 | True | False | [] | 0.43 | 0 | 0.71 |  |
| 14 | please | 127.55 | 130.58 | correct | please | 0.974 | True | False | [] | 0.04 | 1 | 0.83 |  |
| 15 | please | 135.58 | 138.63 | correct | please | 0.988 | True | False | [] | 0.04 | 0 | 0.85 |  |
| 16 | shower | 143.63 | 146.65 | correct | shower | 0.918 | True | False | ['please'] | 0.04 | 0 | 0.81 |  |
| 17 | shower | 151.66 | 154.68 | correct | shower | 0.936 | True | False | [] | 0.07 | 0 | 0.60 |  |
| 18 | tuesday | 159.72 | 162.72 | correct | tuesday | 0.679 | True | False | ['shower'] | 0.47 | 0 | 0.46 |  |
| 19 | tuesday | 167.73 | 170.73 | correct | tuesday | 0.457 | True | False | [] | 0.04 | 0 | 0.67 |  |

```json
{
 "attempts": 20,
 "outcomes": {
  "correct": 13,
  "wrong": 6,
  "missed": 1
 },
 "accuracy_all_attempts": 0.65,
 "accuracy_when_answered": 0.6842105263157895,
 "top5_rate_when_answered": 0.7368421052631579,
 "attempts_with_duplicates": 1,
 "early_events_total": 15,
 "pre_go_events_total": 0,
 "suppressed_events_total": 0,
 "attempts_with_hands_up_before_go": 0,
 "legacy_accuracy_all_attempts": 0.65,
 "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
 "rejection_reasons": {},
 "delay_s": {
  "n": 19,
  "median": 0.042712624999694526,
  "p95": 0.4726294953055913
 },
 "confidence_correct_median": 0.7305868864059448,
 "confidence_wrong_median": 0.46120627224445343,
 "mean_hand_visible_fraction": 0.7099405306902529,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "closer, upper body in frame, hands rest out of view",
 "notes": "session 3: threaded capture, gated GO",
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
 "frames": 5270,
 "dropped_frames": 28,
 "camera_fps_measured": 30.00275032517587,
 "camera_requested": {
  "width": 640,
  "height": 480,
  "fps": 30.0
 },
 "camera_reported_fps": 30.0,
 "camera_frames_captured": 5300,
 "capture_options": {
  "threaded": true,
  "preview_every": 3,
  "hands_down_s": 0.5,
  "max_wait_s": 6.0
 },
 "framing_at_end": "framing: shoulders 25% of width (training 26-42%) - move closer",
 "input_diagnostics": {
  "frames": 5270,
  "frame_interval_ms": {
   "median": 33.54916599346325,
   "p95": 38.12195839127525,
   "share_over_50ms": 0.0015183146707155057
  },
  "processed_fps": 29.807000275203293,
  "pose_present": 0.9954459203036053,
  "any_hand_present": 0.49240986717267554,
  "both_hands": 0.20683111954459202,
  "left_hand_only": 0.024667931688804556,
  "right_hand_only": 0.26091081593927895,
  "shoulder_width_over_frame_width_median": 0.2549734264612198,
  "read_wait_ms_median": 0.08160450670402497,
  "loop_ms_median": 33.54916599346325
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
 "capture_fps_median": 29.807000275203293,
 "processed_fps_median": 29.807000275203293,
 "mediapipe_ms": {
  "median": 28.05541700217873,
  "p95": 28.416837492841296
 },
 "events_outside_attempts": 1,
 "created_utc": "2026-10-19T08:41:15+00:00"
}
```
