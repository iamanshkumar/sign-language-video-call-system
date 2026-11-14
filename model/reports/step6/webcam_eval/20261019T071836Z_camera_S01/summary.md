# Live evaluation - REAL-WORLD webcam session

| attempt | expected | outcome | predicted | confidence | top-5 hit | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|
| 0 | africa | wrong | religion | 0.493 | False | 0.49 | 1 | 0.55 |  |
| 1 | africa | wrong | pepper | 0.771 | False | 0.44 | 0 | 0.49 |  |
| 2 | blind | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 3 | blind | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 4 | computer | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 5 | computer | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 6 | director | rejected | None |  | False | 0.43 | 0 | 0.24 | too_few_decoded_frames |
| 7 | director | wrong | history | 0.492 | False | 0.5 | 0 | 0.28 |  |
| 8 | forbid | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 9 | forbid | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 10 | here | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 11 | here | wrong | day | 0.499 | False | 0.09 | 2 | 0.96 |  |
| 12 | more | missed | None |  | False |  | 0 | 0.01 | no segment detected |
| 13 | more | wrong | blind | 0.767 | False | 0.5 | 0 | 0.32 |  |
| 14 | please | rejected | None |  | False | 0.43 | 0 | 0.57 | active_both_hands_missing_fraction |
| 15 | please | wrong | pepper | 0.664 | False | 0.5 | 1 | 0.49 |  |
| 16 | shower | wrong | onion | 0.931 | False | 0.44 | 0 | 0.41 |  |
| 17 | shower | wrong | delicious | 0.961 | False | 0.04 | 1 | 0.32 |  |
| 18 | tuesday | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 19 | tuesday | missed | None |  | False |  | 0 | 0.00 | no segment detected |

```json
{
 "attempts": 20,
 "outcomes": {
  "wrong": 8,
  "missed": 10,
  "rejected": 2
 },
 "accuracy_all_attempts": 0.0,
 "accuracy_when_answered": 0.0,
 "top5_rate_when_answered": 0.0,
 "attempts_with_duplicates": 4,
 "rejection_reasons": {
  "too_few_decoded_frames": 1,
  "active_both_hands_missing_fraction": 1
 },
 "delay_s": {
  "n": 10,
  "median": 0.43751024949597195,
  "p95": 0.50328360220592
 },
 "confidence_correct_median": null,
 "confidence_wrong_median": 0.7155261635780334,
 "mean_hand_visible_fraction": 0.23301801801801805,
 "mode": "camera",
 "real_world": true,
 "signer_id": "S01",
 "lighting": "daylight, frontal",
 "camera_position": "laptop, arm's length",
 "notes": "",
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
  1920,
  1080
 ],
 "frames": 2413,
 "dropped_frames": 0,
 "capture_fps_median": 14.9955294926846,
 "mediapipe_ms": {
  "median": 29.963665991090238,
  "p95": 32.7492831973359
 },
 "events_outside_attempts": 7,
 "created_utc": "2026-10-19T07:18:36+00:00"
}
```
