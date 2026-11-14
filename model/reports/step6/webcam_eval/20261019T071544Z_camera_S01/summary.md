# Live evaluation - REAL-WORLD webcam session

| attempt | expected | outcome | predicted | confidence | top-5 hit | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|
| 0 | africa | missed | None |  | False |  | 0 | 0.01 | no segment detected |
| 1 | africa | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 2 | blind | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 3 | blind | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 4 | computer | wrong | secret | 0.894 | False | 0.08 | 0 | 1.00 |  |
| 5 | computer | wrong | frog | 0.457 | False | 0.16 | 0 | 1.00 |  |
| 6 | director | wrong | who | 0.454 | False | 0.47 | 0 | 0.11 |  |
| 7 | director | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 8 | forbid | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 9 | forbid | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 10 | here | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 11 | here | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 12 | more | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 13 | more | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 14 | please | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 15 | please | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 16 | shower | wrong | jealous | 0.577 | False | 0.5 | 0 | 0.45 |  |
| 17 | shower | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 18 | tuesday | missed | None |  | False |  | 0 | 0.00 | no segment detected |
| 19 | tuesday | missed | None |  | False |  | 0 | 0.00 | no segment detected |

```json
{
 "attempts": 20,
 "outcomes": {
  "missed": 16,
  "wrong": 4
 },
 "accuracy_all_attempts": 0.0,
 "accuracy_when_answered": 0.0,
 "top5_rate_when_answered": 0.0,
 "attempts_with_duplicates": 0,
 "rejection_reasons": {},
 "delay_s": {
  "n": 4,
  "median": 0.3126866870006779,
  "p95": 0.4960696730995551
 },
 "confidence_correct_median": null,
 "confidence_wrong_median": 0.5167855620384216,
 "mean_hand_visible_fraction": 0.12848055066364925,
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
 "frames": 2435,
 "dropped_frames": 0,
 "capture_fps_median": 15.014752031037725,
 "mediapipe_ms": {
  "median": 22.52200001385063,
  "p95": 33.64099128521047
 },
 "events_outside_attempts": 3,
 "created_utc": "2026-10-19T07:15:44+00:00"
}
```
