# Live evaluation - SIMULATION (recorded clip replay) - not a real-world result

| attempt | expected | outcome | predicted | confidence | top-5 hit | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|
| 0 | bottle | correct | bottle | 0.999 | True | 0.49 | 0 | 0.59 |  |

```json
{
 "attempts": 1,
 "outcomes": {
  "correct": 1
 },
 "accuracy_all_attempts": 1.0,
 "accuracy_when_answered": 1.0,
 "top5_rate_when_answered": 1.0,
 "attempts_with_duplicates": 0,
 "rejection_reasons": {},
 "delay_s": {
  "n": 1,
  "median": 0.4880305830081926,
  "p95": 0.4880305830081926
 },
 "confidence_correct_median": 0.9993720650672913,
 "confidence_wrong_median": null,
 "mean_hand_visible_fraction": 0.5882352941176471,
 "mode": "video_replay",
 "real_world": false,
 "signer_id": "dataset-test-clip",
 "lighting": "",
 "camera_position": "",
 "notes": "SIMULATION replay of test clip part_1/025114512408390022-BOTTLE.mp4",
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
  480
 ],
 "frames": 102,
 "dropped_frames": 31,
 "capture_fps_median": 30.303030303030173,
 "mediapipe_ms": {
  "median": 28.288292000070214,
  "p95": 29.327466356335208
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-19T06:53:28+00:00"
}
```
