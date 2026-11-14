# Live evaluation - SIMULATION (recorded clip replay) - not a real-world result

| attempt | expected | outcome | predicted | confidence | top-5 hit | delay s | dup | hand visible | reason |
|---|---|---|---|---|---|---|---|---|---|
| 0 | calm | correct | calm | 0.536 | True | 0.43 | 0 | 0.42 |  |

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
  "median": 0.43390537500800574,
  "p95": 0.43390537500800574
 },
 "confidence_correct_median": 0.5358781814575195,
 "confidence_wrong_median": null,
 "mean_hand_visible_fraction": 0.4222222222222222,
 "mode": "video_replay",
 "real_world": false,
 "signer_id": "dataset-test-clip",
 "lighting": "",
 "camera_position": "",
 "notes": "SIMULATION replay of test clip part_2/23621758913828517-CALM.mp4",
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
 "frames": 45,
 "dropped_frames": 18,
 "capture_fps_median": 30.303030303030173,
 "mediapipe_ms": {
  "median": 28.600708988960832,
  "p95": 29.907133406959474
 },
 "events_outside_attempts": 0,
 "created_utc": "2026-10-19T06:53:39+00:00"
}
```
