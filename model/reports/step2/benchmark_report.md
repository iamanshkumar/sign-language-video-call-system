# Step 2 Phase A - MediaPipe Holistic benchmark

All numbers are measured by `scripts/s2_03_benchmark_report.py` (`reports/step2/benchmark_stats.json`). Full-dataset figures are extrapolations from this benchmark.

## Benchmark set

- 500 videos from the frozen TRAIN split, 465 classes (`data/landmarks/benchmark/benchmark_sample.csv`).
- Stopped early at user request (Mac heating): attempted 400 videos (383 classes), first by sample_id order; by selection reason: {'frame_count_quintile_2': 77, 'frame_count_quintile_3': 75, 'frame_count_quintile_4': 72, 'frame_count_quintile_1': 69, 'frame_count_quintile_5': 67, 'duplicate_group': 15, 'augmented': 11, 'short_clip': 7, 'non_default_resolution': 7}.
- Attempted 400, download failures 0, processed 26, rejected 374 (93.5%).
- Rejection reasons: {'both_hands_missing_fraction': 357, 'too_few_decoded_frames': 9, 'both_hands_missing_run': 8}
- Output consistency problems: 0; unreferenced array files (from the interrupted batch, ignored): 11

## Throughput

- Workers [4], wall time 10.0 min (includes download, overlapped).
- **40.2 videos/min**, **56.2 source frames/s** (33,606 frames through Holistic).
- Holistic alone: 15.5 frames/s per worker.
- RAM/CPU: not available (run stopped at user request before the resource summary was written). Wall-time source: ['reconstructed from run.log timestamps (run stopped at user request; resource monitor summary lost)'].

| per video (s) | mean | median | p95 |
|---|---|---|---|
| total_s | 5.63 | 5.11 | 11.76 |
| mediapipe_s | 5.40 | 4.91 | 11.29 |
| decode_s | 0.11 | 0.09 | 0.24 |
| post_s | 0.01 | 0.01 | 0.01 |

## Configurations

One Holistic pass produces all groups; configurations are feature selections of the same stored arrays, so extraction time is identical for all three. Differences are feature size and storage.

| configuration | groups | feature dim | float32 MB per video if stored alone |
|---|---|---|---|
| hands | left_hand, right_hand | 128 | 0.044 |
| hands_pose | left_hand, right_hand, pose | 261 | 0.090 |
| hands_pose_face | left_hand, right_hand, pose, face | 1666 | 0.578 |

## Missing landmarks (30 FPS grid, before gap filling)

| group | mean % frames missing | median % |
|---|---|---|
| left_hand | 67.1 | 63.5 |
| right_hand | 61.0 | 60.5 |
| pose | 0.3 | 0.0 |
| face | 0.7 | 0.0 |
| both_hands | 52.2 | 54.3 |

### Rest frames (hands outside the frame before/after the sign)

- Leading frames without hands: mean 16.4, median 13; trailing: mean 25.4, median 23.
- Rest share of clip: mean 48.6%, median 50.8%.
- Active signing segment (first→last hand detection): mean 44.9 frames, median 38; 100 segments shorter than 30 frames.
- WHAT-IF (not applied): hand rules evaluated only on the active segment → {'accepted': 360, 'both_hands_missing_fraction': 29, 'both_hands_missing_run': 2}; 334 current hand-rule rejections would pass.

Both-hands-missing intervals: 1153 (≤5 frames: 410, 6-15: 270, >15: 473); videos with a run >15: 327.
Interpolated frames (processed videos): {'left_hand': 62, 'right_hand': 70, 'pose': 0, 'face': 3}; zero-filled: {'left_hand': 430, 'right_hand': 404, 'pose': 2, 'face': 23}.
Frames using carried-over shoulder frame: 2; non-finite detections repaired: 0.

## Full-dataset estimates (108,616 videos)

| basis | hours (per-video rate) | hours (per-frame rate) |
|---|---|---|
| benchmark_workers | 45.1 | 45.3 |

Storage: stored NPZ (all groups, float32) 0.581 MB per saved video (measured on 391). Full dataset: **4.1 GB** at the specified rules (acceptance 6.5%), **56.8 GB** at the what-if rule (acceptance 90.0%), 63.1 GB if every video were saved. Per configuration stored alone (float32, all videos): hands 4.8 GB, hands_pose 9.8 GB, hands_pose_face 62.7 GB. Free disk now: 52.0 GB.

The per-frame estimate uses the class-size-weighted mean decoded frame count from the Step 1 sample (84.5); the benchmark over-represents some properties by design (quotas), so the per-frame estimate is the more representative one.

