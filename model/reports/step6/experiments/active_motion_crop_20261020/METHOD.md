# Active-motion cropping — exploratory offline experiment (2026-10-20)

**Predeclared before any cropped result was computed.** Settings below are fixed; all of them are
reported, none is chosen afterwards.

## Question
Does removing the hand-raising and hand-lowering portions of a live webcam segment change recognition,
by making the segment's active interval temporally closer to the training sequences (training active
interval median 1.2 s; webcam hand runs 2.7–4.6 s)?

## Data and model (unchanged)
- Sessions: `reports/step6/webcam_eval/20261019T084115Z_camera_S01` (S3), `20261020T045903Z_camera_S01` (S4);
  saved `landmarks.npz` (per-frame MediaPipe output), prompt times from `attempts.csv`.
- Model: `models/candidates/hands_pose_w30_bilstm_e100_s3/` (best_model.pt + class_mapping.json), loaded as
  in `scripts/s6_05_replay_session.py`; session settings (pre-roll 0.5 s, end rest 0.4 s, cap 4.0 s,
  threshold 0, smoothing 1, cooldown 0).

## Eligible attempts
Every attempt whose strict answer (current scoring, `assign_events` + `attempt_row`) is a classified
segment (status `ok`): expected 20 per session. The segment is the answer event's own segment (a forced
cut is NOT merged with its continuation), reconstructed from the landmarks as the frames with
`start_s <= t <= end_s`, after the recognizer's invalid-group repair.

## Baseline
The reconstructed segment through `SignRecognizer.predict_raw` (training preprocessing: active interval =
first >=3-frame hand run to last hand frame on the 30 fps grid, uniform resampling to 30 frames).
Validity checks: (1) replay of the whole session reproduces every recorded event (`matches_recorded`);
(2) each reconstructed baseline segment reproduces its event's label and confidence (|Δ| < 1e-5).

## Motion signal
- Active hand: the hand (left/right) observed in more frames inside the segment's hand range (tie: right).
- Height h(t): the active hand's centroid y (mean of its 21 landmarks) minus the shoulder-midline y
  (pose 11/12), converted to shoulder widths: h = (y_hand − y_sh) · H / (|x11 − x12| · W), using the
  segment's median shoulder width. Positive = below the shoulder line.
- Frames where the active hand is missing: linear interpolation in time between observed frames.
- Smoothing: centered moving average over 5 frames (≈0.17 s), edges padded with the nearest value.
- Vertical velocity v = dh/dt (shoulder widths per second); v < 0 = moving up.

## Crop rule (per threshold θ)
- Raise: if the hand is rising at the first hand frame (v < −θ), the raise lasts while v < −θ; the crop
  starts at the first frame after it. Otherwise nothing is removed at the start.
- Lower: if the hand is falling at the last hand frame (v > +θ), the lowering is the final run of frames
  with v > +θ; the crop ends at the last frame before it. Otherwise nothing is removed at the end.
- Boundaries snap inward to frames where a hand is observed.
- θ ∈ {0.5, 1.0, 2.0} shoulder widths/s (all reported). θ = ∞ (no crop) is the identity control and must
  reproduce the baseline exactly.
- Minimum retained duration 0.5 s; below it the crop is marked `crop_failed`, and that attempt keeps its
  baseline result in the aggregate (count reported).

## Boundary handling (how the crop is applied)
Frames outside the crop keep their pose but have both hands marked unobserved (coordinates zeroed), as
in training clips where hands are out of view outside the active interval. The decoded frame count is
unchanged (no artificial `too_few_decoded_frames`), and the training preprocessing then finds the
active interval = the crop. No other preprocessing or model setting changes.

## Reported
Per attempt: baseline vs cropped label, confidence, correct, expected-in-top-5, rank of expected,
crop start/end offsets, crop duration, retained fraction of the hand range, flags. Aggregates per θ, per
session and per sign; paired changes (wrong→correct, correct→wrong) with an exact two-sided sign test.
Boundary validity: automatic check (height at crop start is above the height at the first hand frame;
height at crop end is above the height at the last hand frame) and a per-attempt plot of h(t) with the
crop boundaries.

## Interpretation rules
Exploratory: 40 attempts, one signer, 10 signs. No threshold is selected by its result; a handful of
changed predictions is not evidence of improvement.
