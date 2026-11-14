# Step 6b — Webcam evaluation with an unseen signer

## Status update (2026-11-14)

- Real webcam sessions **have been run**: 10 camera sessions (180 attempts) on 2026-10-19 and 2026-10-20, all with one
  signer (S01) and 10 of the 100 signs (`reports/step6/webcam_eval/*_camera_S01/`). Sessions 1–3 are
  analysed below; the later blocks are summarised in the frozen replay set
  (`reports/e0/webcam_replay_set/`).
- Candidate on the replay set (92 core attempts): top-1 0.402, top-5 0.587
  (`reports/e0/webcam_replay_set/summary_by_model.csv`).
- Still **not established**: accuracy for an unseen signer, or for the other 90 signs. S01's sessions
  were inspected during development, so they are development data, not a held-out test
  (`docs/evaluation_protocol.md`).
- The 2026-10-19 status below is kept as the historical record of what was known at that point.

## Status (2026-10-19, superseded)

- **Not yet performed with a real webcam and human signer.** In the automated environment the
  camera is blocked by macOS privacy permission (OpenCV: `not authorized to capture video (status 0)`;
  `s6_01 --camera 0` exits with "cannot read from camera 0 (permission or device problem)"). This is a
  permission state, not a code defect; granting it is a user decision.
- What was verified automatically: the camera command fails safely without a camera; the guided
  evaluation tool (`scripts/s6_03_webcam_eval.py`) runs end to end in **simulation mode** (a recorded
  clip replayed in real time as a stand-in camera); scoring, delay measurement and frame-dropping
  logic have unit tests.

## Manual procedure (needs a person at the Mac)

1. **Camera permission.** System Settings → Privacy & Security → Camera → enable the app that will run
   the command (e.g. Terminal or iTerm), then quit and reopen that app. Close other apps using the camera.
2. **Pick the camera.** Index 0 is not guaranteed to be the built-in camera (this Mac also exposes a
   Continuity iPhone camera). Check with the plain recognizer for a few seconds; `--preview` on the
   evaluation tool shows the image:
   ```bash
   cd model
   PYTHONPATH=src .venv/bin/python scripts/s6_01_run_inference.py --camera 0 --model-dir models/candidates/hands_pose_w30_bilstm_e100_s3 --max-seconds 15
   ```
3. **Signer.** Someone who is not in the dataset (e.g. you). Use a pseudonym (`--signer-id S01`); no
   video or images are saved, only predictions, timings and landmark-derived visibility fractions.
4. **Signs.** By default the tool prompts every 10th class of the vocabulary (a fixed rule, not chosen
   from model results): africa, blind, computer, director, forbid, here, more, please, shower, tuesday
   — 2 attempts each (`--attempts`). Look up each sign beforehand (the dataset's dominant source is
   ASL Citizen-style dictionary recordings); a sign the signer performs differently from the dataset is
   itself a finding — note it.
5. **Run one session per condition** (same signs, change one condition at a time):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/s6_03_webcam_eval.py --camera 0 --preview \
       --model-dir models/candidates/hands_pose_w30_bilstm_e100_s3 \
       --signer-id S01 --lighting "daylight, frontal" --camera-position "laptop, arm's length, upper body visible"
   ```
   Suggested conditions: (A) good light, frontal, upper body and both hands in frame; (B) dim / backlit;
   (C) camera off-axis or closer so hands leave the frame at times. During each attempt: hands down during
   "get ready", sign once on "SIGN NOW", then lower the hands and keep them down.
6. **Results** are written to `reports/step6/webcam_eval/<UTC stamp>_camera_<signer>/`:
   `attempts.csv` (expected, predicted, confidence, top-5 hit, outcome correct/wrong/rejected/
   low_confidence/missed, rejection reason, delay after the hands came to rest, duplicates, fraction of
   frames with a visible hand), `events.jsonl` (every segment), `summary.json`/`summary.md` (accuracy,
   duplicates, rejections, delay median/p95, effective camera fps, dropped frames, MediaPipe time). Write
   observed effects of lighting/position into `--notes`.

Reading the results: compare accuracy to the offline test accuracy (64 %); look at `capture_fps_median`
and `dropped_frames` (see the frame-rate risk below), at rejection reasons (hand visibility vs
`too_few_decoded_frames`), and at whether the delay stays flat across the session (growing delay would
mean the camera buffers stale frames - not observable without a camera).

## Simulation results (pipeline check only — NOT real-world, NOT an unseen signer)

Three dev100 test clips (bottle, fruit, calm) replayed in real time through the same tool and settings:

| clip | outcome | confidence | delay after hands rest | processed / total frames | effective fps | segment frames |
|---|---|---|---|---|---|---|
| bottle | correct | 0.999 | 0.49 s | 102 / 133 | 23.0 | 81 |
| fruit | correct | 0.968 | 0.48 s | 154 / 172 | 26.9 | 102 |
| calm | correct | 0.536 | 0.43 s | 45 / 63 | 21.4 | 33 |

These clips are from the dataset's own test split, so their accuracy says nothing about unseen signers.
What they do show:

- **Delay** from the end of signing to the label ≈ 0.43–0.49 s, i.e. the `end_rest_s` wait (0.4 s) plus
  processing.
- **Frame-rate risk.** MediaPipe Holistic needs ~28.5 ms per frame on this M2; with the rest of the loop
  the stream ran at 21–27 fps and 10–29 % of frames were skipped. The training rule rejects segments with
  fewer than 30 frames (`too_few_decoded_frames`); the `calm` segment had 33. On a slower or busier
  machine (e.g. during a video call) short signs could be rejected. Not observed as a failure, so no
  default was changed; if it appears in the webcam sessions, raising `segmentation.pre_roll_s` (e.g.
  0.5 → 1.0) adds rest frames to the segment without changing the model input (the model only sees the
  active signing interval).

## Confidence calibration (`scripts/s6_04_confidence_calibration.py`, `reports/step6/calibration_*.json`)

| | validation (725) | test (706) |
|---|---|---|
| accuracy | 67.6 % | 64.0 % |
| mean confidence | 0.768 | 0.757 |
| expected calibration error (15 bins) | 0.093 | 0.119 |
| median confidence, correct / wrong | 0.954 / 0.551 | 0.963 / 0.544 |
| wrong predictions with confidence ≥ 0.8 | 48 | 36 |

The model is over-confident, but confidence separates right from wrong reasonably well. Withholding
predictions below a threshold trades coverage for precision (validation: t = 0.6 → 74 % coverage, 80.9 %
accurate; t = 0.8 → 56 %, 88.1 %). The lowest threshold that reaches 80 % precision **on validation** is
0.57; on test it gives 74.9 % coverage and 77.7 % accuracy of accepted predictions (a check, not a fit).
`configs/inference.yaml` still defaults to 0.0 (pass-through); a threshold should be confirmed on the
webcam sessions before being made the default.

## Readiness assessment

Ready: the inference path itself — same preprocessing as training (bit-identical features on 20 real
clips), validated loading, safe failure modes, ~1.7 ms model time, ~0.45 s label delay in simulation.

Not established: recognition quality for signers outside the dataset. Offline evidence argues for
caution: 64 % test accuracy on a split that is not signer-independent, ~32-point train–validation gap,
over-confident errors. Until the webcam sessions are run, the model should be integrated only as an
explicitly experimental, word-level caption with confidence shown — not as a dependable translator —
and `models/best_model.pt` is left unchanged.

## Session S01 (2026-10-19, `reports/step6/webcam_eval/20261019T071836Z_camera_S01/`) — investigation

Recorded: 20 attempts, 0 correct, 8 wrong, 10 missed, 2 rejected, mean hand-visible fraction 0.233,
capture 1920×1080 at a median 14.996 fps, 22 events of which 7 fell outside attempts. The original
files are kept unchanged.

**Evaluation correctness.** The report has no prompt times, but the fixed schedule (3 s WAIT + 5 s
GO per attempt) reproduces its matching exactly for a first prompt at 1.2–2.1 s after stream start and
≤ 0.1 s loop overrun per cycle (909 parameter sets). Under that reconstruction:
- 6 of the 7 unmatched events ended during a WAIT phase, i.e. the signer signed as soon as the word
  appeared, before GO; the following GO windows were often empty;
- the 1.5 s grace after an attempt overlapped the next WAIT phase, so 2 events (attempts 1 and 11)
  were credited to the previous sign although they may answer the next one;
- re-scoring under any assignment (strict, grace, or WAIT-phase events counted for the upcoming sign)
  still gives 0 correct and **no event with the expected word in its top-5**.

**Hand visibility.** `hand_visible_fraction` = frames with any hand detected / frames in the GO window,
averaged over attempts — implementation and aggregation are correct (0.233 = 4.66 / 20). Ten attempts
had 0 or 1 hand frames in the whole 5 s window (attempts 2–5, 8–10, 12, 18–19); there were no hand
detections at all from ~21 s to ~50 s and from ~66 s to ~90 s. Whether the hands were out of frame,
undetected, or simply not raised cannot be told from the report (no images or landmarks were saved).

**Frame capture.** No resolution or fps was requested, so the camera delivered 1920×1080. MediaPipe
took a median 30.0 ms/frame in the session; offline, the loop's MediaPipe + colour conversion + preview
preparation costs ~29 ms at 1080p, 720p and 640×480 alike, so processing alone allows ~34 fps. The
measured 14.996 fps is exactly half of 30 fps: either the loop (MediaPipe + 1080p preview window + key
polling) slightly exceeded one camera frame period so every second frame was skipped, or the camera
itself ran at 15 fps (exposure). The report's `dropped_frames: 0` was not a measurement (the camera
source cannot observe drops).

**Segmentation.** The 10 missed attempts are input failures — no hand run of 3 frames in the window,
so no segment could start. Of 6 rejected segments (2 counted as attempt outcomes), 5 were
`too_few_decoded_frames` with 16–29 frames: at 15 fps a 1.1–1.9 s segment has < 30 frames (the
training rule), while at 30 fps it would have 32–58. One was `active_both_hands_missing_fraction`.
Four segments were cut at `max_segment_s` (hands kept up for > 4 s between signs).

**Preprocessing parity (code path).** Camera frames are BGR→RGB converted (tested), then go through
the same `holistic_result_to_frame` → `StreamingRecognizer` → `preprocess_raw` (pixel space with the
camera's own width/height, shoulder normalization, 30 FPS resampling by timestamps) →
`resample_interval` → `assemble_features`, with the checkpoint-verified label mapping. No code
difference from training was found. Input-domain differences that the code cannot remove: 16:9 1080p
framing instead of the dataset's 640×480, 15 instead of ~30 fps, a single Holistic instance for the
whole session, unknown handedness/framing of the signer, and dictionary-style signing in the dataset.

**Prediction quality.** The candidate's top-5 accuracy is 83.7 % on test (85.7 % on validation; note
added 2026-11-14: these two figures are not stored in any saved report, since the saved predictions
have no probabilities, so they cannot be re-checked from the repository). With
in-distribution inputs, 0 of 8 answered attempts having the expected word in the top-5 has probability
≈ 5 × 10⁻⁷, so the result reflects a systematic input/protocol mismatch, not ordinary classifier error.
Live features cannot be compared with training features for S01 because no landmarks were recorded.

**Confirmed bugs (fixed, with regression tests):** (1) prompt times were not recorded, so the matching
could not be audited; (2) the grace window overlapped the next WAIT phase; WAIT-phase events were
dropped instead of reported; (3) `dropped_frames` was reported as 0 for the camera although it is not
observable. **Protocol change:** prompts now say WAIT (do not sign yet) / GO, the overlay colours them
differently, and early (WAIT-phase) signs are reported per attempt. **Instrumentation added:**
`frames.csv` (per-frame timestamps, read wait, MediaPipe time, pose/left/right presence, shoulder width
as a fraction of frame width), optional `landmarks.npz` (`--save-landmarks`; landmarks only, no
images), input diagnostics next to the training reference (shoulder width / frame width median
0.371), and the camera now requests 640×480 at 30 fps (`--width/--height/--fps`), the dataset's format.

**Hypotheses still open (need the next session's frames.csv / landmarks):** hands outside the frame or
too small at 1080p arm's length; signer handedness or sign variants differing from the dataset; 15 fps
from loop timing vs camera exposure; whether any of these explains the 0 % top-5.

Is the 0 % trustworthy? As a description of this session's end-to-end behaviour, yes — no matching
rule recovers a correct or top-5 prediction. As a measure of the classifier on an unseen signer, no:
half the windows had no detected hands, the frame rate was half the training rate, and the inputs were
not recorded.

## Session 2 (2026-10-19, `reports/step6/webcam_eval/20261019T075328Z_camera_S01/`) — investigation

Recorded (legacy first-event scoring): 8/20 correct, 9 wrong, 3 missed; 47.1 % accuracy and 76.5 % top-5
among answered attempts; any-hand visibility 0.724; 15.0 processed fps; 14 answers flagged as started before
GO; 7 attempts with duplicates. Raw files are unchanged (checksummed); analyses are written to new
`replay_*` subfolders.

**1. Capture (confirmed by measurement).** The camera delivered 640×360 (16:9; 640×480 was requested and is
not offered) and runs at 30 fps: 2,410 of 2,432 frame intervals are exactly 2 camera periods (66.6 ms).
Per loop, MediaPipe took 31.4 ms and the rest 12.1 ms (median); `imshow` + `waitKey(1)` of the preview
alone costs 13.4 ms on this Mac (everything else < 0.03 ms). 43.5 ms of work > 33.3 ms period in 99.5 % of
loops, so each `cap.read()` waited ~23 ms for the next frame and every second frame was skipped.
Fix: `--threaded-capture` (default on) reads frames in a background thread, timestamps them on arrival,
always processes the newest frame and counts skipped frames (`dropped_frames`, `camera_fps_measured`);
`--preview-every N` (default 3) redraws the preview every N frames. Expected loop ≈ 31.4 + 13.4/3 ≈ 36 ms
(~28 fps) — **not yet measured with the camera**.

**2. Evaluation timing (confirmed).** `started_before_go` compared the segment start — which includes the
0.5 s pre-roll — with GO; the flag now uses the segment's first hand frame (`SignEvent.first_hand_s`).
Recomputed from `frames.csv`, all 14 flagged answers really had hands up before GO (0.2–2.1 s), and in 13 of
17 answers hands were visible during the last 0.5 s before GO: hands were visible in 48 % of WAIT frames.
Resting hands stayed inside the 16:9 frame, so segments started during WAIT and ran into the sign; 11 of 17
answers were cut at `max_segment_s` (4 s). Fixes: GO is gated — it appears only after no hand has been seen
for `--hands-down-s` (0.5 s; up to `--max-wait-s` 6 s, else flagged `hands_down_at_go=False`); the preview
turns red with "lower your hands out of view" during WAIT; a 5 s framing check precedes the first prompt.
Strict scoring (now the default): an attempt's answer is the first non-suppressed segment whose first hand
frame is at/after GO; earlier segments are `pre_go_events` (outcome `pre_go_only` if nothing else), and the
old rule is kept as `legacy_*`.

**3. Framing (measured).** Shoulder width / frame width: median 0.225 vs training median 0.371
(interquartile 0.257–0.421). Part of the gap is the 16:9 frame (a 4:3 crop of the same image would give
≈ 0.30); the rest is distance. A framing line in the preview and the framing check now say "move closer" /
"OK" / "move back" against the training interquartile range. Cropping to 4:3 was not added (no evidence yet
that it helps; it could cut off wide signs).

**4. Duplicates (measured).** Two causes: forced cuts splitting one continuous hand-visible stretch
(attempts 5, 6, 7, 16) and separate hand bursts within a window (1, 10, 11). An optional `cooldown_s`
(StreamingRecognizer / `postprocess.cooldown_s`, default 0 = off) marks a segment that starts within
`cooldown_s` of the previous one as `suppressed` (classified, not emitted). Replayed on this session with
1.0 s it removes all duplicates but also suppresses real signs (strict correct 4 → 3, missed 3 → 6),
because with hands always in view nearly every segment follows the previous one within a second. **Not
recommended to enable**; the real fix is hands out of view between signs (gated GO + framing).

**5. Re-evaluation by replay (`scripts/s6_05_replay_session.py`).** Replaying `landmarks.npz` with the
session's settings reproduces all 39 recorded events exactly (labels, confidences, frames, timing) —
live/offline parity of the recognition path on real webcam data. Strict re-scoring
(`replay_default/`): 4 correct, 5 wrong, 7 pre_go_only, 3 missed, 1 rejected. **Superseded** — see Session 3 §6: events were keyed on their last hand
frame, and continuations of a forced cut counted as clean; corrected re-score in `replay_firsthand/`.

| attempts (20) | category |
|---|---|
| 7 | segmentation/protocol: hands up before GO — rest and sign mixed, not scorable |
| 2 | protocol: signed during WAIT, nothing after GO (both WAIT segments were recognised as other words) |
| 1 | segmentation: no hand run after GO |
| 1 | capture: `too_few_decoded_frames` (15 fps) |
| 4 | correct on a clean (post-GO) segment |
| 5 | **classifier errors on clean segments** (expected word in top-5 in 3 of them) |

Of 9 clean segments, 4 were correct and 7 had the expected word in the top-5; 4 WAIT-phase segments were
recognised as the upcoming sign (africa, more, please, shower) but are correctly not credited. With 9 clean
segments from one signer no accuracy estimate is meaningful yet. No real-time claim is made: the capture fix
has not been measured with the camera.

## Session 3 (2026-10-19, `reports/step6/webcam_eval/20261019T084115Z_camera_S01/`) — analysis

Threaded capture, preview every 3rd frame, gated GO, framing check; same model
(`hands_pose_w30_bilstm_e100_s3`), threshold 0, cooldown 0. Raw files unchanged (checksummed).

**1. Capture (measured).** Camera 30.0 fps measured (5,300 frames captured), 5,270 processed (29.8 fps),
28 dropped (0.5 %). Frame interval median 33.5 ms, p95 38.1 ms, 0.15 % > 50 ms (the 300 ms maximum is at
start-up). MediaPipe median 28.1 ms (p95 28.4). `read_wait` median 0.08 ms: the loop now has almost no
slack, so heavier per-frame work would start dropping frames again. The preview's cost is not separately
logged in this session. The camera delivered 640×480 this time (640×360 in session 2).

**2. GO gate (measured).** In all 20 attempts no hand was visible in the 0.5 s before GO
(`hands_down_at_go` true for all; the gate added 0.7–3.3 s in 5 attempts). Hands visible in 21.7 % of
WAIT frames (48 % in session 2). In 6 attempts the sign was rehearsed during WAIT and then lowered
(3 of those recognised as the upcoming sign); they are early events, not credited. 0 pre-GO answers.

**3. Segmentation (measured).** Hands appeared 0.7–1.8 s after GO, but stayed visible to the end of the
5 s window in 12/20 attempts, so 11/19 live answers were cut at `max_segment_s`; 9 of those produced a
continuation segment (duplicates — they persist with cooldown off, as expected). Delay: 0.04 s median for
forced cuts (n=11, emitted at the cut), 0.46 s for segments closed by a rest (n=8, ≈ `end_rest_s`).
Correct: 7/8 rest-closed vs 6/11 forced — confounded with which signs were forced (computer, forbid, here);
a hypothesis only.

**4. Framing (measured).** Shoulder/frame width median 0.255 during attempts (training interquartile
0.257–0.421, median 0.371): at the lower edge; the framing line said "move closer" at the end.

**5. Results (strict, after §6).** 13/20 correct, 7 wrong, 0 missed/rejected/pre-GO; top-5 14/20.
Errors: computer → church ×2, forbid → church ×2, here → impossible/scared, director → awake ('church'
was predicted in 5 attempts, never expected; computer → church also in session 2). One signer, 10 signs ×
2 attempts: 13/20 has a 95 % Wilson interval of ≈ 45–81 %; not a reliable accuracy estimate.

**6. Scoring fix.** (a) Events were attached to attempts by last hand frame; the next prompt's WAIT starts
right at the end of the GO window, so a sign started inside the window but cut past its end (and its tail)
landed in the next WAIT. Session 3 attempt 8 was scored `missed` although its sign was recognised as
'church', and 8 tails were reported as WAIT-phase signing. (b) After a forced cut the continuation's
`first_hand_s` is the frame after the cut, so in session 2 continuations of a hand run that began in
WAIT were scored as clean answers. Fix: `SignEvent.run_start_s` (start of the uninterrupted hand run; a
continuation within `end_rest_s` inherits it); an event whose run started inside a GO window belongs to
that attempt; otherwise the last-hand rule applies (pre-GO segments ending after GO stay `pre_go`). Events
without a recorded first hand frame (session 1 logs) keep the old rule. Re-scored by replay
(`replay_firsthand/`): session 3 13 correct / 6 wrong / 1 missed → 13 / 7 / 0; session 2 4 correct /
5 wrong / 7 pre_go_only / 3 missed / 1 rejected → 8 / 6 / 6 / 0 / 0 (attempt 6 'director', previously
credited, was a continuation of a WAIT hand run).

