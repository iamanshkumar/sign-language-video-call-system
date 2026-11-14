# Active-motion cropping — results (2026-10-20)

Method and settings: `METHOD.md` (written before any cropped result). Per-attempt data: `per_attempt.csv`;
aggregates: `summary.json`; crop boundaries on the height signal: `crop_boundaries_S3.png`, `crop_boundaries_S4.png`.
Analysis script: scratch-only (not in the repository). No source, model, config, data or earlier report was
modified (pre-existing files under `reports/step6/webcam_eval/` and the checkpoints re-verified by SHA-256).

## Validity checks (all passed)
- Replay of both sessions reproduces every recorded event (S3: 36, S4: 23).
- All 40 reconstructed baseline segments reproduce their recorded label and confidence.
- Identity control (θ = ∞) is identical to baseline for all 40.
- 40/40 attempts eligible (all strict answers were classified segments); 0 crops failed (minimum kept 1.30 s),
  0 rejected by the training preprocessing; 0 automatic boundary-check failures.

## Aggregate (40 attempts: S3 20, S4 20)
| variant | correct | top-5 | wrong→correct | correct→wrong | labels changed | removed s (median / max) | kept duration median | kept fraction median |
|---|---|---|---|---|---|---|---|---|
| baseline | 22 (S3 13, S4 9) | 26 | – | – | – | – | 3.36 s | 1.00 |
| θ 0.5 | 20 (12, 8) | 24 | 0 | 2 | 6 | 0.35 / 0.90 | 3.07 s | 0.89 |
| θ 1.0 | 21 (13, 8) | 25 | 0 | 1 | 4 | 0.17 / 0.63 | 3.16 s | 0.95 |
| θ 2.0 | 22 (13, 9) | 26 | 0 | 0 | 1 | 0.05 / 0.50 | 3.36 s | 0.99 |

Sign tests on the paired changes: p ≥ 0.5 for every variant (no evidence either way).
Correct → wrong: S3 #0 africa → bank (θ 0.5), S4 #17 shower → sun (θ 0.5, 1.0). Other label changes were
wrong → wrong (S3 #9, S4 #5, #6, #8, #11). Expected-word rank improved in no attempt. On unchanged labels the
confidence moved by 0.08 on average (both directions).

By sign (correct of 4, baseline → θ 0.5 / 1.0 / 2.0): africa 2 → 1/2/2, shower 4 → 3/3/4; all other signs
unchanged (blind 4, more 4, please 4, tuesday 2, director 1, here 1, computer 0, forbid 0).

## Boundary inspection (plots)
- Where the hand rises from below the frame, the rise to signing height takes about 0.2–0.35 s and the crop
  start lands at its end (e.g. S4 #2, #3, #9, #10, #12, #13); lowering at the end is similar and is removed
  where it is a single final sweep.
- Not captured by the rule: the multi-stage end of AFRICA (S4 #0/#1, S3 #0/#1: the hand drops to shoulder
  level at ~1.0–1.5 s and stays there 0.5–1.5 s before leaving), and hands that first appear already at
  signing height (several S3 segments: no rise to remove).
- Forced cuts have no lowering inside the segment, so only the start is cropped.

## Findings
**Measured.** Raise and lower phases, as defined, are short: 0.35 s removed in total per segment at the
loosest threshold, leaving a median of 3.07 s versus the 1.2 s training median. Cropping them changed no
wrong answer into a correct one in any of the three settings; the loosest setting changed two correct
answers to wrong. The height traces show where the rest of the duration is: long plateaus at signing
height (BLIND, MORE, FORBID, AFRICA's low hold) and repeated cycles (PLEASE 3–4 strokes, HERE, TUESDAY,
SHOWER), i.e. inside the sign, not in the raise/lower.

**Hypotheses (not tested here).** The webcam–training duration gap comes mainly from holding the sign and
repeating its movement, not from raising and lowering the hands. Errors on COMPUTER, FORBID, HERE,
DIRECTOR and TUESDAY are not explained by raise/lower content. The height signal is vertical only;
horizontal and hand-shape motion were not used.

## Limitations
One signer, 10 signs, 40 attempts; forced-cut segments were not merged with their continuations; a
single vertical-height signal; the thresholds are a small predeclared range, not tuned; two sessions with
different signing behaviour.

## Recommendation
Raise/lower cropping does not deserve a controlled live test on this evidence: it removes too little of
the segment to address the duration gap and produced no improvement. If duration is pursued further, the
question to test is hold/repetition, which needs either a protocol arm (sign each word once, at a
natural pace, without holding) or an offline analysis of within-sign motion — both to be agreed first.
